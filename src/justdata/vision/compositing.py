"""Paste annotated instances into a frame and update its dense labels.

``composite_instances`` places each requested instance by its anchor (the
bottom centre of its crop, at the waterline), checks the placement constraints,
blends it, and writes its class where the final alpha is at least 0.5. Every
instance is either placed or rejected with counted reasons; nothing is
substituted. An instance without a class is a null paste: it takes background
pixels from another allowed location of the same frame, goes through the same
placement and blending, and writes no label, which measures how much a model
reacts to compositing seams alone (Dwibedi et al., ICCV 2017; Blum et al.,
Fishyscapes, IJCV 2021).

Placement draws use stateless Philox seeds folded from ``seed``, the instance
index, and the attempt index. Compositing runs eagerly.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import tensorflow as tf

from justdata.vision.colorspace import lab_to_rgb, rgb_to_lab
from justdata.vision.corruptions.registry import draw_seed
from justdata.vision.filters import gaussian_blur_hwc
from justdata.vision.horizon import Horizon, horizon_coordinates
from justdata.vision.poisson import clone_bounds, clone_region

COMPOSITE_VERSION = "1.0.0"
LABEL_ALPHA_THRESHOLD = 0.5
REFERENCE_WIDTH = 1024
PLACEMENT_SCALES = ("keep_row", "horizon")
BLENDING_MODES = ("hard", "feather", "poisson")
_NO_ANCHOR = "no_allowed_anchor"
_FLAT_LAB_STD = 1e-3


@dataclass(frozen=True, eq=False)
class Instance:
    """An annotated object crop to paste.

    ``image`` is ``[h, w, 3]`` uint8 and ``alpha`` ``[h, w]`` bool or float in
    ``[0, 1]``. The crop's bottom row is the waterline, at
    ``anchor_row_fraction`` of the source image height. ``class_id=None`` makes
    a null paste. ``source_horizon_offset_px`` is the distance in source pixels
    from the source horizon down to the waterline, needed for horizon scaling.
    """

    image: Any
    alpha: Any
    class_id: int | None
    source_id: str
    anchor_row_fraction: float
    source_horizon_offset_px: float | None = None


@dataclass(frozen=True, slots=True)
class PlacementRule:
    """Where and at what scale instances may be placed.

    ``keep_row`` pastes at scale 1 with the anchor on the row at the instance's
    source row fraction. ``horizon`` scales by the ratio of the anchor's
    distance below the target horizon to ``source_horizon_offset_px``. A
    placement is rejected when more than ``max_outside_allowed_share`` of its
    label footprint covers non-allowed pixels, when it shows fewer than
    ``min_visible_pixels``, when it overlaps an earlier paste (unless
    ``allow_overlap``), or when its box comes closer than ``border_margin_px``
    to the image border.
    """

    scale: str = "keep_row"
    max_outside_allowed_share: float = 0.05
    min_visible_pixels: int = 64
    allow_overlap: bool = False
    border_margin_px: int = 0
    max_attempts: int = 50

    def __post_init__(self) -> None:
        if self.scale not in PLACEMENT_SCALES:
            raise ValueError(f"PlacementRule scale must be one of {PLACEMENT_SCALES}.")
        if not 0.0 <= self.max_outside_allowed_share <= 1.0:
            raise ValueError("max_outside_allowed_share must be in [0, 1].")
        if self.min_visible_pixels < 1 or self.border_margin_px < 0:
            raise ValueError(
                "min_visible_pixels must be positive and border_margin_px non-negative."
            )
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "scale": self.scale,
            "max_outside_allowed_share": self.max_outside_allowed_share,
            "min_visible_pixels": self.min_visible_pixels,
            "allow_overlap": self.allow_overlap,
            "border_margin_px": self.border_margin_px,
            "max_attempts": self.max_attempts,
        }


@dataclass(frozen=True, slots=True)
class BlendingRule:
    """How a placed instance is blended.

    ``hard`` uses the binary alpha, ``feather`` blurs it with a Gaussian of
    ``feather_sigma_px``, and ``poisson`` clones its gradients like OpenCV's
    ``seamlessClone(NORMAL_CLONE)``. Independently, harmonization moves the
    instance's CIELAB mean and standard deviation toward those of a background
    ring ``harmonization_ring_px`` wide, by ``harmonization_strength``. Pixel
    sizes refer to an image 1024 pixels wide and scale with the image width.
    """

    mode: str = "hard"
    feather_sigma_px: float = 2.0
    harmonization_strength: float = 0.0
    harmonization_ring_px: float = 8.0

    def __post_init__(self) -> None:
        if self.mode not in BLENDING_MODES:
            raise ValueError(f"BlendingRule mode must be one of {BLENDING_MODES}.")
        if self.feather_sigma_px <= 0 or self.harmonization_ring_px <= 0:
            raise ValueError(
                "feather_sigma_px and harmonization_ring_px must be positive."
            )
        if not 0.0 <= self.harmonization_strength <= 1.0:
            raise ValueError("harmonization_strength must be in [0, 1].")

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "feather_sigma_px": self.feather_sigma_px,
            "harmonization_strength": self.harmonization_strength,
            "harmonization_ring_px": self.harmonization_ring_px,
        }


@dataclass(frozen=True, slots=True)
class CompositeDescriptor:
    """Versioned, serializable semantics of one compositing configuration."""

    placement: PlacementRule
    blending: BlendingRule
    version: str = COMPOSITE_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "placement": self.placement.to_dict(),
            "blending": self.blending.to_dict(),
            "label_alpha_threshold": LABEL_ALPHA_THRESHOLD,
            "reference_width": REFERENCE_WIDTH,
            "anchor": "crop_bottom_row_and_centre_column",
            "draws": "stateless_fold_in_seed_instance_then_attempt_philox",
            "resize": "bilinear_antialias_round_size",
            "feather": "gaussian_blur_reflect_101_of_binary_alpha",
            "poisson": "opencv_normal_clone_dst1_float64_clip_truncate",
            "harmonization": "cielab_mean_std_toward_square_dilation_ring",
            "label_write": "final_alpha_at_least_threshold_and_valid",
        }

    @property
    def identity(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return f"composite@{self.version}:sha256:{digest}"


@dataclass(frozen=True, slots=True)
class PasteRecord:
    """Outcome of one requested instance.

    ``box`` is ``(top, left, bottom, right)`` with exclusive ends;
    ``visible_pixels`` counts the instance's pixels in the final label (or null)
    map; ``rejections`` counts failed attempts by reason.
    """

    source_id: str
    class_id: int | None
    status: str
    reason: str | None
    attempts: int
    rejections: tuple[tuple[str, int], ...]
    box: tuple[int, int, int, int] | None
    anchor: tuple[int, int] | None
    null_source_anchor: tuple[int, int] | None
    scale: float | None
    visible_pixels: int
    instance_id: int | None
    blending: tuple[tuple[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "class_id": self.class_id,
            "status": self.status,
            "reason": self.reason,
            "attempts": self.attempts,
            "rejections": dict(self.rejections),
            "box": None if self.box is None else list(self.box),
            "anchor": None if self.anchor is None else list(self.anchor),
            "null_source_anchor": (
                None
                if self.null_source_anchor is None
                else list(self.null_source_anchor)
            ),
            "scale": self.scale,
            "visible_pixels": self.visible_pixels,
            "instance_id": self.instance_id,
            "blending": dict(self.blending),
        }


@dataclass(frozen=True, eq=False)
class CompositeResult:
    image: tf.Tensor
    mask: tf.Tensor
    instance_map: tf.Tensor | None
    pasted: tf.Tensor
    null_pasted: tf.Tensor
    records: tuple[PasteRecord, ...]
    identity: str


@dataclass(frozen=True)
class _Placement:
    anchor: tuple[int, int]
    scale: float
    box: tuple[int, int, int, int]
    canvas: tuple[int, int, int, int]
    alpha: np.ndarray
    footprint: np.ndarray
    rgb: np.ndarray
    null_source_anchor: tuple[int, int] | None


class _Scene:
    """Mutable state of one compositing call."""

    def __init__(self, image, mask, allowed, valid, instance_map, horizon):
        self.original = image.astype(np.float32)
        self.canvas = self.original.copy()
        self.original_mask = mask
        self.mask = mask.copy()
        self.original_instances = instance_map
        self.instances = None if instance_map is None else instance_map.copy()
        self.allowed = allowed
        self.valid = valid
        self.horizon = horizon
        self.height, self.width = mask.shape
        self.pixel_scale = self.width / REFERENCE_WIDTH
        self.pasted = np.zeros_like(allowed)
        self.null_pasted = np.zeros_like(allowed)
        self.occupied = np.zeros_like(allowed)
        self.owner = np.full(mask.shape, -1, np.int64)
        self.anchors = np.argwhere(allowed)
        self.next_instance_id = (
            None if instance_map is None else int(instance_map.max()) + 1
        )


def composite_instances(
    image: Any,
    mask: Any,
    instances: list[Instance] | tuple[Instance, ...],
    *,
    allowed: Any,
    placement: PlacementRule,
    blending: BlendingRule,
    seed: tf.Tensor,
    horizon: Horizon | None = None,
    instance_map: Any = None,
    valid: Any = None,
) -> CompositeResult:
    """Paste ``instances`` into ``image`` (HWC uint8) and its class map ``mask``.

    ``allowed`` (HW bool) is where an anchor may fall; the caller chooses it,
    from labels or not. ``valid`` (HW bool) marks pixels that may be
    overwritten. ``instance_map`` (HW integer) receives new IDs from its maximum
    plus one. ``horizon`` is required for ``PlacementRule(scale="horizon")``.
    """
    if not tf.executing_eagerly():
        raise RuntimeError("composite_instances runs eagerly, outside tf.function.")
    scene = _validated_scene(image, mask, allowed, valid, instance_map, horizon)
    seed = _validated_seed(seed)
    if placement.scale == "horizon" and horizon is None:
        raise ValueError("Horizon scaling needs the target horizon.")

    records = []
    for index, instance in enumerate(instances):
        records.append(
            _paste(scene, index, instance, placement, blending, draw_seed(seed, index))
        )
    records = [
        replace(record, visible_pixels=int(np.sum(scene.owner == index)))
        for index, record in enumerate(records)
    ]
    descriptor = CompositeDescriptor(placement=placement, blending=blending)
    image_out = np.rint(np.clip(scene.canvas, 0.0, 255.0)).astype(np.uint8)
    return CompositeResult(
        image=tf.constant(image_out),
        mask=tf.constant(scene.mask),
        instance_map=None if scene.instances is None else tf.constant(scene.instances),
        pasted=tf.constant(scene.pasted),
        null_pasted=tf.constant(scene.null_pasted),
        records=tuple(records),
        identity=descriptor.identity,
    )


def _validated_scene(image, mask, allowed, valid, instance_map, horizon) -> _Scene:
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("image must be HWC uint8 RGB.")
    shape = image.shape[:2]
    mask = np.asarray(mask)
    if not np.issubdtype(mask.dtype, np.integer) or mask.shape != shape:
        raise ValueError("mask must be an integer HW class map matching the image.")
    allowed = np.asarray(allowed)
    if allowed.dtype != bool or allowed.shape != shape:
        raise ValueError("allowed must be a bool HW map matching the image.")
    if valid is None:
        valid = np.ones(shape, bool)
    valid = np.asarray(valid)
    if valid.dtype != bool or valid.shape != shape:
        raise ValueError("valid must be a bool HW map matching the image.")
    if instance_map is not None:
        instance_map = np.asarray(instance_map)
        if (
            not np.issubdtype(instance_map.dtype, np.integer)
            or instance_map.shape != shape
        ):
            raise ValueError(
                "instance_map must be an integer HW map matching the image."
            )
    if horizon is not None and not isinstance(horizon, Horizon):
        raise TypeError("horizon must be a justdata.vision.horizon.Horizon.")
    return _Scene(image, mask, allowed, valid, instance_map, horizon)


def _validated_seed(seed: Any) -> tf.Tensor:
    seed = tf.convert_to_tensor(seed)
    if seed.dtype != tf.int32 or seed.shape != (2,):
        raise ValueError("seed must be an int32 tensor of shape [2].")
    return seed


def _instance_arrays(instance: Instance, mask_dtype) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.asarray(instance.image)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(
            f"Instance '{instance.source_id}' image must be HWC uint8 RGB."
        )
    alpha = np.asarray(instance.alpha)
    if alpha.shape != rgb.shape[:2]:
        raise ValueError(f"Instance '{instance.source_id}' alpha must match its image.")
    alpha = alpha.astype(np.float32)
    if alpha.min() < 0.0 or alpha.max() > 1.0:
        raise ValueError(f"Instance '{instance.source_id}' alpha must be in [0, 1].")
    if not 0.0 <= instance.anchor_row_fraction <= 1.0:
        raise ValueError(
            f"Instance '{instance.source_id}' anchor_row_fraction must be in [0, 1]."
        )
    if instance.class_id is not None:
        limits = np.iinfo(mask_dtype)
        if not limits.min <= instance.class_id <= limits.max:
            raise ValueError(
                f"Instance '{instance.source_id}' class_id does not fit the mask."
            )
    return rgb.astype(np.float32), alpha


def _paste(scene, index, instance, placement, blending, instance_seed) -> PasteRecord:
    rgb, alpha = _instance_arrays(instance, scene.mask.dtype)
    if (
        placement.scale == "horizon"
        and not (instance.source_horizon_offset_px or 0) > 0
    ):
        raise ValueError(
            f"Instance '{instance.source_id}' needs a positive source_horizon_offset_px "
            "for horizon scaling."
        )
    rejections: Counter[str] = Counter()
    attempts = 0
    for attempt in range(placement.max_attempts):
        attempts += 1
        draws = tf.random.stateless_uniform(
            [2], seed=draw_seed(instance_seed, attempt), alg="philox"
        ).numpy()
        proposal = _propose(scene, instance, rgb, alpha, placement, blending, draws)
        if isinstance(proposal, str):
            rejections[proposal] += 1
            if proposal == _NO_ANCHOR:
                break
            continue
        resolved = _blend(scene, proposal, blending)
        instance_id = _write_labels(scene, index, instance, proposal)
        return PasteRecord(
            source_id=instance.source_id,
            class_id=instance.class_id,
            status="placed",
            reason=None,
            attempts=attempts,
            rejections=tuple(sorted(rejections.items())),
            box=proposal.box,
            anchor=proposal.anchor,
            null_source_anchor=proposal.null_source_anchor,
            scale=proposal.scale,
            visible_pixels=0,
            instance_id=instance_id,
            blending=resolved,
        )
    return PasteRecord(
        source_id=instance.source_id,
        class_id=instance.class_id,
        status="rejected",
        reason=max(rejections, key=lambda reason: (rejections[reason], reason)),
        attempts=attempts,
        rejections=tuple(sorted(rejections.items())),
        box=None,
        anchor=None,
        null_source_anchor=None,
        scale=None,
        visible_pixels=0,
        instance_id=None,
        blending=(),
    )


def _propose(scene, instance, rgb, alpha, placement, blending, draws):
    """A checked placement, or the reason this attempt failed."""
    if placement.scale == "keep_row":
        row = min(scene.height - 1, int(instance.anchor_row_fraction * scene.height))
        columns = np.flatnonzero(scene.allowed[row])
        if columns.size == 0:
            return _NO_ANCHOR
        anchor = (
            row,
            int(columns[min(int(draws[0] * columns.size), columns.size - 1)]),
        )
        scale = 1.0
    else:
        if scene.anchors.size == 0:
            return _NO_ANCHOR
        anchor = _pick(scene.anchors, draws[0])
        assert scene.horizon is not None and instance.source_horizon_offset_px
        below, _along = horizon_coordinates(
            tf.constant([scene.horizon.left_row, scene.horizon.right_row]),
            scene.width,
            tf.constant(anchor[1] + 0.5),
            tf.constant(anchor[0] + 0.5),
        )
        if float(below) <= 0.0:
            return "above_horizon"
        scale = float(below) / instance.source_horizon_offset_px

    height = max(1, round(scale * alpha.shape[0]))
    width = max(1, round(scale * alpha.shape[1]))
    top, left = anchor[0] - height + 1, anchor[1] - width // 2
    box = (top, left, top + height, left + width)
    margin = placement.border_margin_px
    if (
        top < margin
        or left < margin
        or box[2] > scene.height - margin
        or box[3] > scene.width - margin
    ):
        return "outside_margin"

    scaled_rgb, scaled_alpha = _resized(rgb, alpha, (height, width))
    pad = _canvas_pad(blending, scene.pixel_scale)
    canvas = (
        max(0, top - pad),
        max(0, left - pad),
        min(scene.height, box[2] + pad),
        min(scene.width, box[3] + pad),
    )
    offset = (top - canvas[0], left - canvas[1])
    canvas_shape = (canvas[2] - canvas[0], canvas[3] - canvas[1])
    binary = np.zeros(canvas_shape, np.float32)
    binary[offset[0] : offset[0] + height, offset[1] : offset[1] + width] = (
        scaled_alpha >= LABEL_ALPHA_THRESHOLD
    )
    if blending.mode == "feather":
        sigma = blending.feather_sigma_px * scene.pixel_scale
        final_alpha = gaussian_blur_hwc(binary[..., None], sigma, sigma).numpy()[..., 0]
    else:
        final_alpha = binary
    window = (slice(canvas[0], canvas[2]), slice(canvas[1], canvas[3]))
    final_alpha = final_alpha * scene.valid[window]
    footprint = final_alpha >= LABEL_ALPHA_THRESHOLD

    covered = int(footprint.sum())
    if covered == 0:
        return "too_small"
    outside = np.sum(footprint & ~scene.allowed[window]) / covered
    if outside > placement.max_outside_allowed_share:
        return "outside_allowed"
    if not placement.allow_overlap and np.any(footprint & scene.occupied[window]):
        return "overlap"
    if covered < placement.min_visible_pixels:
        return "too_small"

    null_source_anchor = None
    if instance.class_id is None:
        null_source_anchor = (
            _pick(scene.anchors, draws[1]) if scene.anchors.size else None
        )
        source = _null_source(scene, anchor, null_source_anchor, canvas, box)
        if source is None:
            return "no_null_source"
        canvas_rgb = source
    else:
        canvas_rgb = np.pad(
            scaled_rgb,
            [
                [offset[0], canvas_shape[0] - offset[0] - height],
                [offset[1], canvas_shape[1] - offset[1] - width],
                [0, 0],
            ],
            mode="edge",
        )
    return _Placement(
        anchor=anchor,
        scale=scale,
        box=box,
        canvas=canvas,
        alpha=final_alpha,
        footprint=footprint,
        rgb=canvas_rgb,
        null_source_anchor=null_source_anchor,
    )


def _pick(anchors: np.ndarray, draw: float) -> tuple[int, int]:
    row, column = anchors[min(int(draw * len(anchors)), len(anchors) - 1)]
    return int(row), int(column)


def _resized(rgb, alpha, size) -> tuple[np.ndarray, np.ndarray]:
    if size == alpha.shape:
        return rgb, alpha
    stacked = np.concatenate([rgb, alpha[..., None]], axis=-1)
    resized = tf.image.resize(stacked, size, method="bilinear", antialias=True).numpy()
    return resized[..., :3], np.clip(resized[..., 3], 0.0, 1.0)


def _canvas_pad(blending: BlendingRule, pixel_scale: float) -> int:
    if blending.mode == "feather":
        return math.ceil(3.0 * blending.feather_sigma_px * pixel_scale)
    if blending.mode == "poisson":
        return 1
    return 0


def _null_source(scene, anchor, source_anchor, canvas, box) -> np.ndarray | None:
    """Original background under the canvas shifted to ``source_anchor``, if usable."""
    if source_anchor is None:
        return None
    shift = (source_anchor[0] - anchor[0], source_anchor[1] - anchor[1])
    top, left = canvas[0] + shift[0], canvas[1] + shift[1]
    bottom, right = canvas[2] + shift[0], canvas[3] + shift[1]
    if top < 0 or left < 0 or bottom > scene.height or right > scene.width:
        return None
    source_box = (
        box[0] + shift[0],
        box[1] + shift[1],
        box[2] + shift[0],
        box[3] + shift[1],
    )
    disjoint = (
        source_box[2] <= box[0]
        or box[2] <= source_box[0]
        or source_box[3] <= box[1]
        or box[3] <= source_box[1]
    )
    if not disjoint:
        return None
    return scene.original[top:bottom, left:right].copy()


def _blend(
    scene, proposal: _Placement, blending: BlendingRule
) -> tuple[tuple[str, Any], ...]:
    window = (
        slice(proposal.canvas[0], proposal.canvas[2]),
        slice(proposal.canvas[1], proposal.canvas[3]),
    )
    rgb = proposal.rgb
    ring_px = None
    harmonized = False
    if blending.harmonization_strength > 0:
        ring_px = max(1, math.ceil(blending.harmonization_ring_px * scene.pixel_scale))
        harmonized_rgb = _harmonize(
            scene, proposal, ring_px, blending.harmonization_strength
        )
        if harmonized_rgb is not None:
            rgb, harmonized = harmonized_rgb, True

    current = scene.canvas[window]
    if blending.mode == "poisson":
        blended = _poisson(current, rgb, proposal.footprint)
        blended = np.where(scene.valid[window][..., None], blended, current)
    else:
        alpha = proposal.alpha[..., None]
        blended = current * (1.0 - alpha) + rgb * alpha
    scene.canvas[window] = blended
    return (
        ("mode", blending.mode),
        (
            "feather_sigma_px",
            blending.feather_sigma_px * scene.pixel_scale
            if blending.mode == "feather"
            else None,
        ),
        ("harmonization_strength", blending.harmonization_strength),
        ("harmonization_ring_px", ring_px),
        ("harmonized", harmonized),
    )


def _poisson(current: np.ndarray, rgb: np.ndarray, footprint: np.ndarray) -> np.ndarray:
    bounds = clone_bounds(footprint)
    if bounds is None:
        return current
    top, left, bottom, right = bounds
    solution = clone_region(
        tf.constant(current[top:bottom, left:right]),
        tf.constant(rgb[top:bottom, left:right]),
        tf.constant(footprint[top:bottom, left:right]),
    ).numpy()
    blended = current.copy()
    blended[top:bottom, left:right] = np.floor(np.clip(solution, 0.0, 255.0))
    return blended


def _harmonize(scene, proposal: _Placement, ring_px: int, strength: float):
    """Instance RGB shifted in CIELAB toward the background ring, or None without ring."""
    top = max(0, proposal.canvas[0] - ring_px)
    left = max(0, proposal.canvas[1] - ring_px)
    bottom = min(scene.height, proposal.canvas[2] + ring_px)
    right = min(scene.width, proposal.canvas[3] + ring_px)
    footprint = np.zeros((bottom - top, right - left), np.float32)
    inner = (proposal.canvas[0] - top, proposal.canvas[1] - left)
    footprint[
        inner[0] : inner[0] + proposal.footprint.shape[0],
        inner[1] : inner[1] + proposal.footprint.shape[1],
    ] = proposal.footprint
    dilated = tf.nn.max_pool2d(
        footprint[None, :, :, None], ksize=2 * ring_px + 1, strides=1, padding="SAME"
    ).numpy()[0, :, :, 0]
    ring = (dilated > 0) & (footprint == 0) & scene.valid[top:bottom, left:right]
    if not ring.any():
        return None

    background = rgb_to_lab(scene.canvas[top:bottom, left:right][ring] / 255.0).numpy()
    lab = rgb_to_lab(proposal.rgb / 255.0).numpy()
    inside = lab[proposal.footprint]
    instance_mean, instance_std = inside.mean(axis=0), inside.std(axis=0)
    ring_mean, ring_std = background.mean(axis=0), background.std(axis=0)
    # Float32 Lab of a flat instance has a spread of ~1e-6, not 0; dividing by it
    # would amplify rounding noise instead of leaving a flat instance flat.
    flat = instance_std < _FLAT_LAB_STD
    ratio = np.divide(ring_std, instance_std, out=np.ones(3), where=~flat)
    gain = 1.0 + strength * (ratio - 1.0)
    shifted = (
        (lab - instance_mean) * gain
        + instance_mean
        + strength * (ring_mean - instance_mean)
    )
    return lab_to_rgb(shifted).numpy() * 255.0


def _write_labels(scene, index, instance: Instance, proposal: _Placement) -> int | None:
    window = (
        slice(proposal.canvas[0], proposal.canvas[2]),
        slice(proposal.canvas[1], proposal.canvas[3]),
    )
    footprint = proposal.footprint
    scene.occupied[window] |= footprint
    scene.owner[window][footprint] = index
    if instance.class_id is None:
        # Background now covers whatever an earlier paste labelled here.
        restore = footprint & scene.pasted[window]
        scene.mask[window][restore] = scene.original_mask[window][restore]
        if scene.instances is not None:
            scene.instances[window][restore] = scene.original_instances[window][restore]
        scene.pasted[window] &= ~footprint
        scene.null_pasted[window] |= footprint
        return None
    scene.mask[window][footprint] = instance.class_id
    scene.pasted[window] |= footprint
    scene.null_pasted[window] &= ~footprint
    if scene.instances is None:
        return None
    instance_id = scene.next_instance_id
    scene.instances[window][footprint] = instance_id
    scene.next_instance_id += 1
    return instance_id


__all__ = [
    "BLENDING_MODES",
    "PLACEMENT_SCALES",
    "BlendingRule",
    "CompositeDescriptor",
    "CompositeResult",
    "Instance",
    "PasteRecord",
    "PlacementRule",
    "composite_instances",
]

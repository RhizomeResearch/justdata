"""Paste annotated instances into a frame and update its dense labels.

``composite_instances`` places each requested instance by its anchor (the
bottom centre of its crop, at the waterline), checks the placement constraints,
blends it, and writes its class on the object's pixels whose rendered weight
(the final alpha) is at least 0.5. Every blending mode composites with an
explicit alpha, so labels describe the pixels that were rendered. Every instance is either
placed or rejected with counted reasons; nothing is substituted. An instance
without a class is a null paste: it takes background pixels from another allowed
location of the same frame, goes through the same placement and blending, and
writes no label, which measures how much a model reacts to compositing seams
alone (Dwibedi et al., ICCV 2017; Blum et al., Fishyscapes, IJCV 2021).

Each paste draws its blending mode from the rule's weighted mix, because mixing
modes keeps a model from learning any one mode's seams (Dwibedi et al.). Draws
use stateless Philox seeds folded from ``seed`` and the instance index, then
from the attempt index or a fixed salt for the mode. Compositing runs eagerly.
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

from justdata.vision.colorspace import (
    estimate_illuminant,
    linear_to_srgb,
    srgb_to_linear,
)
from justdata.vision.corruptions.registry import draw_seed
from justdata.vision.filters import gaussian_blur_hwc
from justdata.vision.horizon import Horizon, horizon_coordinates

COMPOSITE_VERSION = "2.0.0"
LABEL_ALPHA_THRESHOLD = 0.5
REFERENCE_WIDTH = 1024
PLACEMENT_SCALES = ("keep_row", "horizon")
BLENDING_MODES = ("hard", "soft_edge", "harmonized")
_BLENDING_DRAW = 0x424C4E44
_NO_ANCHOR = "no_allowed_anchor"
_SCALE_OUT_OF_RANGE = "scale_out_of_range"


@dataclass(frozen=True, eq=False)
class Instance:
    """An annotated object crop to paste.

    ``image`` is ``[h, w, 3]`` uint8 and ``alpha`` ``[h, w]`` bool or float in
    ``[0, 1]``. The crop's bottom row is the waterline, at
    ``anchor_row_fraction`` of the source image height. ``class_id=None`` makes
    a null paste. ``source_horizon_offset_px`` is the distance in source pixels
    from the source horizon down to the waterline, needed for horizon scaling.
    ``has_context`` states that the crop's pixels where alpha is below 0.5 are
    the real surroundings from the source frame, which soft edges may then
    show. Without context those pixels are replaced by the nearest object
    colours before any resampling or blending, so they never reach the output.
    ``source_illuminant`` is ``estimate_illuminant`` of the source frame;
    harmonized blending adapts the object from it to the destination frame's.
    """

    image: Any
    alpha: Any
    class_id: int | None
    source_id: str
    anchor_row_fraction: float
    source_horizon_offset_px: float | None = None
    has_context: bool = False
    source_illuminant: tuple[float, float, float] | None = None


@dataclass(frozen=True, slots=True)
class PlacementRule:
    """Where and at what scale instances may be placed.

    ``keep_row`` pastes at scale 1 with the anchor on the row at the instance's
    source row fraction. ``horizon`` scales by the ratio of the anchor's
    distance below the target horizon to ``source_horizon_offset_px`` and draws
    anchors only where that scale lies in ``[min_scale, max_scale]``, so a crop
    is placed where it fits instead of being stretched; upscaling past about 2
    visibly softens a crop. An instance is rejected at once with
    ``scale_out_of_range`` when no anchor fits (or, for ``keep_row``, when 1 is
    outside the range). A placement is rejected when more than
    ``max_outside_allowed_share`` of its
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
    min_scale: float = 0.0
    max_scale: float = 2.0

    def __post_init__(self) -> None:
        if self.scale not in PLACEMENT_SCALES:
            raise ValueError(f"PlacementRule scale must be one of {PLACEMENT_SCALES}.")
        if not (
            math.isfinite(self.min_scale)
            and math.isfinite(self.max_scale)
            and 0.0 <= self.min_scale <= self.max_scale
            and self.max_scale > 0.0
        ):
            raise ValueError(
                "PlacementRule needs finite 0 <= min_scale <= max_scale with "
                "max_scale > 0."
            )
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
            "min_scale": self.min_scale,
            "max_scale": self.max_scale,
        }


@dataclass(frozen=True, slots=True)
class BlendingRule:
    """How placed instances are blended.

    Each paste draws one mode from ``mix``, ``(mode, weight)`` pairs with
    positive weights. Every mode composites ``output = background * (1 - alpha)
    + object * alpha``:

    - ``hard``: alpha is the instance's alpha thresholded at 0.5.
    - ``soft_edge``: that binary alpha blurred by a Gaussian of
      ``edge_sigma_px``, centred on the edge; it hides jagged edges without
      fading the object or pulling in much of what surrounds it.
    - ``harmonized``: ``soft_edge`` plus a correction of the object for the
      change of illumination and exposure between frames: per-channel
      (von Kries) gains in linear light, the ratio of the destination's
      Grey-Edge illuminant to the instance's ``source_illuminant``, raised to
      ``harmonization_strength``. Without ``source_illuminant`` it is
      ``soft_edge``. The gains come from whole frames, not from the pixels
      around the object: water reflects its surroundings and has its own
      colour, so its local brightness and colour do not measure the light
      falling on the object, and matching them tinted objects and pushed
      white ones toward black.

    Pixel sizes refer to an image 1024 pixels wide and scale with the width.
    """

    mix: tuple[tuple[str, float], ...] = (("hard", 1.0),)
    edge_sigma_px: float = 1.0
    harmonization_strength: float = 1.0

    def __post_init__(self) -> None:
        mix = tuple((str(mode), float(weight)) for mode, weight in self.mix)
        object.__setattr__(self, "mix", mix)
        modes = [mode for mode, _weight in mix]
        if not mix or len(set(modes)) != len(modes):
            raise ValueError("BlendingRule mix needs at least one mode, each once.")
        for mode, weight in mix:
            if mode not in BLENDING_MODES:
                raise ValueError(f"BlendingRule modes must be in {BLENDING_MODES}.")
            if not (math.isfinite(weight) and weight > 0.0):
                raise ValueError(
                    "BlendingRule mix weights must be finite and positive."
                )
        if not self.edge_sigma_px > 0:
            raise ValueError("edge_sigma_px must be positive.")
        if not 0.0 <= self.harmonization_strength <= 1.0:
            raise ValueError("harmonization_strength must be in [0, 1].")

    def draw_mode(self, draw: float) -> str:
        """The mode whose share of the total weight contains ``draw`` in [0, 1)."""
        total = sum(weight for _mode, weight in self.mix)
        cumulative = 0.0
        for mode, weight in self.mix:
            cumulative += weight / total
            if draw < cumulative:
                return mode
        return self.mix[-1][0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mix": [[mode, weight] for mode, weight in self.mix],
            "edge_sigma_px": self.edge_sigma_px,
            "harmonization_strength": self.harmonization_strength,
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
            "mode_draw": f"stateless_fold_in_seed_instance_then_{_BLENDING_DRAW}",
            "scale_limits": "anchors_restricted_to_scale_range",
            "resize": "bilinear_antialias_round_size",
            "without_context": "outside_alpha_filled_from_nearest_object_pixels",
            "soft_edge": "gaussian_blur_reflect_101_of_binary_alpha",
            "harmonization": (
                "linear_light_von_kries_gains_of_frame_illuminant_ratio_"
                "strength_as_exponent"
            ),
            "illuminant_estimate": (
                "grey_edge_order_1_minkowski_6_sigma_width_over_1024_"
                "gradients_per_reference_pixel"
            ),
            "composite": "background_times_one_minus_alpha_plus_object_times_alpha",
            "label_write": "object_pixels_with_final_alpha_at_least_threshold_and_valid",
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
    source_illuminant: np.ndarray | None
    mode: str
    null_source_anchor: tuple[int, int] | None


class _Scene:
    """Mutable state of one compositing call."""

    @property
    def illuminant(self) -> np.ndarray:
        """Grey-Edge estimate of the original frame, computed once when needed."""
        if self._illuminant is None:
            self._illuminant = (
                estimate_illuminant(self.original / 255.0, self.valid)
                .numpy()
                .astype(np.float64)
            )
        return self._illuminant

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
        self._illuminant = None
        self.anchor_below = None
        if horizon is not None and self.anchors.size:
            below, _along = horizon_coordinates(
                tf.constant([horizon.left_row, horizon.right_row]),
                self.width,
                tf.constant(self.anchors[:, 1] + 0.5),
                tf.constant(self.anchors[:, 0] + 0.5),
            )
            self.anchor_below = below.numpy().astype(np.float64)
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
    if instance.source_illuminant is not None:
        illuminant = np.asarray(instance.source_illuminant, np.float64)
        if illuminant.shape != (3,) or not np.all(
            np.isfinite(illuminant) & (illuminant > 0)
        ):
            raise ValueError(
                f"Instance '{instance.source_id}' source_illuminant must be three "
                "positive finite numbers."
            )
    if instance.class_id is not None:
        limits = np.iinfo(mask_dtype)
        if not limits.min <= instance.class_id <= limits.max:
            raise ValueError(
                f"Instance '{instance.source_id}' class_id does not fit the mask."
            )
    rgb = rgb.astype(np.float32)
    if not instance.has_context:
        rgb = _extend_object_colours(rgb, alpha >= LABEL_ALPHA_THRESHOLD)
    return rgb, alpha


def _extend_object_colours(rgb: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """Replace pixels outside the object, ring by ring, with their object neighbours' mean."""
    rgb = rgb.copy()
    known = inside.copy()
    if not known.any():
        return rgb
    while not known.all():
        padded_known = np.pad(known, 1).astype(np.float32)
        padded_rgb = np.pad(rgb * known[..., None], [(1, 1), (1, 1), (0, 0)])
        count = np.zeros(known.shape, np.float32)
        total = np.zeros(rgb.shape, np.float32)
        for dy in range(3):
            for dx in range(3):
                window = (
                    slice(dy, dy + known.shape[0]),
                    slice(dx, dx + known.shape[1]),
                )
                count += padded_known[window]
                total += padded_rgb[window]
        frontier = ~known & (count > 0)
        rgb[frontier] = total[frontier] / count[frontier][:, None]
        known |= frontier
    return rgb


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
    mode_draw = tf.random.stateless_uniform(
        [], seed=draw_seed(instance_seed, _BLENDING_DRAW), alg="philox"
    )
    mode = blending.draw_mode(float(mode_draw))
    rejections: Counter[str] = Counter()
    attempts = 0
    for attempt in range(placement.max_attempts):
        attempts += 1
        draws = tf.random.stateless_uniform(
            [2], seed=draw_seed(instance_seed, attempt), alg="philox"
        ).numpy()
        proposal = _propose(
            scene, instance, rgb, alpha, placement, blending, mode, draws
        )
        if isinstance(proposal, str):
            rejections[proposal] += 1
            if proposal in (_NO_ANCHOR, _SCALE_OUT_OF_RANGE):
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


def _propose(scene, instance, rgb, alpha, placement, blending, mode, draws):
    """A checked placement, or the reason this attempt failed."""
    if placement.scale == "keep_row":
        if not placement.min_scale <= 1.0 <= placement.max_scale:
            return _SCALE_OUT_OF_RANGE
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
        assert scene.anchor_below is not None and instance.source_horizon_offset_px
        offset_px = instance.source_horizon_offset_px
        below = scene.anchor_below
        fits = np.flatnonzero(
            (below > 0.0)
            & (below >= placement.min_scale * offset_px)
            & (below <= placement.max_scale * offset_px)
        )
        if fits.size == 0:
            return _SCALE_OUT_OF_RANGE
        chosen = fits[min(int(draws[0] * fits.size), fits.size - 1)]
        anchor = (int(scene.anchors[chosen][0]), int(scene.anchors[chosen][1]))
        scale = float(below[chosen]) / offset_px

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
    pad = _canvas_pad(blending, mode, scene.pixel_scale)
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
    if mode == "hard":
        final_alpha = binary
    else:
        sigma = blending.edge_sigma_px * scene.pixel_scale
        final_alpha = gaussian_blur_hwc(binary[..., None], sigma, sigma).numpy()[..., 0]
    window = (slice(canvas[0], canvas[2]), slice(canvas[1], canvas[3]))
    final_alpha = final_alpha * scene.valid[window]
    # A soft edge can reach half weight just outside the object, at concave
    # corners; those pixels show the crop's surroundings, not the object.
    footprint = (final_alpha >= LABEL_ALPHA_THRESHOLD) & (binary > 0)

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
        # A null paste comes from this frame, under the same illuminant.
        source_illuminant=(
            None
            if instance.class_id is None or instance.source_illuminant is None
            else np.asarray(instance.source_illuminant, np.float64)
        ),
        mode=mode,
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


def _canvas_pad(blending: BlendingRule, mode: str, pixel_scale: float) -> int:
    if mode == "hard":
        return 0
    return max(1, math.ceil(3.0 * blending.edge_sigma_px * pixel_scale))


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
    current = scene.canvas[window]
    rgb = proposal.rgb
    gains = None
    strength = blending.harmonization_strength
    if proposal.mode == "harmonized" and proposal.source_illuminant is not None:
        gains = np.power(scene.illuminant / proposal.source_illuminant, strength)
        linear = srgb_to_linear(rgb / 255.0).numpy().astype(np.float64) * gains
        rgb = linear_to_srgb(np.clip(linear, 0.0, 1.0)).numpy() * 255.0

    alpha = proposal.alpha[..., None]
    scene.canvas[window] = current * (1.0 - alpha) + np.clip(rgb, 0.0, 255.0) * alpha
    return (
        ("mode", proposal.mode),
        (
            "edge_sigma_px",
            None
            if proposal.mode == "hard"
            else blending.edge_sigma_px * scene.pixel_scale,
        ),
        ("harmonization_strength", strength if proposal.mode == "harmonized" else None),
        ("gains", None if gains is None else [float(g) for g in gains]),
    )


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

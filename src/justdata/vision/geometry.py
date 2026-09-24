"""Versioned, paired dense geometry in HWC coordinates.

Records are dictionaries of numeric tensors, suitable for tf.data, NumPy storage,
and replay without a random generator. Version 1 uses half-pixel resize centers,
bilinear RGB sampling, nearest categorical sampling, and bottom/right padding.
"""

import math
from dataclasses import asdict, dataclass

import tensorflow as tf

from justdata.vision.encodings.panoptic_targets import (
    _lookup,
    _pad_segment_table,
    validate_panoptic_sample,
)
from justdata.vision.encodings.semantic_targets import (
    _assert_declared_labels,
    _categorical_mask,
    _class_contract,
)


@dataclass(frozen=True)
class DenseGeometryConfig:
    """Geometry and categorical-label contract for one dense view.

    Training samples either an integer shorter-side size from the inclusive
    resize range or a scale relative to fitting the image inside the crop.
    It pads to fit a square crop, samples its origin uniformly, optionally
    flips horizontally, then pads to the patch multiple. Evaluation caps the
    longer side, without upscaling by default, and only pads to the multiple.
    Dimensions round half up and are clamped to at least one pixel.
    """

    class_values: tuple[int, ...]
    ignore_value: int = 255
    train_crop_size: int = 512
    train_resize_range: tuple[int, int] | None = (512, 1024)
    train_scale_range: tuple[float, float] | None = None
    horizontal_flip_probability: float = 0.5
    eval_long_side: int = 1024
    eval_upscale: bool = False
    patch_size: int = 16
    image_pad_mode: str = "CONSTANT"
    image_pad_value: tuple[float, float, float] = (0.0, 0.0, 0.0)
    image_antialias: bool = True

    def __post_init__(self):
        values = _class_contract(self.class_values, self.ignore_value)
        for name in ("train_crop_size", "eval_long_side", "patch_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (self.train_resize_range is None) == (self.train_scale_range is None):
            raise ValueError("exactly one training resize or scale range is required")
        if self.train_resize_range is not None:
            resize_range = tuple(self.train_resize_range)
            if (
                len(resize_range) != 2
                or any(type(v) is not int or v <= 0 for v in resize_range)
                or resize_range[0] > resize_range[1]
            ):
                raise ValueError(
                    "train_resize_range must be two ordered positive integers"
                )
            object.__setattr__(self, "train_resize_range", resize_range)
        if self.train_scale_range is not None:
            scale_range = tuple(self.train_scale_range)
            if (
                len(scale_range) != 2
                or any(
                    isinstance(v, bool)
                    or not isinstance(v, (int, float))
                    or not math.isfinite(v)
                    or v <= 0
                    for v in scale_range
                )
                or scale_range[0] > scale_range[1]
            ):
                raise ValueError(
                    "train_scale_range must be two ordered positive numbers"
                )
            object.__setattr__(self, "train_scale_range", scale_range)
        if not 0 <= self.horizontal_flip_probability <= 1:
            raise ValueError("horizontal_flip_probability must be in [0, 1]")
        if (
            type(self.eval_upscale) is not bool
            or type(self.image_antialias) is not bool
        ):
            raise ValueError("eval_upscale and image_antialias must be booleans")
        if self.image_pad_mode not in {"CONSTANT", "REFLECT", "SYMMETRIC"}:
            raise ValueError("image_pad_mode must be CONSTANT, REFLECT, or SYMMETRIC")
        fill = tuple(self.image_pad_value)
        if len(fill) != 3 or any(
            not math.isfinite(v) or not 0 <= v <= 255 for v in fill
        ):
            raise ValueError(
                "image_pad_value must contain three finite RGB values in [0, 255]"
            )
        object.__setattr__(self, "class_values", values)
        object.__setattr__(self, "image_pad_value", fill)


@dataclass(frozen=True)
class PanopticGeometryConfig:
    """Spatial settings shared with dense semantic geometry, without label IDs."""

    train_crop_size: int = 512
    train_resize_range: tuple[int, int] | None = (512, 1024)
    train_scale_range: tuple[float, float] | None = None
    horizontal_flip_probability: float = 0.5
    eval_long_side: int = 1024
    eval_upscale: bool = False
    patch_size: int = 16
    image_pad_mode: str = "CONSTANT"
    image_pad_value: tuple[float, float, float] = (0.0, 0.0, 0.0)
    image_antialias: bool = True

    def __post_init__(self):
        validated = _spatial_config(self)
        object.__setattr__(self, "train_resize_range", validated.train_resize_range)
        object.__setattr__(self, "train_scale_range", validated.train_scale_range)
        object.__setattr__(self, "image_pad_value", validated.image_pad_value)


def _spatial_config(config: PanopticGeometryConfig) -> DenseGeometryConfig:
    """Express panoptic spatial settings through the semantic implementation."""
    return DenseGeometryConfig(class_values=(0,), ignore_value=-1, **asdict(config))


def _train_input_size(train_crop_size: int, patch_size: int) -> int:
    """Side of a square training view: the crop padded to the patch multiple."""
    return (train_crop_size + patch_size - 1) // patch_size * patch_size


def _bottom_right(amount):
    return tf.stack([0, amount[0], 0, amount[1]])


def sample_dense_geometry(
    original_size, config: DenseGeometryConfig, *, is_training, seed=None
):
    """Create a version-1 record; training requires an int32/int64 seed of shape [2]."""
    original = tf.ensure_shape(tf.convert_to_tensor(original_size, tf.int32), [2])
    tf.debugging.assert_positive(original, message="original_size must be positive")
    if is_training:
        if seed is None:
            raise ValueError("training geometry requires a stateless seed")
        seed = tf.ensure_shape(tf.convert_to_tensor(seed), [2])
        if seed.dtype not in (tf.int32, tf.int64):
            raise TypeError("seed must have int32 or int64 dtype")
        seeds = tf.random.experimental.stateless_split(seed, 4)
        if config.train_resize_range is not None:
            low, high = config.train_resize_range
            target = tf.random.stateless_uniform(
                [], seeds[0], low, high + 1, dtype=tf.int32
            )
            scale = tf.cast(target, tf.float64) / tf.cast(
                tf.reduce_min(original), tf.float64
            )
        elif config.train_scale_range is not None:
            low, high = config.train_scale_range
            fraction = tf.random.stateless_uniform([], seeds[0], dtype=tf.float64)
            factor = low + (high - low) * fraction
            crop = tf.cast(config.train_crop_size, tf.float64)
            scale = factor * tf.reduce_min(crop / tf.cast(original, tf.float64))
    else:
        scale = config.eval_long_side / tf.cast(tf.reduce_max(original), tf.float64)
        if not config.eval_upscale:
            scale = tf.minimum(scale, 1.0)
    resized = tf.maximum(
        1, tf.cast(tf.floor(tf.cast(original, tf.float64) * scale + 0.5), tf.int32)
    )
    if is_training:
        before = tf.maximum(config.train_crop_size - resized, 0)
        available = resized + before - config.train_crop_size
        top = tf.random.stateless_uniform(
            [], seeds[1], 0, available[0] + 1, dtype=tf.int32
        )
        left = tf.random.stateless_uniform(
            [], seeds[2], 0, available[1] + 1, dtype=tf.int32
        )
        cropped = tf.constant(
            [config.train_crop_size, config.train_crop_size], tf.int32
        )
        flip = (
            tf.random.stateless_uniform([], seeds[3])
            < config.horizontal_flip_probability
        )
    else:
        before = tf.zeros([2], tf.int32)
        top, left = 0, 0
        cropped = resized
        flip = tf.constant(False)
    after = (config.patch_size - cropped % config.patch_size) % config.patch_size
    input_size = cropped + after
    return {
        "version": tf.constant(1, tf.int32),
        "original_size": original,
        "resized_size": resized,
        "crop_box": tf.stack([top, left, cropped[0], cropped[1]]),
        "pre_padding": _bottom_right(before),
        "post_padding": _bottom_right(after),
        "horizontal_flip": flip,
        "is_training": tf.constant(is_training),
        "model_input_size": input_size,
        "patch_size": tf.constant(config.patch_size, tf.int32),
        "grid_size": input_size // config.patch_size,
        # Numeric enum values are part of version 1, documented in docs/vision.md.
        "image_interpolation": tf.constant(1, tf.int32),
        "mask_interpolation": tf.constant(0, tf.int32),
        "alignment": tf.constant(1, tf.int32),
        "image_antialias": tf.constant(config.image_antialias),
        "image_pad_mode": tf.constant(
            ("CONSTANT", "REFLECT", "SYMMETRIC").index(config.image_pad_mode), tf.int32
        ),
        "image_pad_value": tf.constant(config.image_pad_value, tf.float32),
        "mask_fill_value": tf.constant(config.ignore_value, tf.int32),
        "class_values": tf.constant(config.class_values, tf.int32),
    }


def sample_panoptic_geometry(
    original_size, config: PanopticGeometryConfig, *, void_value, is_training, seed=None
):
    """Sample version-1 geometry with a panoptic label contract."""
    record = sample_dense_geometry(
        original_size, _spatial_config(config), is_training=is_training, seed=seed
    )
    return record | {
        "mask_fill_value": tf.constant(void_value, tf.int64),
        "class_values": tf.constant([], tf.int32),
    }


def _check_record(record):
    record = tf.nest.map_structure(tf.convert_to_tensor, record)
    record["version"] = tf.ensure_shape(record["version"], [])
    tf.debugging.assert_equal(
        record["version"],
        1,
        message="unsupported geometry version",
    )
    for key, value in (
        ("alignment", 1),
        ("image_interpolation", 1),
        ("mask_interpolation", 0),
    ):
        tf.debugging.assert_equal(
            record[key], value, message=f"unsupported geometry {key}"
        )
    for key in ("original_size", "resized_size", "model_input_size", "grid_size"):
        record[key] = tf.ensure_shape(record[key], [2])
        tf.debugging.assert_positive(record[key], message=key)
    for key in ("pre_padding", "post_padding", "crop_box"):
        record[key] = tf.ensure_shape(record[key], [4])
        tf.debugging.assert_non_negative(record[key], message=key)
    tf.debugging.assert_positive(record["crop_box"][2:])
    tf.debugging.assert_positive(record["patch_size"])
    tf.debugging.assert_greater_equal(record["image_pad_mode"], 0)
    tf.debugging.assert_less_equal(record["image_pad_mode"], 2)
    pre_size = record["resized_size"] + tf.reduce_sum(
        tf.reshape(record["pre_padding"], [2, 2]), axis=1
    )
    tf.debugging.assert_less_equal(
        record["crop_box"][:2] + record["crop_box"][2:],
        pre_size,
        message="crop outside resized frame",
    )
    output_size = record["crop_box"][2:] + tf.reduce_sum(
        tf.reshape(record["post_padding"], [2, 2]), axis=1
    )
    tf.debugging.assert_equal(
        record["model_input_size"], output_size, message="model input geometry mismatch"
    )
    tf.debugging.assert_equal(
        record["grid_size"] * record["patch_size"],
        output_size,
        message="patch grid mismatch",
    )
    return record


def validate_dense_sample(sample, class_values, ignore_value):
    """Check HWC RGB and integer HW/HW1 masks before any resampling or crop."""
    image = tf.ensure_shape(sample["image"], [None, None, 3])
    tf.debugging.assert_positive(tf.shape(image)[:2])
    result = sample | {"image": image}
    if "mask" in sample:
        mask = _categorical_mask(sample["mask"])
        tf.debugging.assert_equal(
            tf.shape(mask), tf.shape(image)[:2], message="image/mask size mismatch"
        )
        _assert_declared_labels(mask, class_values, ignore_value)
        result["mask"] = mask
    if "annotation_valid_mask" in sample:
        valid = tf.convert_to_tensor(sample["annotation_valid_mask"])
        if valid.dtype != tf.bool:
            raise TypeError("annotation_valid_mask must be boolean")
        tf.debugging.assert_equal(
            tf.shape(valid),
            tf.shape(image)[:2],
            message="annotation validity size mismatch",
        )
        result["annotation_valid_mask"] = tf.ensure_shape(valid, [None, None])
    return result


def _nearest(tensor, size):
    # Gather directly so categorical int64 IDs never pass through float32.
    indices = []
    for axis in (0, 1):
        length = tf.shape(tensor)[axis]
        centers = tf.cast(tf.range(size[axis]), tf.float64) + 0.5
        scale = tf.cast(length, tf.float64) / tf.cast(size[axis], tf.float64)
        nearest = tf.cast(tf.floor(centers * scale), tf.int32)
        indices.append(tf.minimum(nearest, length - 1))
    return tf.gather(tf.gather(tensor, indices[0], axis=0), indices[1], axis=1)


def _constant_pad(tensor, widths, value):
    paddings = tf.reshape(widths, [2, 2])
    if tensor.shape.rank == 3:
        paddings = tf.concat([paddings, [[0, 0]]], 0)
    return tf.pad(tensor, paddings, constant_values=tf.cast(value, tensor.dtype))


def _pad_rgb(image, widths, record):
    def constant():
        padded = _constant_pad(image, widths, 0)
        valid = _constant_pad(tf.ones(tf.shape(image)[:2], tf.bool), widths, False)
        return tf.where(valid[..., None], padded, record["image_pad_value"])

    def mirrored():
        result = image
        for axis in (0, 1):
            length = tf.shape(image)[axis]
            start, end = widths[2 * axis], widths[2 * axis + 1]
            symmetric = tf.cast(record["image_pad_mode"] == 2, tf.int32)
            period = tf.maximum(2 * (length - 1 + symmetric), 1)
            indices = tf.math.floormod(tf.range(-start, length + end), period)
            indices = tf.minimum(indices, period - indices - symmetric)
            result = tf.gather(result, indices, axis=axis)
        return result

    return tf.cond(record["image_pad_mode"] == 0, constant, mirrored)


def replay_dense_geometry(sample, record):
    """Apply a stored record to the original HWC sample, without sampling.

    Unknown sample fields pass through. ``source_valid_mask`` marks image
    support including ignored annotations; ``pixel_valid_mask`` additionally
    excludes ignore IDs and unavailable annotations (all false without a mask).
    Replay covers geometry; callers may apply photometric transforms before or
    after replay, depending on the pipeline.
    """
    record = _check_record(record)
    tf.debugging.assert_positive(
        tf.size(record["class_values"]),
        message="semantic geometry requires class values",
    )
    sample = validate_dense_sample(
        sample, record["class_values"], record["mask_fill_value"]
    )
    result, categorical = _replay_spatial(sample, record)
    support = result["source_valid_mask"]
    if "mask" in sample:
        mask = categorical(sample["mask"], record["mask_fill_value"])
        result["mask"] = mask
        valid = support & (mask != tf.cast(record["mask_fill_value"], mask.dtype))
    else:
        valid = tf.zeros_like(support)
    if "annotation_valid_mask" in sample:
        annotation = categorical(sample["annotation_valid_mask"], False)
        result["annotation_valid_mask"] = annotation
        valid &= annotation
    return result | {"pixel_valid_mask": valid}


def _replay_spatial(sample, record):
    tf.debugging.assert_equal(
        tf.shape(sample["image"])[:2],
        record["original_size"],
        message="replay requires original sample dimensions",
    )
    image = tf.cast(sample["image"], tf.float32)
    image = tf.cond(
        record["image_antialias"],
        lambda: tf.image.resize(
            image, record["resized_size"], method="bilinear", antialias=True
        ),
        lambda: tf.image.resize(
            image, record["resized_size"], method="bilinear", antialias=False
        ),
    )

    def crop_flip(tensor):
        top, left, height, width = tf.unstack(record["crop_box"])
        tensor = tensor[top : top + height, left : left + width]
        return tf.cond(
            record["horizontal_flip"], lambda: tf.reverse(tensor, [1]), lambda: tensor
        )

    image = _pad_rgb(image, record["pre_padding"], record)
    image = _pad_rgb(crop_flip(image), record["post_padding"], record)

    def categorical(tensor, fill):
        tensor = _nearest(tensor, record["resized_size"])
        tensor = _constant_pad(tensor, record["pre_padding"], fill)
        return _constant_pad(crop_flip(tensor), record["post_padding"], fill)

    support = categorical(tf.ones(record["original_size"], tf.bool), False)
    result = sample | {"image": image, "geometry": record, "source_valid_mask": support}
    return result, categorical


def _visible_segments(mask, segments, max_segments):
    """Keep segments visible in a replayed view and recompute area and bbox."""
    flat = tf.reshape(mask, [-1])
    width = tf.shape(mask)[1]
    unique, index = tf.unique(flat)
    size = tf.size(unique)
    counts = tf.math.unsorted_segment_sum(tf.ones_like(index), index, size)
    rows = tf.range(tf.size(flat)) // width
    cols = tf.range(tf.size(flat)) % width
    top = tf.math.unsorted_segment_min(rows, index, size)
    left = tf.math.unsorted_segment_min(cols, index, size)
    bottom = tf.math.unsorted_segment_max(rows, index, size)
    right = tf.math.unsorted_segment_max(cols, index, size)
    order = tf.argsort(unique, stable=True)
    sorted_ids = tf.gather(unique, order)
    source_ids = segments["segment_ids"]
    pos = tf.minimum(
        tf.searchsorted(sorted_ids, source_ids, out_type=tf.int32), size - 1
    )
    # Row of each source segment in the per-ID statistics above.
    slot = tf.gather(order, pos)
    found = (tf.gather(sorted_ids, pos) == source_ids) & segments["valid_mask"]
    areas = tf.where(found, tf.gather(counts, slot), 0)
    box_left, box_top = tf.gather(left, slot), tf.gather(top, slot)
    boxes = tf.stack(
        [
            box_left,
            box_top,
            tf.gather(right, slot) - box_left + 1,
            tf.gather(bottom, slot) - box_top + 1,
        ],
        axis=-1,
    )
    boxes = tf.where(found[:, None], boxes, 0)
    kept = found & (areas > 0)
    return _pad_segment_table(
        tf.boolean_mask(source_ids, kept),
        tf.boolean_mask(segments["category_ids"], kept),
        tf.boolean_mask(segments["is_crowd"], kept),
        max_segments,
        area=tf.boolean_mask(areas, kept),
        bbox=tf.boolean_mask(boxes, kept),
    )


def replay_panoptic_geometry(
    sample, record, *, class_values, thing_class_values, void_value, max_segments
):
    """Replay a version-1 record, including integer map and crowd validity."""
    record = _check_record(record)
    tf.debugging.assert_equal(
        tf.size(record["class_values"]),
        0,
        message="panoptic geometry must not contain semantic class values",
    )
    if record["mask_fill_value"].dtype != tf.int64:
        raise TypeError("panoptic geometry mask fill must be int64")
    tf.debugging.assert_equal(
        record["mask_fill_value"],
        tf.cast(void_value, tf.int64),
        message="panoptic geometry void value mismatch",
    )
    sample = validate_panoptic_sample(
        sample,
        class_values=class_values,
        thing_class_values=thing_class_values,
        void_value=void_value,
        max_segments=max_segments,
    )
    result, categorical = _replay_spatial(sample, record)
    mask = categorical(sample["panoptic_mask"], void_value)
    segments = _visible_segments(mask, sample["segments"], max_segments)
    support = result["source_valid_mask"]
    valid = support & (mask != void_value)
    if "annotation_valid_mask" in sample:
        annotation = categorical(sample["annotation_valid_mask"], False)
        result["annotation_valid_mask"] = annotation
        valid &= annotation
    _, crowd = _lookup(mask, segments)
    valid &= ~crowd
    return result | {
        "panoptic_mask": mask,
        "segments": segments,
        "pixel_valid_mask": valid,
    }


def _validate_scores(scores):
    scores = tf.convert_to_tensor(scores)
    if not scores.dtype.is_floating:
        raise TypeError("scores must be floating point; restore before argmax")
    scores = tf.ensure_shape(scores, [None, None, None])
    tf.debugging.assert_positive(tf.shape(scores))
    tf.debugging.assert_all_finite(scores, "scores must be finite")
    return tf.cast(scores, tf.float32)


def _bilinear_scores(scores, size):
    return tf.raw_ops.ResizeBilinear(
        images=scores[None],
        size=size,
        align_corners=False,
        half_pixel_centers=True,
    )[0]


def restore_dense_scores(scores, record, *, from_model_input=False):
    """Restore one HWC score/logit field to its original evaluation frame.

    First bilinearly resize to the padded input, remove padding, then resize to
    the original frame. Both resizes use half-pixel centers, clamped borders,
    no antialiasing, and float32. Training crops have no scoring inverse here.
    Set ``from_model_input=True`` for scores already reduced at the padded input
    resolution; their size is checked and the first resize is skipped.
    """
    record = _check_record(record)
    scores = _validate_scores(scores)
    tf.debugging.assert_equal(
        record["is_training"],
        False,
        message="training crops have no defined scoring inverse",
    )
    tf.debugging.assert_equal(record["horizontal_flip"], False)
    tf.debugging.assert_equal(record["pre_padding"], tf.zeros([4], tf.int32))
    tf.debugging.assert_equal(
        record["crop_box"],
        tf.concat([[0, 0], record["resized_size"]], 0),
        message="evaluation crop has no defined scoring inverse",
    )
    if from_model_input:
        tf.debugging.assert_equal(
            tf.shape(scores)[:2],
            record["model_input_size"],
            message="scores must match padded model input",
        )
    else:
        scores = _bilinear_scores(scores, record["model_input_size"])
    top, _, left, _ = tf.unstack(record["post_padding"])
    height, width = tf.unstack(record["resized_size"])
    scores = scores[top : top + height, left : left + width]
    return _bilinear_scores(scores, record["original_size"])


def restore_dense_predictions(logits, record):
    """Restore pointwise HWC logits, then return argmax channel indices as int32."""
    return tf.argmax(
        restore_dense_scores(logits, record), axis=-1, output_type=tf.int32
    )


__all__ = [
    "DenseGeometryConfig",
    "PanopticGeometryConfig",
    "sample_dense_geometry",
    "sample_panoptic_geometry",
    "validate_dense_sample",
    "replay_dense_geometry",
    "replay_panoptic_geometry",
    "restore_dense_scores",
    "restore_dense_predictions",
]

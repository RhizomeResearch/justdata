"""Panoptic segment validation, projection, and fixed-capacity targets."""

import tensorflow as tf

from justdata.vision.encodings.semantic_targets import _CATEGORICAL_DTYPES

_OVERFLOW_POLICIES = ("error", "exclude_supervision", "merge_by_category")


def decode_panoptic_rgb(rgb):
    """Decode COCO/LaRS RGB segment IDs without floating-point conversion."""
    rgb = tf.convert_to_tensor(rgb)
    if rgb.dtype != tf.uint8:
        raise TypeError("panoptic RGB mask must be uint8")
    rgb = tf.ensure_shape(rgb, [None, None, 3])
    return tf.reduce_sum(
        tf.cast(rgb, tf.int64) * tf.constant([1, 256, 65536], tf.int64), axis=-1
    )


def _values(class_values, thing_class_values, void_value, max_segments):
    classes = tuple(class_values)
    things = tuple(thing_class_values)
    if (
        not classes
        or any(type(value) is not int or not 0 <= value < 2**31 for value in classes)
        or len(set(classes)) != len(classes)
    ):
        raise ValueError("class_values must be unique nonnegative int32 IDs")
    if (
        any(type(value) is not int for value in things)
        or len(set(things)) != len(things)
        or not set(things) <= set(classes)
    ):
        raise ValueError("thing_class_values must be a unique subset of class_values")
    if type(void_value) is not int or not 0 <= void_value < 2**63:
        raise ValueError("void_value must be a nonnegative int64 ID")
    if type(max_segments) is not int or max_segments <= 0:
        raise ValueError("max_segments must be a positive integer")
    return classes, things


def validate_panoptic_sample(
    sample, *, class_values, thing_class_values, void_value=0, max_segments
):
    """Return a sorted, padded segment table and verify exact map references."""
    classes, _ = _values(class_values, thing_class_values, void_value, max_segments)
    image = tf.ensure_shape(tf.convert_to_tensor(sample["image"]), [None, None, 3])
    tf.debugging.assert_positive(tf.shape(image)[:2])
    mask, ids, categories, crowd = _validate_map_segments(
        sample["panoptic_mask"], sample["segments"], classes, void_value, max_segments
    )
    tf.debugging.assert_equal(tf.shape(mask), tf.shape(image)[:2])
    padded = _pad_segment_table(ids, categories, crowd, max_segments)
    result = sample | {"image": image, "panoptic_mask": mask, "segments": padded}
    if "annotation_valid_mask" in sample:
        validity = tf.convert_to_tensor(sample["annotation_valid_mask"])
        if validity.dtype != tf.bool:
            raise TypeError("annotation_valid_mask must be boolean")
        validity = tf.ensure_shape(validity, [None, None])
        tf.debugging.assert_equal(tf.shape(validity), tf.shape(mask))
        result["annotation_valid_mask"] = validity
    return result


def _valid_sorted_segments(segments):
    """Return valid segment IDs, categories, and crowd flags ordered by ID."""
    ids, categories, crowd = (
        tf.boolean_mask(segments[key], segments["valid_mask"])
        for key in ("segment_ids", "category_ids", "is_crowd")
    )
    order = tf.argsort(ids, stable=True)
    return tuple(tf.gather(value, order) for value in (ids, categories, crowd))


def _pad_segment_table(ids, categories, crowd, max_segments, **columns):
    """Pad a segment table to its static capacity, with valid rows first."""
    pad = max_segments - tf.size(ids)

    def padded(value):
        paddings = [[0, pad]] + [[0, 0]] * (value.shape.rank - 1)
        return tf.ensure_shape(
            tf.pad(value, paddings), [max_segments, *value.shape[1:]]
        )

    table = {
        "segment_ids": padded(ids),
        "category_ids": padded(categories),
        "is_crowd": padded(crowd),
        "valid_mask": tf.ensure_shape(
            tf.concat([tf.ones_like(ids, tf.bool), tf.zeros([pad], tf.bool)], 0),
            [max_segments],
        ),
    }
    return table | {key: padded(value) for key, value in columns.items()}


def _validate_map_segments(mask, source, classes, void_value, max_segments=None):
    """Check exact map references without allocating RGB.

    Returns the map and the valid table rows ordered by segment ID.
    ``max_segments`` bounds those rows when the table has a fixed capacity.
    """
    mask = tf.convert_to_tensor(mask)
    if mask.dtype not in _CATEGORICAL_DTYPES:
        raise TypeError("panoptic_mask must have an integer categorical dtype")
    mask = tf.ensure_shape(mask, [None, None])
    mask = tf.cast(mask, tf.int64)
    tf.debugging.assert_non_negative(mask)
    required = {"segment_ids", "category_ids", "is_crowd", "valid_mask"}
    if set(source) not in (required, required | {"area", "bbox"}):
        raise ValueError(
            "segments requires segment_ids, category_ids, is_crowd, valid_mask"
        )
    segments = {}
    for key, dtype in (
        ("segment_ids", tf.int64),
        ("category_ids", tf.int32),
        ("is_crowd", tf.bool),
        ("valid_mask", tf.bool),
    ):
        value = tf.convert_to_tensor(source[key])
        if value.dtype != dtype:
            raise TypeError(f"segments.{key} must be {dtype.name}")
        segments[key] = tf.ensure_shape(value, [None])
        tf.debugging.assert_equal(
            tf.shape(value)[0], tf.shape(segments["segment_ids"])[0]
        )
    ids, categories, crowd = _valid_sorted_segments(segments)
    if max_segments is not None:
        tf.debugging.assert_less_equal(
            tf.size(ids), max_segments, message="segment capacity exceeded"
        )
    tf.debugging.assert_non_negative(ids)
    tf.debugging.assert_none_equal(ids, tf.cast(void_value, tf.int64))
    tf.debugging.assert_equal(
        tf.size(tf.unique(ids).y), tf.size(ids), message="duplicate segment ID"
    )
    allowed = tf.constant(classes, tf.int32)
    tf.debugging.assert_equal(
        tf.reduce_all(tf.reduce_any(categories[:, None] == allowed, axis=1)),
        True,
        message="undeclared panoptic category",
    )
    unique_ids = tf.unique(tf.reshape(mask, [-1])).y
    present = tf.sort(tf.boolean_mask(unique_ids, unique_ids != void_value))
    tf.debugging.assert_equal(
        present, ids, message="panoptic map and segment table disagree"
    )
    return mask, ids, categories, crowd


def _lookup(mask, ids, categories, crowd):
    """Map each pixel's segment ID to its category and crowd flag.

    The table rows are the valid rows ordered by segment ID.
    """
    flat = tf.reshape(mask, [-1])

    def nonempty():
        indices = tf.minimum(
            tf.searchsorted(ids, flat, out_type=tf.int32), tf.size(ids) - 1
        )
        return tf.gather(categories, indices), tf.gather(crowd, indices)

    cat, is_crowd = tf.cond(
        tf.size(ids) > 0,
        nonempty,
        lambda: (tf.zeros_like(flat, tf.int32), tf.zeros_like(flat, tf.bool)),
    )
    return tf.reshape(cat, tf.shape(mask)), tf.reshape(is_crowd, tf.shape(mask))


def panoptic_map_to_semantic(
    mask,
    segments,
    *,
    class_values,
    thing_class_values,
    semantic_values,
    ignore_value=255,
    void_value=0,
    max_segments,
    annotation_valid_mask=None,
    include_crowd=True,
):
    """Project source categories to semantic labels, retaining crowds by default."""
    classes, _ = _values(class_values, thing_class_values, void_value, max_segments)
    if len(semantic_values) != len(classes):
        raise ValueError("semantic_values must match class_values")
    mask, ids, categories, crowd = _validate_map_segments(
        mask, segments, classes, void_value, max_segments
    )
    category, crowd_pixels = _lookup(mask, ids, categories, crowd)
    matches = category[..., None] == tf.constant(classes, tf.int32)
    index = tf.argmax(tf.cast(matches, tf.int32), axis=-1, output_type=tf.int32)
    mapped = tf.gather(tf.constant(tuple(semantic_values), tf.int32), index)
    valid = mask != void_value
    if annotation_valid_mask is not None:
        annotation_valid_mask = tf.convert_to_tensor(annotation_valid_mask)
        if annotation_valid_mask.dtype != tf.bool:
            raise TypeError("annotation_valid_mask must be boolean")
        tf.debugging.assert_equal(tf.shape(annotation_valid_mask), tf.shape(mask))
        valid &= annotation_valid_mask
    if not include_crowd:
        valid &= ~crowd_pixels
    return tf.where(valid, mapped, tf.cast(ignore_value, tf.int32))


def _things_with_own_target(thing_masks, class_index, num_classes, room, policy):
    """Select the things that keep their own target when ``room`` targets remain.

    Things rank by valid area, ties in target order, and the largest that fit
    are kept. Merging also needs one target for each category with things left
    over; a category left with a single thing keeps it as an instance, since it
    takes that target either way.
    """
    count = tf.shape(class_index)[0]
    if policy == "error":
        return tf.ones([count], tf.bool)
    area = tf.reduce_sum(tf.cast(thing_masks, tf.int32), axis=[1, 2])
    rank = tf.math.invert_permutation(
        tf.argsort(area, direction="DESCENDING", stable=True)
    )
    kept = tf.range(count + 1)
    needed = kept
    if policy == "merge_by_category":
        last = tf.math.unsorted_segment_max(rank, class_index, num_classes)
        needed += tf.reduce_sum(tf.cast(last >= kept[:, None], tf.int32), axis=1)
    # Keeping one more thing takes one target and frees at most one merged
    # target, so ``needed`` never decreases and the counts that fit are a prefix.
    fit = tf.reduce_sum(tf.cast(needed <= room, tf.int32)) - 1
    tf.debugging.assert_non_negative(fit, message="panoptic target capacity exceeded")
    own = rank < fit
    if policy == "merge_by_category":
        left = tf.math.unsorted_segment_sum(
            tf.cast(~own, tf.int32), class_index, num_classes
        )
        own |= tf.gather(left, class_index) == 1
    return own


def panoptic_map_to_targets(
    mask,
    segments,
    *,
    class_values,
    thing_class_values,
    void_value=0,
    max_segments,
    pixel_valid_mask,
    example_valid=True,
    overflow_policy="error",
):
    """Make one target per thing instance and one per present stuff category.

    ``max_segments`` is the target capacity; the segment table may hold more
    rows. When the visible things do not fit beside the stuff targets,
    ``"error"`` fails, and the other policies keep the things with the most
    valid pixels that fit: ``"exclude_supervision"`` removes the rest from
    pixel validity, and ``"merge_by_category"`` gives each category's rest one
    shared non-instance target.
    """
    classes, things = _values(
        class_values, thing_class_values, void_value, max_segments
    )
    if overflow_policy not in _OVERFLOW_POLICIES:
        raise ValueError(f"overflow_policy must be one of {_OVERFLOW_POLICIES}")
    mask, ids, categories, crowd = _validate_map_segments(
        mask, segments, classes, void_value
    )
    spatial = tf.ensure_shape(tf.convert_to_tensor(pixel_valid_mask), [None, None])
    if spatial.dtype != tf.bool:
        raise TypeError("pixel_valid_mask must be boolean")
    tf.debugging.assert_equal(tf.shape(mask), tf.shape(spatial))
    row_valid = tf.ensure_shape(tf.convert_to_tensor(example_valid), [])
    if row_valid.dtype != tf.bool:
        raise TypeError("example_valid must be boolean")
    category, crowd_pixels = _lookup(mask, ids, categories, crowd)
    spatial &= row_valid & (mask != void_value) & ~crowd_pixels
    values = tf.constant(classes, tf.int32)
    thing_classes = tf.constant(things, tf.int32)
    isthing_class = tf.reduce_any(values[:, None] == thing_classes, axis=1)
    stuff_indices = tf.boolean_mask(tf.range(len(classes)), ~isthing_class)
    stuff_masks = (
        category[None] == tf.gather(values, stuff_indices)[:, None, None]
    ) & spatial[None]
    present_stuff = tf.reduce_any(stuff_masks, axis=[1, 2])
    stuff_indices = tf.boolean_mask(stuff_indices, present_stuff)
    stuff_masks = tf.boolean_mask(stuff_masks, present_stuff)
    class_index = tf.argmax(
        tf.cast(categories[:, None] == values, tf.int32),
        axis=1,
        output_type=tf.int32,
    )
    is_thing = tf.gather(isthing_class, class_index) & ~crowd
    segment_ids = tf.boolean_mask(ids, is_thing)
    class_index = tf.boolean_mask(class_index, is_thing)
    # Configured category order, then segment ID: the rows are sorted by ID.
    order = tf.argsort(class_index, stable=True)
    segment_ids, class_index = (
        tf.gather(segment_ids, order),
        tf.gather(class_index, order),
    )
    thing_masks = (mask[None] == segment_ids[:, None, None]) & spatial[None]
    present_thing = tf.reduce_any(thing_masks, axis=[1, 2])
    segment_ids = tf.boolean_mask(segment_ids, present_thing)
    class_index = tf.boolean_mask(class_index, present_thing)
    thing_masks = tf.boolean_mask(thing_masks, present_thing)
    own = _things_with_own_target(
        thing_masks,
        class_index,
        len(classes),
        max_segments - tf.size(stuff_indices),
        overflow_policy,
    )
    overflow = tf.reduce_any(tf.boolean_mask(thing_masks, ~own), axis=0)
    merged_indices = tf.zeros([0], tf.int32)
    if overflow_policy == "exclude_supervision":
        spatial &= ~overflow
    elif overflow_policy == "merge_by_category":
        merged_indices = tf.unique(tf.boolean_mask(class_index, ~own)).y
    merged_masks = (
        category[None] == tf.gather(values, merged_indices)[:, None, None]
    ) & overflow[None]
    segment_ids, class_index, thing_masks = (
        tf.boolean_mask(value, own) for value in (segment_ids, class_index, thing_masks)
    )
    indices = tf.concat([stuff_indices, class_index, merged_indices], 0)
    targets = tf.concat([stuff_masks, thing_masks, merged_masks], 0)
    void = tf.cast(void_value, tf.int64)
    target_ids = tf.concat(
        [
            tf.fill([tf.size(stuff_indices)], void),
            segment_ids,
            tf.fill([tf.size(merged_indices)], void),
        ],
        0,
    )
    thing_flags = tf.concat(
        [
            tf.zeros_like(stuff_indices, tf.bool),
            tf.ones_like(segment_ids, tf.bool),
            tf.zeros_like(merged_indices, tf.bool),
        ],
        0,
    )
    count = tf.size(indices)
    tf.debugging.assert_less_equal(
        count, max_segments, message="panoptic target capacity exceeded"
    )
    pad = max_segments - count
    return {
        "class_ids": tf.ensure_shape(
            tf.pad(tf.gather(values, indices), [[0, pad]]), [max_segments]
        ),
        "class_indices": tf.ensure_shape(tf.pad(indices, [[0, pad]]), [max_segments]),
        "segment_ids": tf.ensure_shape(tf.pad(target_ids, [[0, pad]]), [max_segments]),
        "is_thing": tf.ensure_shape(tf.pad(thing_flags, [[0, pad]]), [max_segments]),
        "masks": tf.ensure_shape(
            tf.pad(targets, [[0, pad], [0, 0], [0, 0]]), [max_segments, None, None]
        ),
        "target_valid_mask": tf.ensure_shape(
            tf.concat([tf.ones([count], tf.bool), tf.zeros([pad], tf.bool)], 0),
            [max_segments],
        ),
        "pixel_valid_mask": spatial,
        "num_targets": tf.cast(count, tf.int32),
        "num_overflow_things": tf.reduce_sum(tf.cast(~own, tf.int32)),
        "supervision_valid": tf.reduce_any(spatial),
    }


def with_panoptic_targets(
    sample,
    *,
    class_values,
    thing_class_values,
    void_value=0,
    max_segments,
    overflow_policy="error",
):
    if "targets" in sample:
        raise ValueError("sample already contains targets")
    if "pixel_valid_mask" not in sample:
        raise ValueError("sample requires pixel_valid_mask")
    targets = panoptic_map_to_targets(
        sample["panoptic_mask"],
        sample["segments"],
        class_values=class_values,
        thing_class_values=thing_class_values,
        void_value=void_value,
        max_segments=max_segments,
        pixel_valid_mask=sample["pixel_valid_mask"],
        example_valid=sample.get("padding_mask", True),
        overflow_policy=overflow_policy,
    )
    return sample | {"targets": targets}


__all__ = [
    "decode_panoptic_rgb",
    "validate_panoptic_sample",
    "panoptic_map_to_semantic",
    "panoptic_map_to_targets",
    "with_panoptic_targets",
]

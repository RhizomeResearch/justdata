"""Static-capacity class-mask targets from categorical semantic labels."""

import tensorflow as tf


_CATEGORICAL_DTYPES = (tf.uint8, tf.uint16, tf.int16, tf.int32, tf.int64)


def _class_contract(class_values, ignore_value) -> tuple[int, ...]:
    """Validate declared int32 class IDs and a distinct ignore ID."""
    values = tuple(class_values)
    if not values or any(type(value) is not int for value in values):
        raise ValueError("class_values must contain integer class IDs")
    if len(set(values)) != len(values):
        raise ValueError("class_values must be unique")
    if type(ignore_value) is not int or ignore_value in values:
        raise ValueError("ignore_value must be an integer distinct from class_values")
    if any(not -(2**31) <= value < 2**31 for value in (*values, ignore_value)):
        raise ValueError("class_values and ignore_value must fit int32")
    return values


def _categorical_mask(mask):
    """Return an integer HW mask, accepting a trailing singleton channel."""
    mask = tf.convert_to_tensor(mask)
    if mask.dtype not in _CATEGORICAL_DTYPES:
        raise TypeError(
            "categorical mask must have uint8/uint16/int16/int32/int64 dtype"
        )
    if mask.shape.rank == 3:
        mask = tf.squeeze(mask, axis=-1)
    return tf.ensure_shape(mask, [None, None])


def _assert_declared_labels(mask, class_values, ignore_value):
    """Require every mask value to be a declared class or the ignore value."""
    fill = tf.cast(ignore_value, tf.int64)
    tf.debugging.assert_greater_equal(
        fill,
        tf.constant(mask.dtype.min, tf.int64),
        message="ignore value does not fit mask dtype",
    )
    tf.debugging.assert_less_equal(
        fill,
        tf.constant(mask.dtype.max, tf.int64),
        message="ignore value does not fit mask dtype",
    )
    allowed = tf.concat([tf.cast(class_values, tf.int64), tf.reshape(fill, [1])], 0)
    labels = tf.unique(tf.reshape(tf.cast(mask, tf.int64), [-1])).y
    valid = tf.reduce_any(labels[:, None] == allowed[None, :], axis=1)
    tf.debugging.assert_equal(
        tf.reduce_all(valid),
        True,
        message="mask contains undeclared class/ignore values",
    )


def semantic_map_to_targets(
    mask,
    *,
    class_values,
    ignore_value,
    pixel_valid_mask,
    example_valid=True,
):
    """Convert one HW/HW1 semantic map to a fixed number of class-mask slots.

    Slots follow ``class_values`` order. A false ``supervision_valid`` means the
    whole example has no supervised target, including no query-classification
    loss. Source IDs, geometry, and instance metadata are carried by the sample
    wrapper rather than encoded in class masks.
    """
    values = _class_contract(class_values, ignore_value)

    spatial_valid = tf.convert_to_tensor(pixel_valid_mask)
    if spatial_valid.dtype != tf.bool:
        raise TypeError("pixel_valid_mask must be boolean")
    spatial_valid = tf.ensure_shape(spatial_valid, [None, None])
    row_valid = tf.convert_to_tensor(example_valid)
    if row_valid.dtype != tf.bool:
        raise TypeError("example_valid must be boolean")
    row_valid = tf.ensure_shape(row_valid, [])

    class_ids = tf.constant(values, tf.int32)
    if mask is None:
        tf.debugging.assert_equal(
            tf.reduce_any(spatial_valid),
            False,
            message="a missing mask requires entirely false pixel validity",
        )
        spatial_valid = tf.zeros_like(spatial_valid)
        masks = tf.zeros(
            tf.concat([[len(values)], tf.shape(spatial_valid)], axis=0), tf.bool
        )
    else:
        mask = _categorical_mask(mask)
        tf.debugging.assert_equal(
            tf.shape(mask),
            tf.shape(spatial_valid),
            message="mask/validity size mismatch",
        )
        _assert_declared_labels(mask, class_ids, ignore_value)
        labels = tf.cast(mask, tf.int64)
        spatial_valid = spatial_valid & (labels != ignore_value) & row_valid
        masks = tf.transpose(
            (labels[..., None] == tf.cast(class_ids, tf.int64))
            & spatial_valid[..., None],
            [2, 0, 1],
        )

    target_valid = tf.reduce_any(masks, axis=[1, 2])
    return {
        "class_ids": class_ids,
        "class_indices": tf.range(len(values), dtype=tf.int32),
        "masks": masks,
        "target_valid_mask": target_valid,
        "pixel_valid_mask": spatial_valid,
        "num_targets": tf.reduce_sum(tf.cast(target_valid, tf.int32)),
        "supervision_valid": tf.reduce_any(target_valid),
    }


def with_semantic_targets(sample, *, class_values, ignore_value):
    """Attach targets after paired geometry while preserving sample provenance."""
    if "targets" in sample:
        raise ValueError("sample already contains targets")
    if "pixel_valid_mask" not in sample:
        raise ValueError("sample requires pixel_valid_mask")
    targets = semantic_map_to_targets(
        sample.get("mask"),
        class_values=class_values,
        ignore_value=ignore_value,
        pixel_valid_mask=sample["pixel_valid_mask"],
        example_valid=sample.get("padding_mask", True),
    )
    return sample | {"targets": targets}


__all__ = ["semantic_map_to_targets", "with_semantic_targets"]

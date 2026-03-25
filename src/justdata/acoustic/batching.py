from __future__ import annotations

import tensorflow as tf


def pad_nested(value, pad_size, *, metadata_mode: str = "full"):
    """Pad the first dimension of tensors in nested dictionaries."""
    if isinstance(value, dict):
        padded = {}
        for key, child in value.items():
            child = pad_nested(child, pad_size, metadata_mode=metadata_mode)
            if child is not None:
                padded[key] = child
        return padded

    if metadata_mode != "full" and value.dtype == tf.string:
        return None

    pad_shape = tf.concat([[pad_size], tf.shape(value)[1:]], axis=0)
    if value.dtype == tf.string:
        fill = tf.fill(pad_shape, tf.constant("", dtype=tf.string))
    else:
        fill = tf.zeros(pad_shape, dtype=value.dtype)
    return tf.concat([value, fill], axis=0)


__all__ = ["pad_nested"]

"""Replay paired panoptic geometry and refresh visible segment metadata."""

import tensorflow as tf

from justdata.vision.encodings.panoptic_targets import (
    _lookup,
    validate_panoptic_sample,
)
from justdata.vision.geometry import _check_record, _replay_spatial


def _visible_segments(mask, segments, max_segments):
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
    found = (tf.gather(sorted_ids, pos) == source_ids) & segments["valid_mask"]
    areas = tf.where(found, tf.gather(counts, tf.gather(order, pos)), 0)
    boxes = tf.stack(
        [
            tf.gather(left, tf.gather(order, pos)),
            tf.gather(top, tf.gather(order, pos)),
            tf.gather(right, tf.gather(order, pos))
            - tf.gather(left, tf.gather(order, pos))
            + 1,
            tf.gather(bottom, tf.gather(order, pos))
            - tf.gather(top, tf.gather(order, pos))
            + 1,
        ],
        axis=-1,
    )
    boxes = tf.where(found[:, None], boxes, 0)
    kept = found & (areas > 0)
    ids = tf.boolean_mask(source_ids, kept)
    categories = tf.boolean_mask(segments["category_ids"], kept)
    crowd = tf.boolean_mask(segments["is_crowd"], kept)
    areas = tf.boolean_mask(areas, kept)
    boxes = tf.boolean_mask(boxes, kept)
    pad = max_segments - tf.size(ids)
    return {
        "segment_ids": tf.ensure_shape(tf.pad(ids, [[0, pad]]), [max_segments]),
        "category_ids": tf.ensure_shape(tf.pad(categories, [[0, pad]]), [max_segments]),
        "is_crowd": tf.ensure_shape(tf.pad(crowd, [[0, pad]]), [max_segments]),
        "valid_mask": tf.ensure_shape(
            tf.concat([tf.ones_like(ids, tf.bool), tf.zeros([pad], tf.bool)], 0),
            [max_segments],
        ),
        "area": tf.ensure_shape(tf.pad(areas, [[0, pad]]), [max_segments]),
        "bbox": tf.ensure_shape(tf.pad(boxes, [[0, pad], [0, 0]]), [max_segments, 4]),
    }


def replay_panoptic_geometry(
    sample, record, *, class_values, thing_class_values, void_value, max_segments
):
    """Replay a version-2 record, including integer map and crowd validity."""
    record = _check_record(record, versions=(2,))
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


__all__ = ["replay_panoptic_geometry"]

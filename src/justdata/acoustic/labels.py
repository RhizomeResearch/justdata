from __future__ import annotations

import math
from typing import Any, Mapping

import tensorflow as tf

from justdata.acoustic.configs import LabelTransformConfig


def _is_dense_vector(label: tf.Tensor, num_classes: int) -> bool:
    if label.shape.rank != 1:
        return False
    if label.shape[-1] is not None:
        return label.shape[-1] == num_classes
    return False


def _metadata_with_hard_label(
    label: tf.Tensor,
    config: LabelTransformConfig,
    metadata: dict,
) -> dict:
    if not config.keep_hard_label_in_metadata:
        return metadata
    if label.shape.rank not in (0, None):
        return metadata

    index = tf.cast(tf.reshape(label, []), tf.int64)
    metadata["hard_label"] = index
    if config.class_names is not None:
        names = tf.constant(config.class_names, dtype=tf.string)
        metadata["class_name"] = tf.gather(names, tf.cast(index, tf.int32))
    return metadata


def _require_num_classes(config: LabelTransformConfig) -> int:
    if config.num_classes is None:
        raise ValueError(f"`num_classes` is required for label mode {config.mode!r}.")
    return config.num_classes


def _one_hot(label: tf.Tensor, config: LabelTransformConfig) -> tf.Tensor:
    num_classes = _require_num_classes(config)
    if _is_dense_vector(label, num_classes):
        return tf.cast(label, tf.float32)

    off_value = config.smoothing / float(num_classes)
    on_value = 1.0 - config.smoothing + off_value
    return tf.one_hot(
        tf.cast(tf.reshape(label, []), tf.int32),
        num_classes,
        on_value=on_value,
        off_value=off_value,
        dtype=tf.float32,
    )


def _multi_hot(label: tf.Tensor, config: LabelTransformConfig) -> tf.Tensor:
    num_classes = _require_num_classes(config)
    if _is_dense_vector(label, num_classes):
        values = tf.cast(label, tf.float32)
        if label.dtype.is_floating:
            return values
        is_binary = tf.reduce_all(
            tf.logical_or(tf.equal(label, 0), tf.equal(label, 1))
        )

        def scatter_indices() -> tf.Tensor:
            indices = tf.cast(tf.reshape(label, [-1]), tf.int32)
            updates = tf.ones(tf.shape(indices), dtype=tf.float32)
            return tf.tensor_scatter_nd_max(
                tf.zeros([num_classes], dtype=tf.float32),
                indices[:, tf.newaxis],
                updates,
            )

        return tf.cond(is_binary, lambda: values, scatter_indices)

    indices = tf.cast(tf.reshape(label, [-1]), tf.int32)
    updates = tf.ones(tf.shape(indices), dtype=tf.float32)
    return tf.tensor_scatter_nd_max(
        tf.zeros([num_classes], dtype=tf.float32),
        indices[:, tf.newaxis],
        updates,
    )


def _event_value(event: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in event:
            return event[key]
    return default


def _event_frames(
    label: Any,
    config: LabelTransformConfig,
    metadata: dict,
) -> tf.Tensor:
    num_classes = _require_num_classes(config)
    num_frames = config.num_frames
    if num_frames is None:
        num_frames = metadata.get("num_frames")
    if num_frames is None:
        raise ValueError("`num_frames` is required for event_frames labels.")

    hop_seconds = config.frame_hop_seconds
    if hop_seconds is None and config.hop_length is not None and config.sample_rate:
        hop_seconds = config.hop_length / float(config.sample_rate)
    if hop_seconds is None:
        hop_seconds = metadata.get("frame_hop_seconds")
    if hop_seconds is None and "hop_length" in metadata and "sample_rate" in metadata:
        hop_seconds = metadata["hop_length"] / float(metadata["sample_rate"])
    if hop_seconds is None:
        raise ValueError("A frame hop duration is required for event_frames labels.")

    frames = tf.zeros([int(num_frames), num_classes], dtype=tf.float32)
    events = label
    if isinstance(events, Mapping):
        events = events.get("events", [])

    for event in events:
        start = float(_event_value(event, "start_time", "start", default=0.0))
        end = float(_event_value(event, "end_time", "end", default=start))
        class_id = int(_event_value(event, "class_id", "label", "label_id", default=0))
        start_frame = max(int(start // hop_seconds), 0)
        end_frame = min(math.ceil(end / hop_seconds), int(num_frames))
        if end_frame <= start_frame:
            end_frame = min(start_frame + 1, int(num_frames))
        indices = [[i, class_id] for i in range(start_frame, end_frame)]
        if indices:
            frames = tf.tensor_scatter_nd_update(
                frames,
                indices,
                tf.ones([len(indices)], dtype=tf.float32),
            )
    return frames


def transform_label(
    label: Any,
    config: LabelTransformConfig,
    *,
    metadata: dict | None = None,
) -> tuple[tf.Tensor | str, dict]:
    metadata_out = dict(metadata or {})

    if config.mode == "text":
        return tf.convert_to_tensor(label, dtype=tf.string), metadata_out

    if config.mode == "event_frames":
        return _event_frames(label, config, metadata_out), metadata_out

    label_tensor = tf.convert_to_tensor(label)
    if config.mode == "index":
        label_tensor = tf.cast(tf.reshape(label_tensor, []), tf.int64)
        metadata_out = _metadata_with_hard_label(label_tensor, config, metadata_out)
        return label_tensor, metadata_out

    if config.mode == "one_hot":
        metadata_out = _metadata_with_hard_label(label_tensor, config, metadata_out)
        return _one_hot(label_tensor, config), metadata_out

    if config.mode == "multi_hot":
        return _multi_hot(label_tensor, config), metadata_out

    raise ValueError(f"Unsupported label transform mode: {config.mode!r}")


__all__ = ["transform_label"]

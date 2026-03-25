from __future__ import annotations

import tensorflow as tf

from justdata.acoustic.corruptions.registry import (
    _EPS,
    _severity_value,
    register_audio_corruption,
)


CLIPPING_THRESHOLD = {1: 0.95, 2: 0.8, 3: 0.6, 4: 0.4, 5: 0.25}
DRC_THRESHOLD_DB = {1: -3, 2: -6, 3: -12, 4: -18, 5: -24}
DRC_RATIO = {1: 1.5, 2: 2.0, 3: 4.0, 4: 8.0, 5: 12.0}


@register_audio_corruption("clipping")
def clipping(
    audio_or_features: tf.Tensor,
    severity: int,
    seed: tf.Tensor,
    config: dict | None = None,
) -> tf.Tensor:
    del seed, config
    threshold = _severity_value(CLIPPING_THRESHOLD, severity)
    x = tf.cast(audio_or_features, tf.float32)
    return tf.clip_by_value(x, -threshold, threshold)


@register_audio_corruption("dynamic_range_compression")
def dynamic_range_compression(
    audio_or_features: tf.Tensor,
    severity: int,
    seed: tf.Tensor,
    config: dict | None = None,
) -> tf.Tensor:
    del seed, config
    threshold_db = _severity_value(DRC_THRESHOLD_DB, severity)
    ratio = _severity_value(DRC_RATIO, severity)
    x = tf.cast(audio_or_features, tf.float32)
    magnitude = tf.abs(x)
    level_db = 20.0 * tf.math.log(magnitude + _EPS) / tf.math.log(10.0)
    over = level_db - threshold_db
    gain_db = tf.where(over > 0.0, over / ratio - over, tf.zeros_like(over))
    gain = tf.pow(tf.constant(10.0, dtype=tf.float32), gain_db / 20.0)
    return tf.sign(x) * magnitude * gain


__all__ = [
    "CLIPPING_THRESHOLD",
    "DRC_RATIO",
    "DRC_THRESHOLD_DB",
    "clipping",
    "dynamic_range_compression",
]

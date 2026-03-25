from __future__ import annotations

import tensorflow as tf

from justdata.acoustic.corruptions.registry import (
    _as_tc,
    _fit_length,
    _resize_time,
    _restore_rank,
    _sample_rate,
    _severity_value,
    register_audio_corruption,
)


CODEC_BITRATE_KBPS = {1: 128, 2: 96, 3: 64, 4: 32, 5: 16}
CODEC_MULAW_LEVELS = {1: 256, 2: 128, 3: 64, 4: 32, 5: 16}
RESAMPLING_INTERMEDIATE_HZ_32K = {1: 24000, 2: 16000, 3: 12000, 4: 8000, 5: 4000}


def _quantize_unit(x: tf.Tensor, levels: tf.Tensor) -> tf.Tensor:
    levels = tf.cast(tf.maximum(levels, 2), tf.float32)
    x = tf.clip_by_value(x, -1.0, 1.0)
    return tf.round((x + 1.0) * 0.5 * (levels - 1.0)) / (levels - 1.0) * 2.0 - 1.0


def _mulaw_roundtrip(x: tf.Tensor, levels: tf.Tensor) -> tf.Tensor:
    mu = tf.cast(tf.maximum(levels - 1, 1), tf.float32)
    log_mu = tf.math.log1p(mu)
    encoded = tf.sign(x) * tf.math.log1p(mu * tf.abs(x)) / log_mu
    encoded = _quantize_unit(encoded, levels)
    return tf.sign(encoded) * tf.math.expm1(tf.abs(encoded) * log_mu) / mu


@register_audio_corruption("codec_compression")
def codec_compression(
    audio_or_features: tf.Tensor,
    severity: int,
    seed: tf.Tensor,
    config: dict | None = None,
) -> tf.Tensor:
    del seed
    levels = _severity_value(CODEC_MULAW_LEVELS, severity, dtype=tf.int32)
    if config is not None and "levels" in config:
        levels = tf.cast(config["levels"], tf.int32)
    x = tf.cast(audio_or_features, tf.float32)
    clipped = tf.clip_by_value(x, -1.0, 1.0)
    return tf.clip_by_value(_mulaw_roundtrip(clipped, levels), -1.0, 1.0)


@register_audio_corruption("resampling_degradation")
def resampling_degradation(
    audio_or_features: tf.Tensor,
    severity: int,
    seed: tf.Tensor,
    config: dict | None = None,
) -> tf.Tensor:
    del seed
    sample_rate = _sample_rate(config)
    intermediate_hz = _severity_value(
        RESAMPLING_INTERMEDIATE_HZ_32K,
        severity,
        dtype=tf.int32,
    )
    if config is not None and "intermediate_sample_rate" in config:
        intermediate_hz = tf.cast(config["intermediate_sample_rate"], tf.int32)

    audio, rank = _as_tc(audio_or_features)
    length = tf.shape(audio)[0]
    intermediate_length = tf.cast(
        tf.maximum(
            tf.round(
                tf.cast(length, tf.float32)
                * tf.cast(intermediate_hz, tf.float32)
                / sample_rate
            ),
            1.0,
        ),
        tf.int32,
    )
    degraded = _resize_time(audio, intermediate_length)
    restored = _fit_length(_resize_time(degraded, length), length)
    return _restore_rank(restored, rank)


__all__ = [
    "CODEC_BITRATE_KBPS",
    "CODEC_MULAW_LEVELS",
    "RESAMPLING_INTERMEDIATE_HZ_32K",
    "codec_compression",
    "resampling_degradation",
]

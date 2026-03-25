from __future__ import annotations

import tensorflow as tf

from justdata.acoustic.corruptions.registry import (
    _as_tc,
    _fit_length,
    _resize_time,
    _restore_rank,
    _sample_rate,
    _severity_value,
    _split_seed,
    register_audio_corruption,
)


TIME_STRETCH_FACTOR = {1: 1.02, 2: 1.05, 3: 1.10, 4: 1.20, 5: 1.35}
PITCH_SHIFT_SEMITONES = {1: 0.5, 2: 1.0, 3: 2.0, 4: 3.0, 5: 4.0}
PACKET_DROPOUT_PROBABILITY = {1: 0.01, 2: 0.025, 3: 0.05, 4: 0.10, 5: 0.20}
PACKET_DROPOUT_GAP_MS = {1: 5, 2: 10, 3: 20, 4: 30, 5: 50}


@register_audio_corruption("time_stretch")
def time_stretch(
    audio_or_features: tf.Tensor,
    severity: int,
    seed: tf.Tensor,
    config: dict | None = None,
) -> tf.Tensor:
    del seed
    factor = _severity_value(TIME_STRETCH_FACTOR, severity)
    if config is not None and "factor" in config:
        factor = tf.cast(config["factor"], tf.float32)
    audio, rank = _as_tc(audio_or_features)
    length = tf.shape(audio)[0]
    stretched_length = tf.cast(
        tf.maximum(tf.round(tf.cast(length, tf.float32) * factor), 1.0),
        tf.int32,
    )
    stretched = _resize_time(audio, stretched_length)
    return _restore_rank(_fit_length(stretched, length), rank)


@register_audio_corruption("pitch_shift")
def pitch_shift(
    audio_or_features: tf.Tensor,
    severity: int,
    seed: tf.Tensor,
    config: dict | None = None,
) -> tf.Tensor:
    del seed
    semitones = _severity_value(PITCH_SHIFT_SEMITONES, severity)
    if config is not None and "semitones" in config:
        semitones = tf.cast(config["semitones"], tf.float32)
    audio, rank = _as_tc(audio_or_features)
    length = tf.shape(audio)[0]
    factor = tf.pow(tf.constant(2.0, dtype=tf.float32), semitones / 12.0)
    shifted_length = tf.cast(
        tf.maximum(tf.round(tf.cast(length, tf.float32) / factor), 1.0),
        tf.int32,
    )
    shifted = _resize_time(audio, shifted_length)
    return _restore_rank(_fit_length(shifted, length), rank)


@register_audio_corruption("packet_dropout_gaps")
def packet_dropout_gaps(
    audio_or_features: tf.Tensor,
    severity: int,
    seed: tf.Tensor,
    config: dict | None = None,
) -> tf.Tensor:
    sample_rate = _sample_rate(config)
    probability = _severity_value(PACKET_DROPOUT_PROBABILITY, severity)
    gap_ms = _severity_value(PACKET_DROPOUT_GAP_MS, severity)
    if config is not None:
        if "probability" in config:
            probability = tf.cast(config["probability"], tf.float32)
        if "gap_ms" in config:
            gap_ms = tf.cast(config["gap_ms"], tf.float32)

    dropout_seed, fallback_seed = tf.unstack(_split_seed(seed, 2))
    audio, rank = _as_tc(audio_or_features)
    length = tf.shape(audio)[0]
    gap_length = tf.cast(
        tf.maximum(tf.round(sample_rate * gap_ms / 1000.0), 1.0),
        tf.int32,
    )
    num_segments = tf.cast(
        tf.maximum(
            tf.math.ceil(tf.cast(length, tf.float32) / tf.cast(gap_length, tf.float32)),
            1.0,
        ),
        tf.int32,
    )
    random_values = tf.random.stateless_uniform([num_segments], seed=dropout_seed)
    dropped = random_values < probability

    def add_fallback_drop() -> tf.Tensor:
        index = tf.random.stateless_uniform(
            [],
            seed=fallback_seed,
            minval=0,
            maxval=num_segments,
            dtype=tf.int32,
        )
        return tf.tensor_scatter_nd_update(
            dropped,
            indices=tf.reshape(index, [1, 1]),
            updates=tf.constant([True]),
        )

    dropped = tf.cond(tf.reduce_any(dropped), lambda: dropped, add_fallback_drop)
    mask = tf.repeat(dropped, gap_length)[:length]
    corrupted = tf.where(mask[:, tf.newaxis], tf.zeros_like(audio), audio)
    return _restore_rank(corrupted, rank)


__all__ = [
    "PACKET_DROPOUT_GAP_MS",
    "PACKET_DROPOUT_PROBABILITY",
    "PITCH_SHIFT_SEMITONES",
    "TIME_STRETCH_FACTOR",
    "packet_dropout_gaps",
    "pitch_shift",
    "time_stretch",
]

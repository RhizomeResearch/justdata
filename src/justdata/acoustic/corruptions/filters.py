from __future__ import annotations

import tensorflow as tf

from justdata.acoustic.corruptions.registry import (
    _frequency_bins,
    _rfft_filter,
    _sample_rate,
    _severity_value,
    register_audio_corruption,
)


BANDPASS_HZ_32K = {
    1: (80, 14000),
    2: (120, 10000),
    3: (200, 7000),
    4: (400, 5000),
    5: (800, 3500),
}
LOWPASS_HZ_32K = {1: 12000, 2: 9000, 3: 6000, 4: 4000, 5: 2500}
HIGHPASS_HZ_32K = {1: 40, 2: 120, 3: 400, 4: 1000, 5: 2500}
EQUALIZATION_TILT_DB_PER_OCTAVE = {1: -1, 2: -2, 3: -4, 4: -6, 5: -9}


@register_audio_corruption("low_pass")
def low_pass(
    audio_or_features: tf.Tensor,
    severity: int,
    seed: tf.Tensor,
    config: dict | None = None,
) -> tf.Tensor:
    del seed
    sample_rate = _sample_rate(config)
    cutoff = _severity_value(LOWPASS_HZ_32K, severity)
    freqs = _frequency_bins(audio_or_features, sample_rate)
    response = tf.cast(freqs <= cutoff, tf.float32)
    return _rfft_filter(audio_or_features, response)


@register_audio_corruption("high_pass")
def high_pass(
    audio_or_features: tf.Tensor,
    severity: int,
    seed: tf.Tensor,
    config: dict | None = None,
) -> tf.Tensor:
    del seed
    sample_rate = _sample_rate(config)
    cutoff = _severity_value(HIGHPASS_HZ_32K, severity)
    freqs = _frequency_bins(audio_or_features, sample_rate)
    response = tf.cast(freqs >= cutoff, tf.float32)
    return _rfft_filter(audio_or_features, response)


@register_audio_corruption("band_pass")
def band_pass(
    audio_or_features: tf.Tensor,
    severity: int,
    seed: tf.Tensor,
    config: dict | None = None,
) -> tf.Tensor:
    del seed
    sample_rate = _sample_rate(config)
    low_high = _severity_value(BANDPASS_HZ_32K, severity)
    low = low_high[0]
    high = low_high[1]
    freqs = _frequency_bins(audio_or_features, sample_rate)
    response = tf.cast(tf.logical_and(freqs >= low, freqs <= high), tf.float32)
    return _rfft_filter(audio_or_features, response)


@register_audio_corruption("equalization_tilt")
def equalization_tilt(
    audio_or_features: tf.Tensor,
    severity: int,
    seed: tf.Tensor,
    config: dict | None = None,
) -> tf.Tensor:
    del seed
    sample_rate = _sample_rate(config)
    ref_hz = 1000.0 if config is None else float(config.get("reference_hz", 1000.0))
    tilt = _severity_value(EQUALIZATION_TILT_DB_PER_OCTAVE, severity)
    freqs = _frequency_bins(audio_or_features, sample_rate)
    safe_freqs = tf.maximum(freqs, tf.constant(20.0, dtype=tf.float32))
    octaves = tf.math.log(safe_freqs / tf.cast(ref_hz, tf.float32)) / tf.math.log(2.0)
    gain_db = tilt * octaves
    response = tf.clip_by_value(
        tf.pow(tf.constant(10.0, dtype=tf.float32), gain_db / 20.0),
        0.125,
        4.0,
    )
    return _rfft_filter(audio_or_features, response)


__all__ = [
    "BANDPASS_HZ_32K",
    "EQUALIZATION_TILT_DB_PER_OCTAVE",
    "HIGHPASS_HZ_32K",
    "LOWPASS_HZ_32K",
    "band_pass",
    "equalization_tilt",
    "high_pass",
    "low_pass",
]

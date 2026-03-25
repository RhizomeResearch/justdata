from __future__ import annotations

import tensorflow as tf

from justdata.acoustic.corruptions.registry import (
    _EPS,
    _as_tc,
    _restore_rank,
    _severity_value,
    _split_seed,
    register_audio_corruption,
)


WHITE_NOISE_SNR_DB = {1: 30, 2: 20, 3: 10, 4: 5, 5: 0}
PINK_NOISE_SNR_DB = {1: 30, 2: 20, 3: 10, 4: 5, 5: 0}
BROWN_NOISE_SNR_DB = {1: 30, 2: 20, 3: 10, 4: 5, 5: 0}
BACKGROUND_NOISE_SNR_DB = {1: 25, 2: 18, 3: 12, 4: 6, 5: 0}


def _normalize_noise(noise: tf.Tensor) -> tf.Tensor:
    noise = tf.cast(noise, tf.float32)
    axes = tf.range(tf.rank(noise))
    noise = noise - tf.reduce_mean(noise, axis=axes, keepdims=True)
    rms = tf.sqrt(tf.reduce_mean(tf.square(noise), axis=axes, keepdims=True))
    return tf.math.divide_no_nan(noise, tf.maximum(rms, _EPS))


def _colored_noise(shape: tf.Tensor, seed: tf.Tensor, kind: str) -> tf.Tensor:
    noise = tf.random.stateless_normal(shape, seed=seed, dtype=tf.float32)
    if kind == "white":
        return _normalize_noise(noise)

    audio, rank = _as_tc(noise)
    if kind == "brown":
        return _restore_rank(_normalize_noise(tf.cumsum(audio, axis=0)), rank)
    if kind != "pink":
        raise ValueError("noise kind must be one of 'white', 'pink', or 'brown'.")

    time = tf.shape(audio)[0]
    channels_first = tf.transpose(audio, [1, 0])
    spectrum = tf.signal.rfft(channels_first, fft_length=[time])
    num_bins = tf.shape(spectrum)[-1]
    freqs = tf.cast(tf.range(num_bins), tf.float32)
    weights = tf.where(freqs > 0.0, tf.math.rsqrt(freqs), tf.zeros_like(freqs))
    pink = tf.signal.irfft(
        spectrum * tf.cast(weights[tf.newaxis, :], spectrum.dtype),
        fft_length=[time],
    )
    return _restore_rank(_normalize_noise(tf.transpose(pink, [1, 0])), rank)


def _add_at_snr(
    audio_or_features: tf.Tensor, noise: tf.Tensor, snr_db: tf.Tensor
) -> tf.Tensor:
    x = tf.cast(audio_or_features, tf.float32)
    noise = tf.cast(noise, tf.float32)
    signal_power = tf.reduce_mean(tf.square(x))
    noise_power = tf.reduce_mean(tf.square(noise))
    target_ratio = tf.pow(tf.constant(10.0, dtype=tf.float32), snr_db / 10.0)
    scale = tf.sqrt(tf.math.divide_no_nan(signal_power, noise_power * target_ratio))
    return x + noise * scale


def _add_colored_noise(
    audio_or_features: tf.Tensor,
    severity: int,
    seed: tf.Tensor,
    *,
    kind: str,
    snr_table: dict[int, int],
) -> tf.Tensor:
    snr_db = _severity_value(snr_table, severity)
    noise = _colored_noise(tf.shape(audio_or_features), seed, kind)
    return _add_at_snr(audio_or_features, noise, snr_db)


@register_audio_corruption("additive_white_noise")
def additive_white_noise(
    audio_or_features: tf.Tensor,
    severity: int,
    seed: tf.Tensor,
    config: dict | None = None,
) -> tf.Tensor:
    del config
    return _add_colored_noise(
        audio_or_features,
        severity,
        seed,
        kind="white",
        snr_table=WHITE_NOISE_SNR_DB,
    )


@register_audio_corruption("additive_pink_noise")
def additive_pink_noise(
    audio_or_features: tf.Tensor,
    severity: int,
    seed: tf.Tensor,
    config: dict | None = None,
) -> tf.Tensor:
    del config
    return _add_colored_noise(
        audio_or_features,
        severity,
        seed,
        kind="pink",
        snr_table=PINK_NOISE_SNR_DB,
    )


@register_audio_corruption("additive_brown_noise")
def additive_brown_noise(
    audio_or_features: tf.Tensor,
    severity: int,
    seed: tf.Tensor,
    config: dict | None = None,
) -> tf.Tensor:
    del config
    return _add_colored_noise(
        audio_or_features,
        severity,
        seed,
        kind="brown",
        snr_table=BROWN_NOISE_SNR_DB,
    )


@register_audio_corruption("background_noise")
def background_noise(
    audio_or_features: tf.Tensor,
    severity: int,
    seed: tf.Tensor,
    config: dict | None = None,
) -> tf.Tensor:
    snr_db = _severity_value(BACKGROUND_NOISE_SNR_DB, severity)
    noise_seed, hum_seed = tf.unstack(_split_seed(seed, 2))

    configured = None if config is None else config.get("background")
    if configured is not None:
        background = tf.cast(configured, tf.float32)
    else:
        x = tf.convert_to_tensor(audio_or_features)
        background = _colored_noise(tf.shape(x), noise_seed, "pink")
        time = tf.cast(tf.shape(x)[0], tf.float32)
        positions = tf.range(tf.shape(x)[0], dtype=tf.float32)
        cycles = tf.random.stateless_uniform(
            [],
            seed=hum_seed,
            minval=1.0,
            maxval=8.0,
            dtype=tf.float32,
        )
        hum = tf.sin(
            2.0 * 3.141592653589793 * cycles * positions / tf.maximum(time, 1.0)
        )
        if x.shape.rank == 2:
            hum = hum[..., tf.newaxis]
        background = background + 0.25 * hum

    return _add_at_snr(audio_or_features, background, snr_db)


__all__ = [
    "BACKGROUND_NOISE_SNR_DB",
    "BROWN_NOISE_SNR_DB",
    "PINK_NOISE_SNR_DB",
    "WHITE_NOISE_SNR_DB",
    "additive_brown_noise",
    "additive_pink_noise",
    "additive_white_noise",
    "background_noise",
]

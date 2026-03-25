from __future__ import annotations

from collections.abc import Callable

import tensorflow as tf

from justdata.acoustic.configs import FrontendConfig, STFTConfig
from justdata.acoustic.normalization import apply_feature_normalization
from justdata.acoustic.registry import register_audio_frontend


def ensure_waveform_tc(audio: tf.Tensor) -> tf.Tensor:
    audio = tf.cast(tf.convert_to_tensor(audio), tf.float32)
    if audio.shape.rank == 1:
        audio = tf.expand_dims(audio, -1)
    if audio.shape.rank != 2:
        raise ValueError(
            "Audio frontend input must be a waveform with shape [T] or [T, C]"
        )
    return audio


def _window_fn(
    config: STFTConfig,
) -> Callable[[tf.Tensor, tf.dtypes.DType], tf.Tensor] | None:
    if config.window == "rectangular":
        return lambda frame_length, dtype: tf.ones([frame_length], dtype=dtype)

    def _hann(frame_length: tf.Tensor, dtype: tf.dtypes.DType) -> tf.Tensor:
        return tf.signal.hann_window(
            frame_length,
            periodic=config.window_periodic,
            dtype=dtype,
        )

    if config.window == "hann":
        return _hann

    if config.window == "hamming":
        return lambda frame_length, dtype: tf.signal.hamming_window(
            frame_length,
            periodic=config.window_periodic,
            dtype=dtype,
        )

    if config.window == "povey":
        return lambda frame_length, dtype: tf.pow(_hann(frame_length, dtype), 0.85)

    raise ValueError(f"Unknown STFT window: {config.window!r}")


def reflect_or_constant_pad(audio: tf.Tensor, pad: int, pad_mode: str) -> tf.Tensor:
    if pad <= 0:
        return audio

    paddings = [[pad, pad], [0, 0]]
    if pad_mode == "constant":
        return tf.pad(audio, paddings, mode="CONSTANT")
    if pad_mode != "reflect":
        raise ValueError(f"Unknown STFT pad_mode: {pad_mode!r}")

    return tf.cond(
        tf.shape(audio)[0] > pad,
        lambda: tf.pad(audio, paddings, mode="REFLECT"),
        lambda: tf.pad(audio, paddings, mode="CONSTANT"),
    )


def _full_spectrum_from_onesided(spectrum: tf.Tensor, n_fft: int) -> tf.Tensor:
    if n_fft % 2 == 0:
        reflected = spectrum[:, 1:-1, :]
    else:
        reflected = spectrum[:, 1:, :]
    return tf.concat([spectrum, tf.reverse(reflected, axis=[1])], axis=1)


def stft_power_spectrogram(audio: tf.Tensor, config: STFTConfig | dict) -> tf.Tensor:
    config = STFTConfig.from_dict(config)
    audio = ensure_waveform_tc(audio)

    if config.center:
        audio = reflect_or_constant_pad(audio, config.n_fft // 2, config.pad_mode)

    channels_first = tf.transpose(audio, [1, 0])
    frames = tf.signal.stft(
        signals=channels_first,
        frame_length=config.win_length,
        frame_step=config.hop_length,
        fft_length=config.n_fft,
        window_fn=_window_fn(config),
        pad_end=False,
    )
    frames = tf.transpose(frames, [1, 2, 0])

    if config.normalized:
        frames = frames / tf.cast(
            tf.sqrt(tf.cast(config.win_length, tf.float32)), frames.dtype
        )

    magnitude = tf.abs(frames)
    if config.power == 1:
        spectrogram = magnitude
    elif config.power == 2:
        spectrogram = tf.square(magnitude)
    else:
        spectrogram = tf.pow(
            tf.maximum(magnitude, tf.cast(config.eps, magnitude.dtype)), config.power
        )

    if not config.onesided:
        spectrogram = _full_spectrum_from_onesided(spectrogram, config.n_fft)

    return tf.cast(spectrogram, tf.float32)


@register_audio_frontend("raw_waveform")
def raw_waveform(audio: tf.Tensor, config: FrontendConfig | dict) -> tf.Tensor:
    config = FrontendConfig.from_dict(config)
    waveform = ensure_waveform_tc(audio)
    return apply_feature_normalization(waveform, config.norm)


@register_audio_frontend("stft_magnitude")
def stft_magnitude(audio: tf.Tensor, config: FrontendConfig | dict) -> tf.Tensor:
    config = FrontendConfig.from_dict(config)
    if config.stft is None:
        raise ValueError("stft_magnitude frontend requires config.stft")
    features = stft_power_spectrogram(audio, config.stft)
    return apply_feature_normalization(features, config.norm)


def compute_num_frames(
    num_samples: int,
    *,
    n_fft: int,
    win_length: int,
    hop_length: int,
    center: bool,
) -> int:
    pad = n_fft // 2 if center else 0
    numerator = num_samples + 2 * pad - win_length
    if numerator < 0:
        return 0
    return 1 + numerator // hop_length


__all__ = [
    "compute_num_frames",
    "ensure_waveform_tc",
    "raw_waveform",
    "reflect_or_constant_pad",
    "stft_magnitude",
    "stft_power_spectrogram",
]

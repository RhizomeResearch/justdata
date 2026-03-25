from __future__ import annotations

import tensorflow as tf

from justdata.acoustic.configs import FrontendConfig
from justdata.acoustic.frontends.stft import ensure_waveform_tc
from justdata.acoustic.registry import register_audio_frontend


_KALDI_EPSILON = tf.constant(1.1920928955078125e-7, dtype=tf.float32)


def _next_power_of_two(value: int) -> int:
    return 1 if value == 0 else 2 ** (value - 1).bit_length()


def _kaldi_hz_to_mel(frequencies: tf.Tensor) -> tf.Tensor:
    return 1127.0 * tf.math.log1p(frequencies / 700.0)


def _kaldi_mel_to_hz(mels: tf.Tensor) -> tf.Tensor:
    return 700.0 * (tf.exp(mels / 1127.0) - 1.0)


def _ast_mel_banks(
    *,
    num_mel_bins: int,
    padded_window_size: int,
    sample_rate: int,
    low_freq: float = 20.0,
    high_freq: float = 0.0,
) -> tf.Tensor:
    nyquist = 0.5 * float(sample_rate)
    if high_freq <= 0.0:
        high_freq += nyquist

    fft_bin_width = float(sample_rate) / float(padded_window_size)
    mel_low = 1127.0 * tf.math.log1p(tf.constant(low_freq, dtype=tf.float32) / 700.0)
    mel_high = 1127.0 * tf.math.log1p(tf.constant(high_freq, dtype=tf.float32) / 700.0)
    mel_delta = (mel_high - mel_low) / float(num_mel_bins + 1)

    bins = tf.cast(tf.range(num_mel_bins), tf.float32)[:, tf.newaxis]
    left_mel = mel_low + bins * mel_delta
    center_mel = mel_low + (bins + 1.0) * mel_delta
    right_mel = mel_low + (bins + 2.0) * mel_delta

    num_fft_bins = padded_window_size // 2
    mel = _kaldi_hz_to_mel(
        fft_bin_width * tf.cast(tf.range(num_fft_bins), tf.float32)
    )[tf.newaxis, :]

    up_slope = (mel - left_mel) / (center_mel - left_mel)
    down_slope = (right_mel - mel) / (right_mel - center_mel)
    weights = tf.maximum(tf.zeros([], dtype=tf.float32), tf.minimum(up_slope, down_slope))
    return tf.pad(weights, [[0, 0], [0, 1]])


@register_audio_frontend("ast_kaldi_fbank")
def ast_kaldi_fbank(audio: tf.Tensor, config: FrontendConfig | dict) -> tf.Tensor:
    """AST's torchaudio.compliance.kaldi.fbank subset.

    This matches the AST dataloader settings:
    htk_compat=True, use_energy=False, window_type="hanning", dither=0.0,
    frame_shift=10 ms, and 128 mel bins by default.
    """

    config = FrontendConfig.from_dict(config)
    if config.stft is None:
        raise ValueError("ast_kaldi_fbank frontend requires config.stft")
    if config.mel is None:
        raise ValueError("ast_kaldi_fbank frontend requires config.mel")

    stft = config.stft
    mel = config.mel
    waveform = ensure_waveform_tc(audio)[:, 0]
    waveform = waveform - tf.reduce_mean(waveform)

    padded_window_size = _next_power_of_two(stft.win_length)
    if stft.n_fft != padded_window_size:
        raise ValueError("AST fbank expects n_fft to be the next power of two win_length")

    frames = tf.signal.frame(
        waveform,
        frame_length=stft.win_length,
        frame_step=stft.hop_length,
        pad_end=False,
    )
    frames = tf.cast(frames, tf.float32)
    frames = frames - tf.reduce_mean(frames, axis=1, keepdims=True)

    previous = tf.concat([frames[:, :1], frames[:, :-1]], axis=1)
    frames = frames - 0.97 * previous

    window = tf.signal.hann_window(stft.win_length, periodic=False, dtype=tf.float32)
    frames = frames * window[tf.newaxis, :]
    frames = tf.pad(frames, [[0, 0], [0, padded_window_size - stft.win_length]])

    spectrum = tf.abs(tf.signal.rfft(frames))
    spectrum = tf.square(spectrum)
    weights = _ast_mel_banks(
        num_mel_bins=mel.n_mels,
        padded_window_size=padded_window_size,
        sample_rate=stft.sample_rate,
        low_freq=mel.f_min,
        high_freq=0.0 if mel.f_max is None else mel.f_max,
    )
    mel_energies = tf.matmul(spectrum, weights, transpose_b=True)
    mel_energies = tf.math.log(tf.maximum(mel_energies, _KALDI_EPSILON))
    return mel_energies[:, :, tf.newaxis]


__all__ = ["ast_kaldi_fbank"]

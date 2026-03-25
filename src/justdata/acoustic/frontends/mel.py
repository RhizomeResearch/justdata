from __future__ import annotations

import tensorflow as tf

from justdata.acoustic.configs import FrontendConfig, MelConfig, STFTConfig
from justdata.acoustic.normalization import apply_feature_normalization
from justdata.acoustic.registry import register_audio_frontend
from justdata.acoustic.frontends.stft import stft_power_spectrogram


def _f_max(config: MelConfig, sample_rate: int) -> float:
    f_max = float(sample_rate) / 2.0 if config.f_max is None else float(config.f_max)
    if f_max > float(sample_rate) / 2.0:
        raise ValueError("f_max must not exceed Nyquist frequency")
    return f_max


def _kaldi_hz_to_mel(frequencies: tf.Tensor) -> tf.Tensor:
    return 1127.0 * tf.math.log1p(frequencies / 700.0)


def _kaldi_mel_to_hz(mels: tf.Tensor) -> tf.Tensor:
    return 700.0 * (tf.exp(mels / 1127.0) - 1.0)


def _kaldi_mel_weight_matrix(config: MelConfig, stft: STFTConfig) -> tf.Tensor:
    num_bins = stft.n_fft // 2 + 1
    f_max = _f_max(config, stft.sample_rate)

    low_mel = _kaldi_hz_to_mel(tf.constant(float(config.f_min), dtype=tf.float32))
    high_mel = _kaldi_hz_to_mel(tf.constant(f_max, dtype=tf.float32))
    mel_points = tf.linspace(low_mel, high_mel, config.n_mels + 2)
    hz_points = _kaldi_mel_to_hz(mel_points)
    frequencies = tf.linspace(0.0, float(stft.sample_rate) / 2.0, num_bins)

    lower = hz_points[:-2]
    center = hz_points[1:-1]
    upper = hz_points[2:]
    frequencies = tf.expand_dims(frequencies, axis=1)

    lower_slope = (frequencies - lower) / tf.maximum(center - lower, 1e-12)
    upper_slope = (upper - frequencies) / tf.maximum(upper - center, 1e-12)
    return tf.maximum(0.0, tf.minimum(lower_slope, upper_slope))


def mel_weight_matrix(config: FrontendConfig | dict) -> tf.Tensor:
    config = FrontendConfig.from_dict(config)
    if config.stft is None:
        raise ValueError("Mel frontends require config.stft")
    if config.mel is None:
        raise ValueError("Mel frontends require config.mel")

    stft = config.stft
    mel = config.mel
    num_bins = stft.n_fft // 2 + 1
    f_max = _f_max(mel, stft.sample_rate)

    if mel.filterbank_impl == "tf":
        if mel.mel_scale != "htk" or mel.mel_norm != "none":
            raise ValueError("TensorFlow mel filters support HTK scale without Slaney normalization")
        return tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=mel.n_mels,
            num_spectrogram_bins=num_bins,
            sample_rate=stft.sample_rate,
            lower_edge_hertz=mel.f_min,
            upper_edge_hertz=f_max,
            dtype=tf.float32,
        )

    if mel.filterbank_impl == "librosa":
        try:
            import librosa
        except ImportError as exc:
            raise ImportError("librosa mel filters require the acoustic optional dependencies") from exc
        weights = librosa.filters.mel(
            sr=stft.sample_rate,
            n_fft=stft.n_fft,
            n_mels=mel.n_mels,
            fmin=mel.f_min,
            fmax=f_max,
            htk=mel.mel_scale == "htk",
            norm="slaney" if mel.mel_norm == "slaney" else None,
        )
        return tf.constant(weights.T, dtype=tf.float32)

    if mel.filterbank_impl == "kaldi_compatible":
        return _kaldi_mel_weight_matrix(mel, stft)

    if mel.filterbank_impl == "torchaudio":
        raise ValueError("torchaudio mel filters are not available in this TensorFlow frontend")

    raise ValueError(f"Unknown mel filterbank implementation: {mel.filterbank_impl!r}")


def mel_spectrogram(audio: tf.Tensor, config: FrontendConfig | dict) -> tf.Tensor:
    config = FrontendConfig.from_dict(config)
    if config.stft is None:
        raise ValueError("Mel frontends require config.stft")
    if config.mel is None:
        raise ValueError("Mel frontends require config.mel")
    if not config.stft.onesided:
        raise ValueError("Mel frontends require onesided STFT bins")

    spectrogram = stft_power_spectrogram(audio, config.stft)
    weights = mel_weight_matrix(config)
    return tf.einsum("tfc,fm->tmc", spectrogram, weights)


@register_audio_frontend("mel_power")
def mel_power(audio: tf.Tensor, config: FrontendConfig | dict) -> tf.Tensor:
    config = FrontendConfig.from_dict(config)
    features = mel_spectrogram(audio, config)
    return apply_feature_normalization(features, config.norm)


__all__ = [
    "mel_power",
    "mel_spectrogram",
    "mel_weight_matrix",
]

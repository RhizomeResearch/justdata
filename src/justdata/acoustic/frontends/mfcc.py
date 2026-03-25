from __future__ import annotations

import tensorflow as tf

from justdata.acoustic.configs import FrontendConfig, LogCompressionConfig
from justdata.acoustic.frontends.log import compress_log
from justdata.acoustic.frontends.mel import mel_spectrogram
from justdata.acoustic.normalization import apply_feature_normalization
from justdata.acoustic.registry import register_audio_frontend


@register_audio_frontend("mfcc")
def mfcc(audio: tf.Tensor, config: FrontendConfig | dict) -> tf.Tensor:
    config = FrontendConfig.from_dict(config)
    compression = config.log or LogCompressionConfig(kind="log")
    log_mel = compress_log(mel_spectrogram(audio, config), compression)
    log_mel = tf.transpose(log_mel, [0, 2, 1])
    coefficients = tf.signal.mfccs_from_log_mel_spectrograms(log_mel)
    coefficients = coefficients[..., : config.n_mfcc]
    coefficients = tf.transpose(coefficients, [0, 2, 1])
    return apply_feature_normalization(coefficients, config.norm)


__all__ = [
    "mfcc",
]

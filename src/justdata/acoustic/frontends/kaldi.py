from __future__ import annotations

from dataclasses import replace

import tensorflow as tf

from justdata.acoustic.configs import FrontendConfig, LogCompressionConfig
from justdata.acoustic.frontends.log import compress_log
from justdata.acoustic.frontends.mel import mel_spectrogram
from justdata.acoustic.normalization import apply_feature_normalization
from justdata.acoustic.registry import register_audio_frontend


@register_audio_frontend("kaldi_fbank")
def kaldi_fbank(audio: tf.Tensor, config: FrontendConfig | dict) -> tf.Tensor:
    config = FrontendConfig.from_dict(config)
    if config.mel is None:
        raise ValueError("kaldi_fbank frontend requires config.mel")

    kaldi_mel = replace(
        config.mel,
        filterbank_impl="kaldi_compatible",
        mel_scale="htk",
        mel_norm="none",
    )
    kaldi_config = replace(config, mel=kaldi_mel)
    compression = config.log or LogCompressionConfig(kind="log")
    features = compress_log(mel_spectrogram(audio, kaldi_config), compression)
    return apply_feature_normalization(features, config.norm)


__all__ = [
    "kaldi_fbank",
]

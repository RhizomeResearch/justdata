from __future__ import annotations

from dataclasses import replace

import tensorflow as tf

from justdata.acoustic.configs import FrontendConfig, LogCompressionConfig
from justdata.acoustic.frontends.log import pcen_compression
from justdata.acoustic.frontends.mel import mel_spectrogram
from justdata.acoustic.normalization import apply_feature_normalization
from justdata.acoustic.registry import register_audio_frontend


@register_audio_frontend("pcen_mel")
def pcen_mel(audio: tf.Tensor, config: FrontendConfig | dict) -> tf.Tensor:
    config = FrontendConfig.from_dict(config)
    compression = config.log or LogCompressionConfig(kind="pcen")
    if compression.kind != "pcen":
        compression = replace(compression, kind="pcen")
    features = pcen_compression(mel_spectrogram(audio, config), compression)
    return apply_feature_normalization(features, config.norm)


__all__ = [
    "pcen_mel",
]

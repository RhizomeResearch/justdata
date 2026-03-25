from __future__ import annotations

import os
from typing import Union

import tensorflow as tf

from justdata.acoustic.registry import register_audio_decoder
from justdata.acoustic.schema import SAMPLE_RATE, WAVEFORM


PathLike = Union[str, bytes, os.PathLike, tf.Tensor]


def decode_wav_bytes(
    contents: tf.Tensor,
    *,
    desired_channels: int | None = None,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Decode WAV bytes to float32 audio with shape [T, C]."""
    channels = -1 if desired_channels is None else desired_channels
    waveform, sample_rate = tf.audio.decode_wav(contents, desired_channels=channels)
    return tf.cast(waveform, tf.float32), tf.cast(sample_rate, tf.int32)


def decode_wav_file(
    path: PathLike,
    *,
    desired_channels: int | None = None,
) -> tuple[tf.Tensor, tf.Tensor]:
    if isinstance(path, os.PathLike):
        path = os.fspath(path)
    contents = tf.io.read_file(path)
    return decode_wav_bytes(contents, desired_channels=desired_channels)


def decode_audio_file(
    path: PathLike,
    *,
    format_hint: str | None = None,
    desired_channels: int | None = None,
) -> tuple[tf.Tensor, tf.Tensor]:
    if format_hint is not None and format_hint.lower() not in {"wav", ".wav"}:
        raise ValueError(
            "Only WAV decoding is available in the TensorFlow decoder path. "
            "Install an optional backend and decode before adaptation for other formats."
        )
    return decode_wav_file(path, desired_channels=desired_channels)


@register_audio_decoder("wav")
def decode_wav_sample(
    sample: dict,
    *,
    path_key: str = "path",
    desired_channels: int | None = None,
) -> dict:
    waveform, sample_rate = decode_wav_file(
        sample[path_key],
        desired_channels=desired_channels,
    )
    result = dict(sample)
    result[WAVEFORM] = waveform
    result[SAMPLE_RATE] = sample_rate
    return result


__all__ = [
    "decode_audio_file",
    "decode_wav_bytes",
    "decode_wav_file",
    "decode_wav_sample",
]

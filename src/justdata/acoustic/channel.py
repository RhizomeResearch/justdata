from __future__ import annotations

import tensorflow as tf

from justdata.acoustic.registry import register_audio_channel_strategy


@register_audio_channel_strategy("mono_mean")
def mono_mean(audio: tf.Tensor) -> tf.Tensor:
    audio = tf.convert_to_tensor(audio)
    return tf.reduce_mean(audio, axis=-1, keepdims=True)


@register_audio_channel_strategy("mono_left")
def mono_left(audio: tf.Tensor) -> tf.Tensor:
    audio = tf.convert_to_tensor(audio)
    return audio[:, :1]


@register_audio_channel_strategy("mono_right")
def mono_right(audio: tf.Tensor) -> tf.Tensor:
    audio = tf.convert_to_tensor(audio)
    return tf.cond(
        tf.shape(audio)[-1] < 2,
        lambda: audio[:, :1],
        lambda: audio[:, 1:2],
    )


@register_audio_channel_strategy("keep")
def keep(audio: tf.Tensor) -> tf.Tensor:
    return tf.convert_to_tensor(audio)


def apply_channel_strategy(audio: tf.Tensor, strategy: str) -> tf.Tensor:
    if strategy == "mono_mean":
        return mono_mean(audio)
    if strategy == "mono_left":
        return mono_left(audio)
    if strategy == "mono_right":
        return mono_right(audio)
    if strategy == "keep":
        return keep(audio)
    raise ValueError(f"Unknown channel strategy: {strategy!r}")


__all__ = [
    "apply_channel_strategy",
    "keep",
    "mono_left",
    "mono_mean",
    "mono_right",
]

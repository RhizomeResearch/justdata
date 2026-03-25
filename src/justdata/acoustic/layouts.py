from __future__ import annotations

from typing import Literal

import tensorflow as tf


AudioLayout = Literal["bt", "btc", "btf", "bft", "bcft", "btfc"]
OutputKind = Literal["waveform", "features"]


def _require_rank(x: tf.Tensor, rank: int, output_kind: str) -> None:
    if x.shape.rank != rank:
        raise ValueError(f"{output_kind} layout conversion expects rank-{rank} input")


def _squeeze_channel(x: tf.Tensor) -> tf.Tensor:
    return tf.squeeze(x, axis=-1)


def convert_audio_layout(
    x: tf.Tensor,
    layout: AudioLayout,
    *,
    output_kind: OutputKind,
) -> tf.Tensor:
    x = tf.convert_to_tensor(x)

    if output_kind == "waveform":
        _require_rank(x, 2, output_kind)
        if layout == "bt":
            return _squeeze_channel(x)
        if layout == "btc":
            return x
        raise ValueError(f"Layout {layout!r} is not valid for waveform output")

    if output_kind != "features":
        raise ValueError(f"Unknown acoustic output kind: {output_kind!r}")

    _require_rank(x, 3, output_kind)
    if layout == "btf":
        return _squeeze_channel(x)
    if layout == "bft":
        return tf.transpose(_squeeze_channel(x), [1, 0])
    if layout == "bcft":
        return tf.transpose(x, [2, 1, 0])
    if layout == "btfc":
        return x
    raise ValueError(f"Layout {layout!r} is not valid for feature output")


def unbatched_shape_for_layout(
    layout: AudioLayout,
    *,
    output_kind: OutputKind,
    time: int | None,
    frequency: int | None = None,
    channels: int | None = None,
) -> tuple[int | None, ...]:
    if output_kind == "waveform":
        if layout == "bt":
            return (time,)
        if layout == "btc":
            return (time, channels)
        raise ValueError(f"Layout {layout!r} is not valid for waveform output")

    if output_kind != "features":
        raise ValueError(f"Unknown acoustic output kind: {output_kind!r}")

    if layout == "btf":
        return (time, frequency)
    if layout == "bft":
        return (frequency, time)
    if layout == "bcft":
        return (channels, frequency, time)
    if layout == "btfc":
        return (time, frequency, channels)
    raise ValueError(f"Layout {layout!r} is not valid for feature output")


__all__ = [
    "AudioLayout",
    "OutputKind",
    "convert_audio_layout",
    "unbatched_shape_for_layout",
]

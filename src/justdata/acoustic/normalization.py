from __future__ import annotations

from collections.abc import Callable
from typing import Any

import tensorflow as tf

from justdata.acoustic.configs import FeatureNormConfig
from justdata.acoustic.registry import register_audio_normalization


_AXES_BY_RANK = {
    2: {"time": 0, "channel": 1},
    3: {"time": 0, "frequency": 1, "channel": 2},
}


def _rank(x: tf.Tensor) -> int:
    rank = x.shape.rank
    if rank not in _AXES_BY_RANK:
        raise ValueError("Feature normalization expects rank-2 or rank-3 tensors")
    return rank


def _axes_from_names(x: tf.Tensor, names: tuple[str, ...]) -> list[int]:
    rank = _rank(x)
    if not names:
        return list(range(rank))
    axes: list[int] = []
    for name in names:
        if name not in _AXES_BY_RANK[rank]:
            raise ValueError(
                f"Unknown normalization axis {name!r} for rank-{rank} tensor"
            )
        axes.append(_AXES_BY_RANK[rank][name])
    return axes


def _as_float_tensor(value: Any, field_name: str) -> tf.Tensor:
    if isinstance(value, str):
        raise ValueError(
            f"{field_name} path references are not loaded by the built-in normalizer"
        )
    return tf.cast(tf.convert_to_tensor(value), tf.float32)


def _broadcast_vector(
    x: tf.Tensor, values: tuple[float, ...], field_name: str
) -> tf.Tensor:
    rank = _rank(x)
    tensor = _as_float_tensor(values, field_name)
    length = len(values)

    if length == 1:
        return tf.reshape(tensor, [1] * rank)

    static_shape = x.shape.as_list()
    if rank == 2:
        channels = static_shape[1]
        if channels is not None and length != channels:
            raise ValueError(
                f"{field_name} length {length} does not match channel dimension {channels}"
            )
        return tf.reshape(tensor, [1, length])

    frequency = static_shape[1]
    channels = static_shape[2]
    if frequency is not None and length == frequency:
        return tf.reshape(tensor, [1, length, 1])
    if channels is not None and length == channels:
        return tf.reshape(tensor, [1, 1, length])

    expected = [dim for dim in (frequency, channels) if dim is not None]
    expected_text = " or ".join(str(dim) for dim in expected) or "frequency or channel"
    raise ValueError(f"{field_name} length {length} must match {expected_text}")


def _broadcast_stat(
    x: tf.Tensor,
    value: tuple[float, ...] | float | str | None,
    field_name: str,
    default: float,
) -> tf.Tensor:
    if value is None:
        return tf.cast(default, tf.float32)
    if isinstance(value, (int, float)):
        return tf.cast(value, tf.float32)
    if isinstance(value, tuple):
        return _broadcast_vector(x, value, field_name)
    return _as_float_tensor(value, field_name)


def _mean_std(x: tf.Tensor, axes: list[int], eps: float) -> tf.Tensor:
    mean, variance = tf.nn.moments(x, axes=axes, keepdims=True)
    return (x - mean) / tf.sqrt(variance + tf.cast(eps, x.dtype))


@register_audio_normalization("none")
def normalize_none(
    x: tf.Tensor, config: FeatureNormConfig | dict | None = None
) -> tf.Tensor:
    return tf.convert_to_tensor(x)


@register_audio_normalization("dataset_mean_std")
def normalize_dataset_mean_std(
    x: tf.Tensor, config: FeatureNormConfig | dict | None = None
) -> tf.Tensor:
    config = FeatureNormConfig.from_dict(config or {})
    x = tf.cast(tf.convert_to_tensor(x), tf.float32)
    mean = _broadcast_stat(x, config.mean, "mean", 0.0)
    std = _broadcast_stat(x, config.std, "std", 1.0)
    return (x - tf.cast(mean, x.dtype)) / (
        tf.cast(std, x.dtype) + tf.cast(config.eps, x.dtype)
    )


@register_audio_normalization("checkpoint_mean_std")
def normalize_checkpoint_mean_std(
    x: tf.Tensor, config: FeatureNormConfig | dict | None = None
) -> tf.Tensor:
    return normalize_dataset_mean_std(x, config)


@register_audio_normalization("per_clip_mean_std")
def normalize_per_clip_mean_std(
    x: tf.Tensor, config: FeatureNormConfig | dict | None = None
) -> tf.Tensor:
    config = FeatureNormConfig.from_dict(config or {})
    x = tf.cast(tf.convert_to_tensor(x), tf.float32)
    return _mean_std(x, _axes_from_names(x, config.axes), config.eps)


@register_audio_normalization("per_frequency_mean_std")
def normalize_per_frequency_mean_std(
    x: tf.Tensor, config: FeatureNormConfig | dict | None = None
) -> tf.Tensor:
    config = FeatureNormConfig.from_dict(config or {})
    x = tf.cast(tf.convert_to_tensor(x), tf.float32)
    return _mean_std(x, [0], config.eps)


@register_audio_normalization("kaldi_cmvn")
def normalize_kaldi_cmvn(
    x: tf.Tensor, config: FeatureNormConfig | dict | None = None
) -> tf.Tensor:
    config = FeatureNormConfig.from_dict(config or {})
    x = tf.cast(tf.convert_to_tensor(x), tf.float32)
    axes = _axes_from_names(x, config.axes or ("time",))
    return _mean_std(x, axes, config.eps)


@register_audio_normalization("affine")
def normalize_affine(
    x: tf.Tensor, config: FeatureNormConfig | dict | None = None
) -> tf.Tensor:
    config = FeatureNormConfig.from_dict(config or {})
    x = tf.cast(tf.convert_to_tensor(x), tf.float32)
    scale = _broadcast_stat(x, config.scale, "scale", 1.0)
    bias = _broadcast_stat(x, config.bias, "bias", 0.0)
    return x * tf.cast(scale, x.dtype) + tf.cast(bias, x.dtype)


_NORMALIZERS: dict[
    str, Callable[[tf.Tensor, FeatureNormConfig | dict | None], tf.Tensor]
] = {
    "none": normalize_none,
    "dataset_mean_std": normalize_dataset_mean_std,
    "checkpoint_mean_std": normalize_checkpoint_mean_std,
    "per_clip_mean_std": normalize_per_clip_mean_std,
    "per_frequency_mean_std": normalize_per_frequency_mean_std,
    "kaldi_cmvn": normalize_kaldi_cmvn,
    "affine": normalize_affine,
}


def apply_feature_normalization(
    x: tf.Tensor,
    config: FeatureNormConfig | dict | None,
) -> tf.Tensor:
    config = FeatureNormConfig.from_dict(config or {})
    if config.kind not in _NORMALIZERS:
        raise ValueError(f"Unknown feature normalization kind: {config.kind!r}")
    return _NORMALIZERS[config.kind](x, config)


__all__ = [
    "apply_feature_normalization",
    "normalize_affine",
    "normalize_checkpoint_mean_std",
    "normalize_dataset_mean_std",
    "normalize_kaldi_cmvn",
    "normalize_none",
    "normalize_per_clip_mean_std",
    "normalize_per_frequency_mean_std",
]

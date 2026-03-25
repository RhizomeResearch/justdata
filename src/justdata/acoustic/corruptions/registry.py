from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import tensorflow as tf

from justdata.acoustic.registry import (
    get_audio_corruption,
    has_audio_corruption,
    list_audio_corruptions,
    register_audio_corruption,
)


SeverityTable = Mapping[int, float | int | tuple[float, ...]]

_EPS = tf.constant(1e-8, dtype=tf.float32)
_MAX_SEED = tf.constant(2**31 - 1, dtype=tf.int64)


def apply_audio_corruption(
    audio_or_features: tf.Tensor,
    corruption: str,
    severity: int,
    seed: tf.Tensor,
    config: dict | None = None,
) -> tf.Tensor:
    """Apply a registered audio corruption with a validated severity level."""
    _validate_severity(severity)
    fn = get_audio_corruption(corruption)
    return fn(
        audio_or_features,
        severity=severity,
        seed=_seed_tensor(seed),
        config=config,
    )


def _validate_severity(severity: int | tf.Tensor) -> tf.Tensor:
    if not tf.is_tensor(severity):
        value = int(severity)
        if value < 1 or value > 5:
            raise ValueError("Audio corruption severity must be in the range 1..5.")

    severity_tensor = tf.cast(tf.convert_to_tensor(severity), tf.int32)
    if tf.executing_eagerly():
        try:
            value = int(severity_tensor.numpy())
        except (TypeError, ValueError):
            value = None
        if value is not None and (value < 1 or value > 5):
            raise ValueError("Audio corruption severity must be in the range 1..5.")

    with tf.control_dependencies(
        [
            tf.debugging.assert_greater_equal(
                severity_tensor,
                tf.constant(1, dtype=tf.int32),
                message="Audio corruption severity must be in the range 1..5.",
            ),
            tf.debugging.assert_less_equal(
                severity_tensor,
                tf.constant(5, dtype=tf.int32),
                message="Audio corruption severity must be in the range 1..5.",
            ),
        ]
    ):
        return tf.identity(severity_tensor)


def _severity_index(severity: int | tf.Tensor) -> tf.Tensor:
    return _validate_severity(severity) - 1


def _severity_value(
    table: SeverityTable,
    severity: int | tf.Tensor,
    *,
    dtype: tf.dtypes.DType = tf.float32,
) -> tf.Tensor:
    values = [table[level] for level in range(1, 6)]
    return tf.gather(tf.constant(values, dtype=dtype), _severity_index(severity))


def _seed_tensor(seed: tf.Tensor | int | None) -> tf.Tensor:
    if seed is None:
        return tf.constant([0, 0], dtype=tf.int32)
    seed = tf.cast(tf.convert_to_tensor(seed), tf.int32)
    if seed.shape.rank == 0:
        return tf.stack([seed, tf.constant(0, dtype=tf.int32)])
    if seed.shape.rank == 1 and seed.shape[0] == 1:
        return tf.stack([seed[0], tf.constant(0, dtype=tf.int32)])
    return seed[:2]


def _split_seed(seed: tf.Tensor | int | None, count: int) -> tf.Tensor:
    return tf.random.split(_seed_tensor(seed), count)


def _seed_from_index(
    seed: int | tf.Tensor, index: tf.Tensor, salt: int = 0
) -> tf.Tensor:
    base = tf.cast(_seed_tensor(seed), tf.int64)
    idx = tf.cast(index, tf.int64)
    salted = tf.math.floormod(base[1] + idx + tf.cast(salt, tf.int64), _MAX_SEED)
    return tf.cast(tf.stack([base[0], salted]), tf.int32)


def _as_tc(x: tf.Tensor) -> tuple[tf.Tensor, int | None]:
    x = tf.convert_to_tensor(x)
    rank = x.shape.rank
    x = tf.cast(x, tf.float32)
    if rank == 1:
        return x[:, tf.newaxis], rank
    if rank == 2:
        return x, rank
    raise ValueError("Audio corruption input must have shape [T] or [T, C].")


def _restore_rank(x: tf.Tensor, rank: int | None) -> tf.Tensor:
    if rank == 1:
        return tf.squeeze(x, axis=-1)
    return x


def _fit_length(x: tf.Tensor, target_length: tf.Tensor) -> tf.Tensor:
    target_length = tf.cast(tf.maximum(target_length, 1), tf.int32)
    x = x[:target_length]
    pad = tf.maximum(target_length - tf.shape(x)[0], 0)
    return tf.pad(x, [[0, pad], [0, 0]])


def _resize_time(x: tf.Tensor, target_length: tf.Tensor) -> tf.Tensor:
    target_length = tf.cast(tf.maximum(target_length, 1), tf.int32)
    resized = tf.image.resize(
        x[tf.newaxis, :, tf.newaxis, :],
        [target_length, 1],
        method="bilinear",
        antialias=True,
    )
    return resized[0, :, 0, :]


def _sample_rate(config: dict | None, default: int = 32000) -> tf.Tensor:
    value = default if config is None else config.get("sample_rate", default)
    return tf.cast(value, tf.float32)


def _rfft_filter(x: tf.Tensor, response: tf.Tensor) -> tf.Tensor:
    audio, rank = _as_tc(x)
    time = tf.shape(audio)[0]
    channels_first = tf.transpose(audio, [1, 0])
    spectrum = tf.signal.rfft(channels_first, fft_length=[time])
    filtered = spectrum * tf.cast(response[tf.newaxis, :], spectrum.dtype)
    result = tf.signal.irfft(filtered, fft_length=[time])
    return _restore_rank(tf.transpose(result, [1, 0]), rank)


def _frequency_bins(x: tf.Tensor, sample_rate: tf.Tensor) -> tf.Tensor:
    audio, _rank = _as_tc(x)
    time = tf.shape(audio)[0]
    num_bins = time // 2 + 1
    return tf.linspace(
        tf.constant(0.0, dtype=tf.float32),
        tf.cast(sample_rate, tf.float32) * 0.5,
        num_bins,
    )


def _convolve_waveform(
    x: tf.Tensor,
    ir: tf.Tensor,
    *,
    compensate_delay: bool = False,
) -> tf.Tensor:
    audio, rank = _as_tc(x)
    time = tf.shape(audio)[0]
    ir = tf.reshape(tf.cast(ir, tf.float32), [-1])
    ir = tf.cond(tf.shape(ir)[0] > 0, lambda: ir, lambda: tf.ones([1], tf.float32))
    kernel = tf.reverse(ir, axis=[0])[:, tf.newaxis, tf.newaxis]
    kernel_length = tf.shape(kernel)[0]
    start = (
        tf.argmax(tf.abs(ir), output_type=tf.int32)
        if compensate_delay
        else (kernel_length - 1) // 2
    )

    def convolve_channel(channel: tf.Tensor) -> tf.Tensor:
        signal = channel[tf.newaxis, :, tf.newaxis]
        padded = tf.pad(
            signal, [[0, 0], [kernel_length - 1, kernel_length - 1], [0, 0]]
        )
        full = tf.nn.conv1d(padded, kernel, stride=1, padding="VALID")[0, :, 0]
        return _fit_length(full[start:, tf.newaxis], time)[:, 0]

    channels_first = tf.transpose(audio, [1, 0])
    convolved = tf.map_fn(
        convolve_channel,
        channels_first,
        fn_output_signature=tf.float32,
    )
    return _restore_rank(tf.transpose(convolved, [1, 0]), rank)


def _metadata_with_corruption(
    sample: dict[str, Any],
    *,
    name: str,
    severity: int,
    domain: str,
) -> dict[str, Any]:
    result = dict(sample)
    current = result.get("metadata")
    metadata = dict(current) if isinstance(current, dict) else {}
    metadata["corruption"] = tf.constant(name)
    metadata["severity"] = tf.cast(severity, tf.int32)
    metadata["corruption_domain"] = tf.constant(domain)
    result["metadata"] = metadata
    return result


__all__ = [
    "apply_audio_corruption",
    "get_audio_corruption",
    "has_audio_corruption",
    "list_audio_corruptions",
    "register_audio_corruption",
]

from __future__ import annotations

import tensorflow as tf

from justdata.acoustic.configs import FrontendConfig, LogCompressionConfig
from justdata.acoustic.frontends.mel import mel_spectrogram
from justdata.acoustic.normalization import apply_feature_normalization
from justdata.acoustic.registry import register_audio_frontend


_LOG10 = tf.math.log(tf.constant(10.0, dtype=tf.float32))


def _log10(x: tf.Tensor) -> tf.Tensor:
    return tf.math.log(x) / tf.cast(_LOG10, x.dtype)


def _example_axes(x: tf.Tensor) -> list[int]:
    if x.shape.rank is not None and x.shape.rank >= 4:
        return list(range(1, x.shape.rank))
    return list(range(x.shape.rank or 0))


def _ref_value(x: tf.Tensor, ref: float | str) -> tf.Tensor:
    if ref == "max":
        return tf.maximum(
            tf.reduce_max(x, axis=_example_axes(x), keepdims=True),
            tf.cast(1e-30, x.dtype),
        )
    return tf.cast(ref, x.dtype)


def pcen_compression(x: tf.Tensor, config: LogCompressionConfig | dict) -> tf.Tensor:
    config = LogCompressionConfig.from_dict(config)
    x = tf.cast(tf.convert_to_tensor(x), tf.float32)
    x = tf.maximum(x, tf.cast(config.amin, x.dtype))
    smooth = tf.cast(config.smooth_coef, x.dtype)

    first = x[0]

    def _scan(prev: tf.Tensor, current: tf.Tensor) -> tf.Tensor:
        return (1.0 - smooth) * prev + smooth * current

    smoother = tf.concat(
        [tf.expand_dims(first, 0), tf.scan(_scan, x[1:], initializer=first)], axis=0
    )
    pcen = tf.pow(
        x / tf.pow(tf.cast(config.eps, x.dtype) + smoother, config.alpha)
        + config.delta,
        config.r,
    ) - tf.pow(tf.cast(config.delta, x.dtype), config.r)
    return tf.maximum(pcen, tf.zeros([], dtype=pcen.dtype))


def compress_log(
    x: tf.Tensor, config: LogCompressionConfig | dict | None = None
) -> tf.Tensor:
    config = LogCompressionConfig.from_dict(config or {})
    x = tf.cast(tf.convert_to_tensor(x), tf.float32)
    floor = tf.maximum(x, tf.cast(config.amin, x.dtype))

    if config.kind == "log":
        return tf.math.log(floor + tf.cast(config.log_offset, x.dtype))

    if config.kind == "log10":
        return _log10(floor + tf.cast(config.log_offset, x.dtype))

    if config.kind == "db":
        ref = _ref_value(floor, config.ref)
        db = 10.0 * _log10(floor / ref)
        if config.top_db is not None:
            peak = tf.reduce_max(db, axis=_example_axes(db), keepdims=True)
            db = tf.maximum(db, peak - tf.cast(config.top_db, db.dtype))
        return db

    if config.kind == "pcen":
        return pcen_compression(floor, config)

    raise ValueError(f"Unknown log compression kind: {config.kind!r}")


@register_audio_frontend("logmel")
def logmel(audio: tf.Tensor, config: FrontendConfig | dict) -> tf.Tensor:
    config = FrontendConfig.from_dict(config)
    compression = config.log or LogCompressionConfig(kind="log")
    features = compress_log(mel_spectrogram(audio, config), compression)
    return apply_feature_normalization(features, config.norm)


__all__ = [
    "compress_log",
    "logmel",
    "pcen_compression",
]

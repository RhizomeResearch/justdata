from __future__ import annotations

import numpy as np
import tensorflow as tf

from justdata.acoustic.registry import register_audio_resampler
from justdata.acoustic.schema import SAMPLE_RATE, WAVEFORM


def _resample_tensorflow(
    waveform: tf.Tensor,
    original_sample_rate: tf.Tensor | int,
    target_sample_rate: tf.Tensor | int,
) -> tf.Tensor:
    """Resample [T, C] waveform with TensorFlow linear interpolation."""
    original_sample_rate = tf.cast(original_sample_rate, tf.float32)
    target_sample_rate = tf.cast(target_sample_rate, tf.float32)

    num_samples = tf.shape(waveform)[0]
    num_channels = tf.shape(waveform)[1]
    target_samples = tf.cast(
        tf.round(tf.cast(num_samples, tf.float32) * target_sample_rate / original_sample_rate),
        tf.int32,
    )
    target_samples = tf.maximum(target_samples, 1)

    channels_first = tf.transpose(waveform, [1, 0])
    image = channels_first[:, :, tf.newaxis]
    resized = tf.image.resize(
        image,
        [num_channels, target_samples],
        method="bilinear",
        antialias=True,
    )
    result = tf.transpose(tf.squeeze(resized, axis=-1), [1, 0])
    result.set_shape([None, waveform.shape[-1]])
    return result


def _soxr_resample_numpy(
    waveform: np.ndarray,
    original_sample_rate: np.ndarray,
    target_sample_rate: np.ndarray,
) -> np.ndarray:
    try:
        import soxr
    except ImportError as exc:  # pragma: no cover - depends on optional extra.
        raise ImportError("Install the acoustic extra to use soxr resampling.") from exc

    result = soxr.resample(
        waveform.astype(np.float32, copy=False),
        int(original_sample_rate),
        int(target_sample_rate),
        axis=0,
    )
    return result.astype(np.float32, copy=False)


def _kaiser_best_resample_numpy(
    waveform: np.ndarray,
    original_sample_rate: np.ndarray,
    target_sample_rate: np.ndarray,
) -> np.ndarray:
    try:
        import librosa
    except ImportError:
        return _soxr_resample_numpy(waveform, original_sample_rate, target_sample_rate)

    result = librosa.resample(
        waveform.astype(np.float32, copy=False),
        orig_sr=int(original_sample_rate),
        target_sr=int(target_sample_rate),
        res_type="kaiser_best",
        axis=0,
    )
    return result.astype(np.float32, copy=False)


def _numpy_resample(
    waveform: tf.Tensor,
    original_sample_rate: tf.Tensor | int,
    target_sample_rate: tf.Tensor | int,
    *,
    method: str,
) -> tf.Tensor:
    """Resample through optional NumPy backends.

    This path is deterministic for fixed inputs but is not graph-pure because
    it uses ``tf.numpy_function``.
    """
    fn = _soxr_resample_numpy if method == "soxr" else _kaiser_best_resample_numpy
    result = tf.numpy_function(
        fn,
        [
            waveform,
            tf.cast(original_sample_rate, tf.int32),
            tf.cast(target_sample_rate, tf.int32),
        ],
        Tout=tf.float32,
    )
    result.set_shape([None, waveform.shape[-1]])
    return result


def _assert_same_sample_rate(
    waveform: tf.Tensor,
    original_sample_rate: tf.Tensor | int,
    target_sample_rate: tf.Tensor | int,
    *,
    method: str,
) -> tf.Tensor:
    with tf.control_dependencies(
        [
            tf.debugging.assert_equal(
                tf.cast(original_sample_rate, tf.int32),
                tf.cast(target_sample_rate, tf.int32),
                message=f"{method} resampling requires audio decoded at target_sample_rate",
            )
        ]
    ):
        return tf.identity(waveform)


def resample_waveform(
    waveform: tf.Tensor,
    original_sample_rate: tf.Tensor | int,
    target_sample_rate: tf.Tensor | int,
    method: str = "tensorflow",
) -> tf.Tensor:
    """Resample a ``[T, C]`` waveform to ``target_sample_rate``."""
    waveform = tf.convert_to_tensor(waveform, dtype=tf.float32)
    same_rate = tf.equal(
        tf.cast(original_sample_rate, tf.int32),
        tf.cast(target_sample_rate, tf.int32),
    )

    if method in {"identity", "hf_audio"}:
        return _assert_same_sample_rate(
            waveform,
            original_sample_rate,
            target_sample_rate,
            method=method,
        )

    if method == "tensorflow":
        return tf.cond(
            same_rate,
            lambda: waveform,
            lambda: _resample_tensorflow(waveform, original_sample_rate, target_sample_rate),
        )

    if method in {"soxr", "kaiser_best"}:
        return tf.cond(
            same_rate,
            lambda: waveform,
            lambda: _numpy_resample(
                waveform,
                original_sample_rate,
                target_sample_rate,
                method=method,
            ),
        )

    raise ValueError(f"Unknown resampling method: {method!r}")


@register_audio_resampler("tensorflow")
def resample_sample(
    sample: dict,
    *,
    target_sample_rate: int,
    method: str = "tensorflow",
) -> dict:
    result = dict(sample)
    result[WAVEFORM] = resample_waveform(
        result[WAVEFORM],
        result[SAMPLE_RATE],
        target_sample_rate,
        method=method,
    )
    result[SAMPLE_RATE] = tf.cast(target_sample_rate, tf.int32)
    return result


@register_audio_resampler("soxr")
def resample_sample_soxr(sample: dict, *, target_sample_rate: int) -> dict:
    return resample_sample(sample, target_sample_rate=target_sample_rate, method="soxr")


@register_audio_resampler("kaiser_best")
def resample_sample_kaiser_best(sample: dict, *, target_sample_rate: int) -> dict:
    return resample_sample(
        sample,
        target_sample_rate=target_sample_rate,
        method="kaiser_best",
    )


@register_audio_resampler("hf_audio")
def resample_sample_hf_audio(sample: dict, *, target_sample_rate: int) -> dict:
    return resample_sample(
        sample,
        target_sample_rate=target_sample_rate,
        method="hf_audio",
    )


__all__ = [
    "resample_sample",
    "resample_sample_hf_audio",
    "resample_sample_kaiser_best",
    "resample_sample_soxr",
    "resample_waveform",
]

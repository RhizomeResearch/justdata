from __future__ import annotations

import tensorflow as tf

from justdata.acoustic.registry import register_audio_resampler
from justdata.acoustic.schema import SAMPLE_RATE, WAVEFORM


def resample_waveform(
    waveform: tf.Tensor,
    original_sample_rate: tf.Tensor | int,
    target_sample_rate: tf.Tensor | int,
) -> tf.Tensor:
    """Resample [T, C] waveform with TensorFlow linear interpolation."""
    waveform = tf.convert_to_tensor(waveform)
    original_sample_rate = tf.cast(original_sample_rate, tf.float32)
    target_sample_rate = tf.cast(target_sample_rate, tf.float32)

    def _resample() -> tf.Tensor:
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
        return tf.transpose(tf.squeeze(resized, axis=-1), [1, 0])

    return tf.cond(
        tf.equal(tf.cast(original_sample_rate, tf.int64), tf.cast(target_sample_rate, tf.int64)),
        lambda: waveform,
        _resample,
    )


@register_audio_resampler("tensorflow")
def resample_sample(
    sample: dict,
    *,
    target_sample_rate: int,
) -> dict:
    result = dict(sample)
    result[WAVEFORM] = resample_waveform(
        result[WAVEFORM],
        result[SAMPLE_RATE],
        target_sample_rate,
    )
    result[SAMPLE_RATE] = tf.cast(target_sample_rate, tf.int32)
    return result


__all__ = ["resample_sample", "resample_waveform"]

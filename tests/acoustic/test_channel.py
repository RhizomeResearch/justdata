import numpy as np
import tensorflow as tf

from justdata.acoustic.channel import mono_left, mono_mean, mono_right
from justdata.acoustic.preprocessing import remove_dc_offset


def test_channel_mono_mean_matches_manual_mean():
    audio = tf.constant([[1.0, 3.0], [2.0, 6.0]], dtype=tf.float32)

    result = mono_mean(audio)

    np.testing.assert_allclose(result.numpy(), [[2.0], [4.0]])


def test_channel_mono_left():
    audio = tf.constant([[1.0, 3.0], [2.0, 6.0]], dtype=tf.float32)

    result = mono_left(audio)

    np.testing.assert_allclose(result.numpy(), [[1.0], [2.0]])


def test_channel_mono_right_falls_back_for_mono():
    audio = tf.constant([[1.0], [2.0]], dtype=tf.float32)

    result = mono_right(audio)

    np.testing.assert_allclose(result.numpy(), [[1.0], [2.0]])


def test_remove_dc_offset_zeroes_mean_per_channel():
    audio = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=tf.float32)

    result = remove_dc_offset(audio)

    np.testing.assert_allclose(
        tf.reduce_mean(result, axis=0).numpy(),
        [0.0, 0.0],
        atol=1e-6,
    )

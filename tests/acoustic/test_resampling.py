import numpy as np
import tensorflow as tf

from justdata.acoustic.adapters import standardize_waveform_layout
from justdata.acoustic.resampling import resample_waveform


def test_standardize_layout_rank1_to_t_c():
    audio = tf.constant([1.0, 2.0, 3.0])

    result = standardize_waveform_layout(audio)

    assert result.shape == (3, 1)
    np.testing.assert_allclose(result.numpy()[:, 0], [1.0, 2.0, 3.0])


def test_standardize_layout_c_t_to_t_c():
    audio = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    result = standardize_waveform_layout(audio, layout_hint="ct")

    assert result.shape == (3, 2)
    np.testing.assert_allclose(result.numpy(), [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])


def test_resample_waveform_changes_time_dimension():
    audio = tf.zeros([160, 1], dtype=tf.float32)

    result = resample_waveform(audio, 16000, 8000)

    assert result.shape == (80, 1)

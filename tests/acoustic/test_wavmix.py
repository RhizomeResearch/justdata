import numpy as np
import pytest
import tensorflow as tf

from justdata.acoustic.augment.batch import audio_wavmix


def test_wavmix_waveform_only():
    spectrogram = tf.ones([2, 4, 3], dtype=tf.float32)

    with pytest.raises(ValueError, match="waveform"):
        audio_wavmix(
            spectrogram,
            seed=[21, 0],
            alpha=1.0,
            prob=1.0,
            input_kind="spectrogram",
        )


def test_wavmix_preserves_shape():
    waveform = tf.reshape(tf.range(2 * 8 * 1, dtype=tf.float32), [2, 8, 1])

    mixed = audio_wavmix(
        waveform,
        seed=[22, 0],
        alpha=1.0,
        prob=1.0,
        input_kind="waveform",
    )

    assert mixed.shape == waveform.shape
    assert np.isfinite(mixed.numpy()).all()

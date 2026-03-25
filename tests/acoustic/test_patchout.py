import numpy as np
import tensorflow as tf

from justdata.acoustic.augment import (
    passt_patchout,
    structured_patchout_frequency,
    structured_patchout_time,
    unstructured_patchout,
)
from justdata.acoustic.schema import FEATURES, LABEL, METADATA


def _grid(time: int = 6, freq: int = 8, channels: int = 2) -> tf.Tensor:
    return tf.reshape(
        tf.range(time * freq * channels, dtype=tf.float32), [time, freq, channels]
    )


def test_patchout_structured_frequency_count():
    x = _grid()

    y = structured_patchout_frequency(x, n_freq_patches=3, seed=[1, 0])

    assert y.shape == (6, 5, 2)


def test_patchout_structured_time_count():
    x = _grid()

    y = structured_patchout_time(x, n_time_patches=2, seed=[2, 0])

    assert y.shape == (4, 8, 2)


def test_patchout_unstructured_count():
    x = _grid(time=4, freq=5, channels=3)

    y = unstructured_patchout(x, n_patches=6, seed=[3, 0])

    assert y.shape == (14, 3)


def test_patchout_disabled_eval():
    x = _grid()

    y = passt_patchout(
        x,
        seed=[4, 0],
        structured_frequency=3,
        structured_time=2,
        unstructured=4,
        is_training=False,
    )

    np.testing.assert_array_equal(y.numpy(), x.numpy())


def test_patchout_metadata_debug():
    sample = {
        FEATURES: _grid(),
        LABEL: tf.constant(2, dtype=tf.int64),
        METADATA: {
            "clip_id": tf.constant("clip-a"),
            "patch_grid": {"input_fdim": 8, "input_tdim": 6},
        },
    }

    result = passt_patchout(
        sample,
        seed=[5, 0],
        structured_frequency=2,
        structured_time=1,
        debug=True,
    )

    assert result[LABEL].numpy() == sample[LABEL].numpy()
    assert result[METADATA]["patch_grid"] == sample[METADATA]["patch_grid"]
    assert result[FEATURES].shape == (5, 6, 2)
    assert result[METADATA]["patchout"]["structured_frequency"].shape == (2,)
    assert result[METADATA]["patchout"]["structured_time"].shape == (1,)
    assert result[METADATA]["patchout"]["unstructured"].shape == (0,)


def test_patchout_metadata_not_recorded_without_debug():
    sample = {
        FEATURES: _grid(),
        LABEL: tf.constant(2, dtype=tf.int64),
        METADATA: {"patch_grid": {"input_fdim": 8, "input_tdim": 6}},
    }

    result = passt_patchout(
        sample,
        seed=[6, 0],
        structured_frequency=1,
        debug=False,
    )

    assert "patchout" not in result[METADATA]

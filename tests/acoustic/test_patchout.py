import numpy as np
import pytest
import tensorflow as tf

from justdata.acoustic.augment import (
    passt_patchout,
    structured_patchout_frequency,
    structured_patchout_time,
    unstructured_patchout,
)
from justdata.acoustic.schema import FEATURES, LABEL, METADATA
from justdata.acoustic.augment.patchout import _drop_axis
from justdata.acoustic.augment.spectrogram import _stateless_shuffled_range


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


def test_patchout_eval_is_opt_in():
    x = _grid()

    y = passt_patchout(
        x,
        seed=[4, 0],
        structured_frequency=3,
        structured_time=2,
        is_training=False,
        augment_eval=True,
    )

    assert y.shape == (4, 5, 2)


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


@pytest.mark.parametrize(
    "size,count", [(0, 0), (8, -2), (8, 0), (8, 20), (128, 4), (512, 257), (512, 1000)]
)
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("dynamic", [False, True])
def test_drop_axis_matches_pairwise_selection(size, count, axis, dynamic):
    shape = [size, 3] if axis == 0 else [3, size]
    x = tf.reshape(tf.range(size * 3), shape)

    @tf.function(
        input_signature=[
            tf.TensorSpec([None, None], tf.int32),
            tf.TensorSpec([], tf.int32),
        ]
    )
    def dynamic_drop(value, amount):
        return _drop_axis(value, axis=axis, count=amount, seed=tf.constant([31, 9]))

    dropped = tf.sort(
        _stateless_shuffled_range(size, tf.constant([31, 9]))[
            : min(max(count, 0), size)
        ]
    )
    keep_mask = ~tf.reduce_any(tf.range(size)[:, None] == dropped[None, :], axis=1)
    expected = tf.gather(x, tf.reshape(tf.where(keep_mask), [-1]), axis=axis)
    if dynamic:
        actual, actual_dropped = dynamic_drop(x, tf.constant(count))
    else:
        actual, actual_dropped = _drop_axis(
            x, axis=axis, count=count, seed=tf.constant([31, 9])
        )
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual_dropped, dropped)

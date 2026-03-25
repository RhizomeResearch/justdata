import numpy as np
import pytest
import tensorflow as tf

from justdata.vision.corruptions.registry import (
    _CORRUPTION_REGISTRY,
    apply_minic_corruption,
    list_corruptions,
)


def test_corruptions_registered():
    assert len(_CORRUPTION_REGISTRY) > 0
    # known corruptions
    expected = ["noise", "blur", "weather", "digital"]
    for e in expected:
        assert e in _CORRUPTION_REGISTRY
    assert set(expected).issubset(set(list_corruptions()))


@pytest.mark.parametrize("corruption", ["noise", "blur", "weather", "digital"])
def test_apply_minic_corruption(corruption):
    image = tf.random.uniform((64, 64, 3), minval=0, maxval=255, dtype=tf.float32)
    seed = tf.constant([1, 2], dtype=tf.int32)

    # Apply corruption with severity 1
    corrupted_1 = apply_minic_corruption(image, corruption, severity=1, seed=seed)
    assert corrupted_1.shape == (64, 64, 3)
    assert corrupted_1.dtype == tf.uint8

    # Apply corruption with severity 5
    corrupted_5 = apply_minic_corruption(image, corruption, severity=5, seed=seed)
    assert corrupted_5.shape == (64, 64, 3)
    assert corrupted_5.dtype == tf.uint8


def test_unknown_corruption():
    image = tf.random.uniform((64, 64, 3), minval=0, maxval=255, dtype=tf.float32)
    seed = tf.constant([1, 2], dtype=tf.int32)
    with pytest.raises(ValueError, match="Unknown corruption"):
        apply_minic_corruption(image, "unknown_xyz", severity=1, seed=seed)


def test_minic_rejects_invalid_severity():
    image = tf.random.uniform((64, 64, 3), minval=0, maxval=255, dtype=tf.float32)
    seed = tf.constant([1, 2], dtype=tf.int32)

    with pytest.raises(ValueError, match="severity"):
        apply_minic_corruption(image, "noise", severity=0, seed=seed)


def test_minic_corruption_same_seed_same_output():
    image = tf.random.uniform((64, 64, 3), minval=0, maxval=255, dtype=tf.float32)
    seed = tf.constant([1, 2], dtype=tf.int32)

    first = apply_minic_corruption(image, "noise", severity=3, seed=seed)
    second = apply_minic_corruption(image, "noise", severity=3, seed=seed)

    np.testing.assert_array_equal(first.numpy(), second.numpy())


def test_minic_corruption_different_seed_can_change_output():
    image = tf.random.uniform((64, 64, 3), minval=0, maxval=255, dtype=tf.float32)

    first = apply_minic_corruption(
        image,
        "noise",
        severity=3,
        seed=tf.constant([1, 2], dtype=tf.int32),
    )
    second = apply_minic_corruption(
        image,
        "noise",
        severity=3,
        seed=tf.constant([1, 3], dtype=tf.int32),
    )

    assert not np.array_equal(first.numpy(), second.numpy())

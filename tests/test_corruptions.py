import pytest
import tensorflow as tf

from justdata.corruptions.registry import _CORRUPTION_REGISTRY, apply_minic_corruption

def test_corruptions_registered():
    assert len(_CORRUPTION_REGISTRY) > 0
    # known corruptions
    expected = ["noise", "blur", "weather", "digital"]
    for e in expected:
        assert e in _CORRUPTION_REGISTRY

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

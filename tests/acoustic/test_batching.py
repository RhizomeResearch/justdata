import numpy as np
import pytest
import tensorflow as tf

from justdata.acoustic.batching import pad_nested


def test_recursive_padding_nested_metadata():
    batch = {
        "metadata": {
            "quality": tf.constant([1.0, 2.0], dtype=tf.float32),
            "nested": {"device_id": tf.constant([3, 4], dtype=tf.int32)},
        }
    }

    padded = pad_nested(batch, tf.constant(1, dtype=tf.int32))

    np.testing.assert_array_equal(
        padded["metadata"]["quality"].numpy(), [1.0, 2.0, 0.0]
    )
    np.testing.assert_array_equal(
        padded["metadata"]["nested"]["device_id"].numpy(),
        [3, 4, 0],
    )


@pytest.mark.parametrize("mode", ["full", "numeric_only", "none", "custom"])
def test_padding_preserves_empty_containers_and_filters_only_strings(mode):
    original = {
        "empty": {},
        "text": {"id": tf.constant(["a", "b"])},
        "numeric": tf.constant([[1], [2]], dtype=tf.int64),
        "flags": tf.constant([True, False]),
    }
    padded = pad_nested(original, tf.constant(1), metadata_mode=mode)

    assert padded["empty"] == {}
    np.testing.assert_array_equal(padded["numeric"], [[1], [2], [0]])
    np.testing.assert_array_equal(padded["flags"], [True, False, False])
    if mode == "full":
        np.testing.assert_array_equal(padded["text"]["id"], [b"a", b"b", b""])
    else:
        assert padded["text"] == {}
    np.testing.assert_array_equal(original["text"]["id"], [b"a", b"b"])

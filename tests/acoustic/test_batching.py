import numpy as np
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

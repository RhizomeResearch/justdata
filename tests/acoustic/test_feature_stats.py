import numpy as np
import tensorflow as tf

from justdata.acoustic.stats import compute_feature_stats, make_stats_iterator


def _stats_dataset():
    return tf.data.Dataset.from_tensor_slices(
        {
            "inputs": [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ],
            "metadata": {
                "device": ["A", "B"],
                "quality": [1.0, 2.0],
            },
        }
    )


def test_stats_iterator_is_deterministic():
    first = next(make_stats_iterator(_stats_dataset(), "dev_train_25", None, 2))
    second = next(make_stats_iterator(_stats_dataset(), "dev_train_25", None, 2))

    np.testing.assert_allclose(first["inputs"], second["inputs"])


def test_stats_iterator_augment_false():
    batch = next(make_stats_iterator(_stats_dataset(), "dev_train_25", None, 2))

    np.testing.assert_allclose(batch["inputs"][0], [[1.0, 2.0], [3.0, 4.0]])
    assert "device" not in batch["metadata"]
    assert "quality" in batch["metadata"]


def test_compute_feature_stats_grouped_by_device():
    stats = compute_feature_stats(_stats_dataset(), groupby="device", feature_key="inputs")

    np.testing.assert_allclose(stats["A"]["mean"].numpy(), [2.0, 3.0])
    np.testing.assert_allclose(stats["A"]["std"].numpy(), [1.0, 1.0])
    assert stats["A"]["count"] == 2
    np.testing.assert_allclose(stats["B"]["mean"].numpy(), [6.0, 7.0])

import numpy as np
import pytest
import tensorflow as tf

from justdata.acoustic.stats import compute_feature_stats, make_stats_iterator
from justdata.core import stats as core_stats


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
    stats = compute_feature_stats(
        _stats_dataset(), groupby="device", feature_key="inputs"
    )

    np.testing.assert_allclose(stats["A"]["mean"].numpy(), [2.0, 3.0])
    np.testing.assert_allclose(stats["A"]["std"].numpy(), [1.0, 1.0])
    assert stats["A"]["count"] == 2
    np.testing.assert_allclose(stats["B"]["mean"].numpy(), [6.0, 7.0])


@pytest.mark.parametrize(
    ("features", "axes"),
    [
        (np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5), (0, 2, 3)),
        (np.arange(6 * 4).reshape(6, 4), ("time",)),
        (np.arange(2 * 3 * 4).reshape(2, 3, 4), ("time", "channel")),
        (np.arange(2 * 3 * 4).reshape(2, 3, 4), (-3, -1)),
        (np.arange(2 * 3).reshape(2, 3), ()),
    ],
)
def test_compute_feature_stats_matches_population_moments(features, axes):
    features = features.astype(np.float64)
    named_axes = {"time": 0, "frequency": 1, "freq": 1, "channel": -1}
    reduce_axes = tuple(
        axis if isinstance(axis, int) else named_axes[axis] for axis in axes
    )
    expected_mean = np.mean(features, axis=reduce_axes, dtype=np.float64)
    expected_std = np.std(features, axis=reduce_axes, dtype=np.float64)

    stats = compute_feature_stats(
        [{"inputs": features}], axes=axes, groupby=None, feature_key="inputs"
    )["all"]

    np.testing.assert_allclose(stats["mean"].numpy(), expected_mean, rtol=1e-6)
    np.testing.assert_allclose(stats["std"].numpy(), expected_std, rtol=1e-6)
    assert stats["count"] == np.prod([features.shape[axis] for axis in reduce_axes])
    assert stats["axes"] == axes
    assert stats["feature_key"] == "inputs"
    assert stats["mean"].dtype == tf.float32
    assert stats["std"].dtype == tf.float32


@pytest.mark.parametrize(
    "features",
    [
        np.full((8, 2), 7.0),
        np.array([[3.0, 4.0]]),
        1e6 + np.array([[-0.25, 0.5], [0.0, -0.5], [0.25, 0.0]]),
    ],
)
def test_compute_feature_stats_is_chunk_invariant(features):
    one_chunk = [{"inputs": features}]
    many_chunks = [{"inputs": row[None, :]} for row in features]

    combined = compute_feature_stats(one_chunk, groupby=None)["all"]
    chunked = compute_feature_stats(many_chunks, groupby=None)["all"]

    np.testing.assert_allclose(chunked["mean"].numpy(), combined["mean"].numpy())
    np.testing.assert_allclose(chunked["std"].numpy(), combined["std"].numpy())
    np.testing.assert_allclose(
        combined["mean"].numpy(), np.mean(features, axis=0, dtype=np.float64)
    )
    np.testing.assert_allclose(
        combined["std"].numpy(), np.std(features, axis=0, dtype=np.float64)
    )
    assert chunked["count"] == combined["count"] == len(features)


def test_compute_feature_stats_handles_empty_chunks():
    stats = compute_feature_stats(
        [{"inputs": np.empty((0, 2))}, {"inputs": np.array([[1.0, 3.0]])}],
        groupby=None,
    )["all"]

    np.testing.assert_allclose(stats["mean"].numpy(), [1.0, 3.0])
    np.testing.assert_allclose(stats["std"].numpy(), [0.0, 0.0])
    assert stats["count"] == 1


def test_stats_iterator_preserves_string_grouping_key():
    iterator = make_stats_iterator(
        _stats_dataset(), "dev_train_25", "device", 2
    )

    stats = compute_feature_stats(iterator, groupby="device", feature_key="inputs")

    assert set(stats) == {"A", "B"}


def test_stats_iterator_preserves_multiple_string_grouping_keys():
    dataset = _stats_dataset().map(
        lambda sample: {
            **sample,
            "metadata": {**sample["metadata"], "site": "city"},
        }
    )
    iterator = make_stats_iterator(
        dataset, "dev_train_25", ["device", "site"], 2
    )

    stats = compute_feature_stats(
        iterator, groupby=["device", "site"], feature_key="inputs"
    )

    assert set(stats) == {("A", "city"), ("B", "city")}


def test_compute_feature_stats_merges_once_per_feature_chunk(monkeypatch):
    merge_calls = 0
    original_merge = core_stats._merge_partial_state

    def counting_merge(*args, **kwargs):
        nonlocal merge_calls
        merge_calls += 1
        return original_merge(*args, **kwargs)

    monkeypatch.setattr(core_stats, "_merge_partial_state", counting_merge)
    batch = np.arange(2 * 3 * 32 * 32).reshape(2, 3, 32, 32)

    compute_feature_stats(
        [{"image": batch}], axes=(0, 2, 3), groupby=None, feature_key="image"
    )

    assert merge_calls == 1

from unittest.mock import Mock

import numpy as np
import pytest
import tensorflow as tf

from justdata.core.finalization import finalize_dataset
from justdata.core.metadata import MetadataSidecar


def _dataset() -> tf.data.Dataset:
    return tf.data.Dataset.from_tensor_slices(
        {
            "x": np.array([1, 2, 3], dtype=np.int32),
            "metadata": {
                "example_id": np.array([11, 12, 13], dtype=np.int64),
                "source": np.array(["a", "b", "c"]),
                "score": np.array([0.1, 0.2, 0.3], dtype=np.float32),
            },
        }
    ).apply(tf.data.experimental.assert_cardinality(3))


def _identity(sample, num_classes=None):
    del num_classes
    return sample


@pytest.mark.parametrize("metadata_mode", ["full", "numeric_only", "none"])
def test_finalize_dataset_applies_metadata_modes(metadata_mode):
    ds, n_batches = finalize_dataset(
        _dataset(),
        postprocess_fn=_identity,
        num_classes=None,
        batch_size=2,
        metadata_mode=metadata_mode,
        prefetch=False,
    )

    batch = next(iter(ds))
    assert n_batches == 2
    if metadata_mode == "none":
        assert "metadata" not in batch
    elif metadata_mode == "numeric_only":
        assert "source" not in batch["metadata"]
        assert "score" in batch["metadata"]
    else:
        assert "source" in batch["metadata"]


def test_finalize_dataset_preserves_stage_order_and_writes_sidecar(tmp_path):
    def postprocess(sample, num_classes=None):
        del num_classes
        return sample | {"x": sample["x"] * 2}

    def after_postprocess(dataset):
        return dataset.map(lambda sample: sample | {"x": sample["x"] + 1})

    def late_augment(batch, num_classes=None, seed=None):
        del num_classes, seed
        size = tf.shape(batch["x"])[0]
        return batch | {"observed_size": tf.fill([size], size)}

    sidecar_path = tmp_path / "metadata.jsonl"
    ds, _n = finalize_dataset(
        _dataset(),
        postprocess_fn=postprocess,
        post_postprocess_transform=after_postprocess,
        num_classes=None,
        batch_size=2,
        metadata_mode="numeric_only",
        sidecar_metadata_path=str(sidecar_path),
        is_training=True,
        late_augment_fn=late_augment,
        rng=tf.random.Generator.from_seed(7),
        deterministic=True,
        prefetch=False,
    )

    first, final = list(ds)
    np.testing.assert_array_equal(first["x"].numpy(), [3, 5])
    np.testing.assert_array_equal(final["x"].numpy(), [7, 0])
    np.testing.assert_array_equal(final["observed_size"].numpy(), [1, 0])
    np.testing.assert_array_equal(final["padding_mask"].numpy(), [True, False])
    sidecar = MetadataSidecar.read_jsonl(str(sidecar_path))
    assert sidecar.records[11]["source"] == "a"


@pytest.mark.parametrize("map_parallel_calls", [None, 1, 2])
@pytest.mark.parametrize("cache_model_inputs", [False, True])
def test_sidecar_respects_map_parallelism(
    tmp_path, map_parallel_calls, cache_model_inputs
):
    sidecar_path = tmp_path / "metadata.jsonl"
    ds, _n = finalize_dataset(
        _dataset(),
        postprocess_fn=_identity,
        num_classes=None,
        batch_size=2,
        metadata_mode="numeric_only",
        sidecar_metadata_path=str(sidecar_path),
        cache_model_inputs=cache_model_inputs,
        map_parallel_calls=map_parallel_calls,
        prefetch=False,
    )

    graph = tf.compat.v1.GraphDef()
    graph.ParseFromString(ds._as_serialized_graph().numpy())
    nodes = {node.name: node for node in graph.node}
    parallel_calls = {
        int(tf.make_ndarray(nodes[node.input[-1]].attr["value"].tensor))
        for node in graph.node
        if node.op == "ParallelMapDatasetV2"
    }
    expected = tf.data.AUTOTUNE if map_parallel_calls is None else map_parallel_calls
    assert parallel_calls == {expected}

    list(ds)
    assert MetadataSidecar.read_jsonl(str(sidecar_path)).records == {
        11: {"source": "a"},
        12: {"source": "b"},
        13: {"source": "c"},
    }


def test_finalize_dataset_drop_remainder_and_numpy_conversion():
    iterator, n_batches = finalize_dataset(
        _dataset(),
        postprocess_fn=_identity,
        num_classes=None,
        batch_size=2,
        drop_remainder=True,
        as_numpy=True,
    )

    batches = list(iterator)
    assert n_batches == 1
    assert len(batches) == 1
    assert isinstance(batches[0]["x"], np.ndarray)
    np.testing.assert_array_equal(batches[0]["padding_mask"], [True, True])


def test_finalize_dataset_reports_unknown_cardinality_as_none():
    filtered = _dataset().filter(lambda sample: sample["x"] > 1)

    ds, n_batches = finalize_dataset(
        filtered,
        postprocess_fn=_identity,
        num_classes=None,
        batch_size=2,
        prefetch=False,
    )

    assert n_batches is None
    assert len(list(ds)) == 1


def test_finalize_dataset_model_input_cache_reuses_postprocessing(tmp_path):
    calls = Mock()

    def postprocess(sample, num_classes=None):
        del num_classes

        def record(value):
            calls()
            return value

        value = tf.py_function(record, [sample["x"]], Tout=tf.int32)
        value.set_shape(sample["x"].shape)
        return sample | {"x": value}

    ds, _n = finalize_dataset(
        _dataset(),
        postprocess_fn=postprocess,
        num_classes=None,
        batch_size=2,
        cache_model_inputs=True,
        model_input_cache_path=str(tmp_path / "model-inputs"),
    )

    list(ds)
    list(ds)
    assert calls.call_count == 3


def test_sidecar_joins_survive_shuffle_model_cache_and_reiteration(tmp_path):
    sidecar_path = tmp_path / "metadata.jsonl"
    ds, _n = finalize_dataset(
        _dataset(),
        postprocess_fn=_identity,
        num_classes=None,
        batch_size=2,
        metadata_mode="numeric_only",
        sidecar_metadata_path=str(sidecar_path),
        cache_model_inputs=True,
        model_input_cache_path=str(tmp_path / "model-inputs"),
        is_training=True,
        shuffle_buffer=3,
        shuffle_seed=7,
        deterministic=True,
        prefetch=False,
    )

    first_ids = [
        int(example_id)
        for batch in ds
        for example_id, keep in zip(
            batch["metadata"]["example_id"].numpy(),
            batch["padding_mask"].numpy(),
        )
        if keep
    ]
    first_contents = sidecar_path.read_bytes()
    second_ids = [
        int(example_id)
        for batch in ds
        for example_id, keep in zip(
            batch["metadata"]["example_id"].numpy(),
            batch["padding_mask"].numpy(),
        )
        if keep
    ]
    sidecar = MetadataSidecar.read_jsonl(str(sidecar_path))

    assert sorted(first_ids) == sorted(second_ids) == [11, 12, 13]
    assert sidecar_path.read_bytes() == first_contents
    assert sidecar.records == {
        11: {"source": "a"},
        12: {"source": "b"},
        13: {"source": "c"},
    }

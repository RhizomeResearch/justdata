from unittest.mock import Mock

import numpy as np
import pytest
import tensorflow as tf

from justdata.acoustic.metadata import MetadataSidecar
from justdata.core.finalization import finalize_dataset


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
    assert int(n_batches.numpy()) == 2
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
    assert int(n_batches.numpy()) == 1
    assert len(batches) == 1
    assert isinstance(batches[0]["x"], np.ndarray)
    np.testing.assert_array_equal(batches[0]["padding_mask"], [True, True])


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

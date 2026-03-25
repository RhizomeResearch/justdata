from unittest.mock import patch

import numpy as np
import tensorflow as tf

from justdata.acoustic.metadata import MetadataSidecar
from justdata.core.loader import load_ds


def _identity(sample, *args, **kwargs):
    return sample


def _nested_metadata_ds():
    raw = tf.data.Dataset.from_tensor_slices(
        {
            "x": np.array([1, 2, 3], dtype=np.int32),
            "metadata": {
                "example_id": np.array([101, 102, 103], dtype=np.int64),
                "clip_id": np.array([b"clip-a", b"clip-b", b"clip-c"]),
                "nested": {
                    "score": np.array([0.1, 0.2, 0.3], dtype=np.float32),
                    "city": np.array([b"Paris", b"Berlin", b"Rome"]),
                },
            },
        }
    )
    return raw.apply(tf.data.experimental.assert_cardinality(3))


def _load(metadata_mode="full", sidecar_metadata_path=None):
    with patch("justdata.core.loader.fetch_ds", return_value=_nested_metadata_ds()):
        return load_ds(
            dataset_names_arg="mock",
            splits_arg="validation",
            dataset_type="validation",
            batch_size=2,
            seed=0,
            preprocess_fn=_identity,
            augment_fn=_identity,
            late_augment_fn=_identity,
            postprocess_fn=_identity,
            cache_dataset=False,
            metadata_mode=metadata_mode,
            sidecar_metadata_path=sidecar_metadata_path,
        )


def test_numeric_only_drops_strings_from_batch():
    ds, _n = _load(metadata_mode="numeric_only")
    batch = next(iter(ds))

    assert "clip_id" not in batch["metadata"]
    assert "city" not in batch["metadata"]["nested"]
    assert "score" in batch["metadata"]["nested"]


def test_loader_sidecar_writes_strings_by_example_id(tmp_path):
    path = tmp_path / "metadata.jsonl"
    ds, _n = _load(metadata_mode="numeric_only", sidecar_metadata_path=str(path))
    next(iter(ds))

    sidecar = MetadataSidecar.read_jsonl(str(path))

    assert sidecar.records[101]["clip_id"] == "clip-a"
    assert sidecar.records[101]["nested"]["city"] == "Paris"


def test_recursive_padding_nested_metadata():
    ds, _n = _load()
    batches = list(ds.take(2))
    final = batches[-1]

    np.testing.assert_allclose(final["metadata"]["nested"]["score"].numpy(), [0.3, 0.0])
    np.testing.assert_array_equal(final["metadata"]["clip_id"].numpy(), [b"clip-c", b""])


def test_padding_mask_added_for_short_final_batch():
    ds, _n = _load()
    final = list(ds.take(2))[-1]

    np.testing.assert_array_equal(final["padding_mask"].numpy(), [True, False])


def test_metadata_none_removes_metadata():
    ds, _n = _load(metadata_mode="none")
    batch = next(iter(ds))

    assert "metadata" not in batch

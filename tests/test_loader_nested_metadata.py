import json
from unittest.mock import patch

import numpy as np
import pytest
import tensorflow as tf

from justdata.core.loader import load_ds
from justdata.core.metadata import (
    MetadataSidecar,
    attach_sidecar_writer,
    stable_int64_hash,
)


def _identity(sample, *args, **kwargs):
    return sample


def _nested_metadata_ds():
    raw = tf.data.Dataset.from_tensor_slices(
        {
            "x": np.array([1, 2, 3], dtype=np.int32),
            "label": np.array([b"class-a", b"class-b", b"class-c"]),
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


@pytest.mark.parametrize(
    ("metadata", "expected_id"),
    [
        ({"example_id": 7, "source": b"integer"}, 7),
        (
            {"example_id": b"example-a", "source": b"string"},
            5862446126654077062,
        ),
        (
            {
                "dataset": b"dataset",
                "split": b"train",
                "clip_id": b"clip-a",
            },
            1147561614405543418,
        ),
    ],
)
def test_sidecar_preserves_valid_id_hashes_and_json_shape(
    tmp_path, metadata, expected_id
):
    path = tmp_path / "metadata.jsonl"
    ds = tf.data.Dataset.from_tensors({"x": 1, "metadata": metadata})

    list(attach_sidecar_writer(ds, str(path)))

    assert expected_id == (
        stable_int64_hash("example-a")
        if metadata.get("example_id") == b"example-a"
        else expected_id
    )
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record == {
        "example_id": expected_id,
        "metadata": {
            key: value.decode("utf-8")
            for key, value in metadata.items()
            if isinstance(value, bytes)
        },
    }
    assert path.read_text(encoding="utf-8") == (
        json.dumps(record, sort_keys=True) + "\n"
    )


def test_sidecar_deduplicates_repeated_dataset_iteration(tmp_path):
    path = tmp_path / "metadata.jsonl"
    ds, _n = _load(metadata_mode="numeric_only", sidecar_metadata_path=str(path))

    list(ds)
    first_contents = path.read_bytes()
    list(ds)

    assert path.read_bytes() == first_contents
    assert len(first_contents.splitlines()) == 3


def test_sidecar_rejects_missing_identity_before_writing(tmp_path):
    path = tmp_path / "metadata.jsonl"
    ds = tf.data.Dataset.from_tensor_slices(
        {"metadata": {"source": [b"first", b"second"]}}
    )
    ds = attach_sidecar_writer(ds, str(path))

    with pytest.raises(
        tf.errors.InvalidArgumentError, match="requires either non-null"
    ):
        list(ds)

    assert path.read_text(encoding="utf-8") == ""


def test_sidecar_rejects_conflicting_duplicate_id_without_values(tmp_path):
    path = tmp_path / "metadata.jsonl"
    ds = tf.data.Dataset.from_tensor_slices(
        {
            "metadata": {
                "example_id": np.array([42, 42], dtype=np.int64),
                "source": np.array([b"private-a", b"private-b"]),
            }
        }
    )
    ds = attach_sidecar_writer(ds, str(path))

    with pytest.raises(tf.errors.InvalidArgumentError) as error:
        list(ds)

    message = str(error.value)
    assert "Conflicting metadata for sidecar ID 42" in message
    assert "private-a" not in message
    assert "private-b" not in message


def test_partial_consumption_writes_joinable_sidecar_records(tmp_path):
    path = tmp_path / "metadata.jsonl"
    ds, _n = _load(metadata_mode="numeric_only", sidecar_metadata_path=str(path))

    first_batch = next(iter(ds))
    sidecar = MetadataSidecar.read_jsonl(str(path))

    for example_id in first_batch["metadata"]["example_id"].numpy():
        assert int(example_id) in sidecar.records


def test_recursive_padding_nested_metadata():
    ds, _n = _load()
    batches = list(ds.take(2))
    final = batches[-1]

    np.testing.assert_allclose(final["metadata"]["nested"]["score"].numpy(), [0.3, 0.0])
    np.testing.assert_array_equal(
        final["metadata"]["clip_id"].numpy(), [b"clip-c", b""]
    )


def test_padding_mask_added_for_short_final_batch():
    ds, _n = _load()
    final = list(ds.take(2))[-1]

    np.testing.assert_array_equal(final["padding_mask"].numpy(), [True, False])


def test_metadata_none_removes_metadata():
    ds, _n = _load(metadata_mode="none")
    batch = next(iter(ds))

    assert "metadata" not in batch


@pytest.mark.parametrize("metadata_mode", ["full", "numeric_only", "none"])
def test_padding_preserves_top_level_strings_across_metadata_modes(metadata_mode):
    ds, _n = _load(metadata_mode=metadata_mode)
    first, final = list(ds.take(2))

    np.testing.assert_array_equal(first["label"].numpy(), [b"class-a", b"class-b"])
    np.testing.assert_array_equal(final["label"].numpy(), [b"class-c", b""])
    np.testing.assert_array_equal(final["padding_mask"].numpy(), [True, False])

    if metadata_mode == "none":
        assert "metadata" not in final
        return

    np.testing.assert_allclose(final["metadata"]["nested"]["score"].numpy(), [0.3, 0.0])
    if metadata_mode == "full":
        np.testing.assert_array_equal(
            final["metadata"]["nested"]["city"].numpy(), [b"Rome", b""]
        )
    else:
        assert "city" not in final["metadata"]["nested"]

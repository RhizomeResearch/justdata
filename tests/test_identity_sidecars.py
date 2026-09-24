import hashlib
import os
from unittest.mock import patch

import numpy as np
import pytest
import tensorflow as tf

import justdata.vision  # noqa: F401
from justdata.core import (
    InventoryRecord,
    InventoryAsset,
    InventorySource,
    MetadataSidecar,
    admit_inventory,
    get_pipeline,
    load_inventory,
)
from justdata.core.finalization import finalize_dataset
from justdata.core.loader import load_ds
from justdata.core.metadata import (
    attach_sidecar_views,
    stable_int64_hash,
    verify_sidecar_keys,
)


def _identity(sample, *args, **kwargs):
    return sample


def _source(ids):
    return tf.data.Dataset.from_tensor_slices(
        {
            "x": np.arange(len(ids), dtype=np.int32),
            "metadata": {
                "example_id": ids,
                "source_group": [b"camera-a"] * len(ids),
                "payload_identity": [b"sha256:abc"] * len(ids),
                "annotation_coverage": [b"dense"] * len(ids),
                "quality": np.full(len(ids), 7, dtype=np.int32),
            },
        }
    )


def test_immutable_mapping_survives_reordering_and_padded_rows():
    sidecar = MetadataSidecar.from_metadata(
        sample["metadata"]
        for sample in _source([b"revision-a", b"revision-b"]).as_numpy_iterator()
    )
    original_digest = sidecar.digest()
    first, _ = finalize_dataset(
        _source([b"revision-b", b"revision-a"]),
        postprocess_fn=_identity,
        num_classes=None,
        batch_size=3,
        metadata_mode="numeric_only",
        metadata_sidecar=sidecar,
        prefetch=False,
    )
    batch = next(iter(first))
    np.testing.assert_array_equal(batch["padding_mask"], [True, True, False])
    np.testing.assert_array_equal(
        batch["metadata"]["row_id"],
        [stable_int64_hash("revision-b"), stable_int64_hash("revision-a"), 0],
    )
    assert "example_id" not in batch["metadata"]
    assert (
        sidecar.records[stable_int64_hash("revision-a")]["annotation_coverage"]
        == "dense"
    )
    assert sidecar.digest() == original_digest


def test_immutable_mapping_is_detached_when_dataset_is_built():
    sidecar = MetadataSidecar.from_metadata([{"example_id": "revision-a"}])
    ds, _ = finalize_dataset(
        tf.data.Dataset.from_tensors(
            {"x": 1, "metadata": {"example_id": tf.constant("revision-a")}}
        ),
        postprocess_fn=_identity,
        num_classes=None,
        batch_size=1,
        metadata_mode="numeric_only",
        metadata_sidecar=sidecar,
    )
    sidecar.records.clear()
    assert next(iter(ds))["metadata"]["row_id"].numpy()[0] == stable_int64_hash(
        "revision-a"
    )


def test_precomputed_view_mapping_keeps_string_descriptors_out_of_batch():
    sidecar = MetadataSidecar.from_metadata([{"example_id": "revision-a"}])
    source_id = stable_int64_hash("revision-a")
    view_metadata = {"corruption": "identity", "severity": 1}
    view_id = sidecar.add_view_metadata(source_id, view_metadata)

    def postprocess(sample, num_classes=None):
        del num_classes
        return sample | {
            "metadata": sample["metadata"]
            | {"corruption": tf.constant("identity"), "severity": tf.constant(1)}
        }

    ds, _ = finalize_dataset(
        tf.data.Dataset.from_tensors(
            {"x": 1, "metadata": {"example_id": tf.constant("revision-a")}}
        ),
        postprocess_fn=postprocess,
        num_classes=None,
        batch_size=1,
        metadata_mode="numeric_only",
        metadata_sidecar=sidecar,
        prefetch=False,
    )
    batch = next(iter(ds))
    assert "corruption" not in batch["metadata"]
    assert int(batch["metadata"]["view_id"][0]) == view_id
    assert sidecar.view_records[(source_id, view_id)] == view_metadata


@pytest.mark.parametrize("bind", [attach_sidecar_views, verify_sidecar_keys])
def test_sidecar_binders_require_exactly_one_mapping_source(tmp_path, bind):
    ds = _source([b"a"])

    with pytest.raises(ValueError, match="Choose either"):
        bind(ds)
    with pytest.raises(ValueError, match="Choose either"):
        bind(ds, sidecar=MetadataSidecar(), path=str(tmp_path / "views.jsonl"))


def test_zero_is_a_valid_real_key_and_padding_is_not_a_mapping():
    sidecar = MetadataSidecar.from_metadata([{"example_id": 0, "source": "zero"}])
    ds, _ = finalize_dataset(
        tf.data.Dataset.from_tensors({"x": 1, "metadata": {"example_id": 0}}),
        postprocess_fn=_identity,
        num_classes=None,
        batch_size=2,
        metadata_mode="numeric_only",
        metadata_sidecar=sidecar,
        prefetch=False,
    )
    batch = next(iter(ds))
    np.testing.assert_array_equal(batch["metadata"]["row_id"], [0, 0])
    np.testing.assert_array_equal(batch["padding_mask"], [True, False])


def test_complete_mapping_rejects_conflicts_and_hash_collisions():
    record = {"example_id": "a", "coverage": "full", "source_group": "g"}
    sidecar = MetadataSidecar.from_metadata([record, record])
    assert len(sidecar.records) == 1
    with pytest.raises(ValueError, match="Conflicting metadata"):
        MetadataSidecar.from_metadata([record, record | {"coverage": "partial"}])
    with patch("justdata.core.metadata.stable_int64_hash", return_value=5):
        with pytest.raises(ValueError, match="Conflicting"):
            MetadataSidecar.from_metadata([record, record | {"example_id": "b"}])


def test_sidecar_resume_validates_and_atomic_failure_preserves_previous_bytes(tmp_path):
    path = tmp_path / "metadata.jsonl"
    first = MetadataSidecar.from_metadata([{"example_id": "a", "coverage": "full"}])
    first.write_jsonl(str(path))
    accepted = path.read_bytes()
    first.write_jsonl(str(path))
    assert path.read_bytes() == accepted

    conflict = MetadataSidecar.from_metadata(
        [{"example_id": "a", "coverage": "partial"}]
    )
    with pytest.raises(ValueError, match="Conflicting metadata"):
        conflict.write_jsonl(str(path))
    assert path.read_bytes() == accepted

    second = MetadataSidecar.from_metadata([{"example_id": "b", "coverage": "full"}])
    with patch(
        "justdata.core.metadata.os.replace", side_effect=OSError("write failed")
    ):
        with pytest.raises(OSError, match="write failed"):
            second.write_jsonl(str(path))
    assert path.read_bytes() == accepted
    assert sorted(tmp_path.iterdir()) == [path]

    second.write_jsonl(str(path))
    assert len(MetadataSidecar.read_jsonl(str(path)).records) == 2
    with pytest.raises(FileExistsError):
        second.write_jsonl(str(path), policy="create")
    second.write_jsonl(str(path), policy="overwrite")
    assert len(MetadataSidecar.read_jsonl(str(path)).records) == 1


def test_legacy_sidecar_is_readable_but_cannot_claim_complete_identity(tmp_path):
    path = tmp_path / "old.jsonl"
    old = MetadataSidecar()
    old.add(7, {"source": "old"})
    old.write_jsonl(str(path))
    assert MetadataSidecar.read_jsonl(str(path)).records[7] == {"source": "old"}
    with pytest.raises(ValueError, match="full identity evidence"):
        finalize_dataset(
            tf.data.Dataset.from_tensors({"x": 1, "metadata": {"example_id": 7}}),
            postprocess_fn=_identity,
            num_classes=None,
            batch_size=1,
            metadata_mode="numeric_only",
            metadata_sidecar=MetadataSidecar.read_jsonl(str(path)),
        )
    resumed, _ = finalize_dataset(
        tf.data.Dataset.from_tensors(
            {"x": 1, "metadata": {"example_id": 7, "source": tf.constant("old")}}
        ),
        postprocess_fn=_identity,
        num_classes=None,
        batch_size=1,
        metadata_mode="numeric_only",
        sidecar_metadata_path=str(path),
    )
    with pytest.raises(tf.errors.InvalidArgumentError, match="rebuild the artifact"):
        list(resumed)


def test_inventory_mapping_carries_source_and_payload_provenance(tmp_path):
    payload = tmp_path / "source.bin"
    payload.write_bytes(b"verified payload")
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    source = InventorySource(
        splits={
            "validation": [
                InventoryRecord(
                    "revision/a",
                    assets={"payload": InventoryAsset(payload, digest)},
                    metadata={"coverage": "dense", "quality": 7},
                )
            ]
        },
        reader=lambda _payloads, metadata: {"x": tf.constant(metadata["quality"])},
    )
    admitted = admit_inventory(
        {"camera-a": source},
        {"camera-a": ["validation"]},
        inventory_id="static-inventory",
        snapshot_dir=tmp_path / "inventory",
        output_signature={"x": tf.TensorSpec([], tf.int32)},
    )
    sidecar = MetadataSidecar.from_inventory(admitted)
    row_id = stable_int64_hash("revision/a")
    provenance = sidecar.records[row_id]["source_record"]
    assert provenance["record_id"] == "revision/a"
    assert provenance["source"] == "camera-a"
    assert provenance["split"] == "validation"
    assert provenance["metadata"]["coverage"] == "dense"
    assert provenance["assets"][0]["role"] == "payload"
    assert provenance["assets"][0]["sha256"] == digest

    batches, _ = load_inventory(
        admitted,
        dataset_type="validation",
        batch_size=2,
        seed=0,
        preprocess_fn=_identity,
        postprocess_fn=_identity,
        metadata_mode="numeric_only",
        metadata_sidecar=sidecar,
    )
    batch = next(iter(batches))
    np.testing.assert_array_equal(batch["metadata"]["row_id"], [row_id, 0])
    np.testing.assert_array_equal(batch["padding_mask"], [True, False])


def test_streaming_restart_and_view_geometry_keep_source_mapping(tmp_path):
    path = tmp_path / "stream.jsonl"

    def load(ids):
        with patch("justdata.core.loader.fetch_ds", return_value=_source(ids)):
            return load_ds(
                "fixture",
                "validation",
                "train",
                2,
                17,
                preprocess_fn=_identity,
                augment_fn=lambda sample, seed: (
                    sample
                    | {
                        "geometry": {
                            "crop_start": tf.random.stateless_uniform(
                                [], seed=seed, maxval=1000, dtype=tf.int32
                            )
                        }
                    }
                ),
                postprocess_fn=_identity,
                metadata_mode="numeric_only",
                sidecar_metadata_path=os.fspath(path),
                deterministic=True,
                shuffle_buffer=1,
                return_raw_ds=True,
                map_parallel_calls=2,
            )

    raw, tools = load([b"revision-a", b"revision-b"])
    first, _ = tools["finalize_epoch"](raw, seed=1)
    first_batch = next(iter(first))
    before = MetadataSidecar.read_jsonl(str(path))
    assert stable_int64_hash("revision-a") in before.records

    restarted, resumed_tools = load([b"revision-b", b"revision-a"])
    second, _ = resumed_tools["finalize_epoch"](restarted, seed=2)
    second_batch = next(iter(second))
    assert set(first_batch["metadata"]["row_id"].numpy()) == set(
        second_batch["metadata"]["row_id"].numpy()
    )
    assert "crop_start" in first_batch["geometry"]
    assert "crop_start" in second_batch["geometry"]
    assert not np.array_equal(
        first_batch["geometry"]["crop_start"],
        second_batch["geometry"]["crop_start"],
    )
    assert all("geometry" not in record for record in before.records.values())
    assert MetadataSidecar.read_jsonl(str(path)).records == before.records


def test_model_cache_cannot_emit_keys_missing_from_reused_sidecar(tmp_path):
    cache_path = str(tmp_path / "model-cache")
    first_path = tmp_path / "first.jsonl"
    first, _ = finalize_dataset(
        _source([b"revision-a"]),
        postprocess_fn=_identity,
        num_classes=None,
        batch_size=1,
        metadata_mode="numeric_only",
        sidecar_metadata_path=str(first_path),
        cache_model_inputs=True,
        model_input_cache_path=cache_path,
        prefetch=False,
    )
    list(first)
    assert first_path.exists()

    reused, _ = finalize_dataset(
        _source([b"revision-a"]),
        postprocess_fn=_identity,
        num_classes=None,
        batch_size=1,
        metadata_mode="numeric_only",
        sidecar_metadata_path=str(tmp_path / "missing.jsonl"),
        cache_model_inputs=True,
        model_input_cache_path=cache_path,
        prefetch=False,
    )
    with pytest.raises(tf.errors.InvalidArgumentError, match="Unknown sidecar ID"):
        list(reused)


def test_model_cache_rejects_changed_mapping_for_same_numeric_key(tmp_path):
    cache_path = str(tmp_path / "model-cache")
    source_metadata = next(iter(_source([b"revision-a"]).as_numpy_iterator()))[
        "metadata"
    ]
    original = MetadataSidecar.from_metadata([source_metadata])
    changed = MetadataSidecar.from_metadata(
        [source_metadata | {"annotation_coverage": b"partial"}]
    )
    first, _ = finalize_dataset(
        _source([b"revision-a"]),
        postprocess_fn=_identity,
        num_classes=None,
        batch_size=1,
        metadata_mode="numeric_only",
        metadata_sidecar=original,
        cache_model_inputs=True,
        model_input_cache_path=cache_path,
        prefetch=False,
    )
    list(first)

    reused, _ = finalize_dataset(
        _source([b"revision-a"]),
        postprocess_fn=_identity,
        num_classes=None,
        batch_size=1,
        metadata_mode="numeric_only",
        metadata_sidecar=changed,
        cache_model_inputs=True,
        model_input_cache_path=cache_path,
        prefetch=False,
    )
    with pytest.raises(
        tf.errors.InvalidArgumentError, match="Conflicting sidecar mapping"
    ):
        list(reused)


def test_executed_config_records_complete_mapping_digest():
    sidecar = MetadataSidecar.from_metadata([{"example_id": "revision-a"}])
    source = tf.data.Dataset.from_tensors(
        {
            "image": tf.zeros([12, 16, 3], tf.uint8),
            "label": tf.constant(0, tf.int64),
            "metadata": {"example_id": tf.constant("revision-a")},
        }
    )
    with patch("justdata.core.loader.fetch_ds", return_value=source):
        _raw, _tools, snapshot = load_ds(
            "fixture",
            "validation",
            "validation",
            1,
            17,
            num_classes=10,
            pipeline=get_pipeline(dataset="cifar10"),
            metadata_mode="numeric_only",
            metadata_sidecar=sidecar,
            return_raw_ds=True,
            return_config=True,
        )
    assert (
        snapshot.to_dict()["execution"]["metadata"]["sidecar_digest"]
        == sidecar.digest()
    )

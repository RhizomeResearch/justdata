import hashlib
import json
import socket
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from justdata.core import (
    InventoryAsset,
    InventoryFilter,
    InventoryLoadError,
    InventoryRecord,
    InventorySource,
    admit_inventory,
    load_inventory,
    open_inventory,
)
from justdata.core.adapters import get_adapter


def _reader(payloads, metadata):
    return {"x": tf.constant(int(payloads["value"]), tf.int32)}


def _source(tmp_path, values=(7, 3, 9), *, prefix="row", reader=_reader):
    records = []
    for index, value in enumerate(values):
        path = tmp_path / f"{prefix}-{index}.bin"
        content = str(value).encode()
        path.write_bytes(content)
        records.append(
            InventoryRecord(
                f"{prefix}-{index}",
                {"value": InventoryAsset(path, hashlib.sha256(content).hexdigest())},
                {"ordinal": index},
            )
        )
    return InventorySource({"test": records}, reader)


def _admit(tmp_path, source=None, **kwargs):
    source = source if source is not None else _source(tmp_path)
    options = {
        "inventory_id": "fixture-revision-1",
        "snapshot_dir": tmp_path / "snapshot",
        "output_signature": {"x": tf.TensorSpec([], tf.int32)},
    }
    options.update(kwargs)
    return admit_inventory({"local": source}, {"local": ["test"]}, **options)


def _rows(dataset):
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    return list(dataset.with_options(options).as_numpy_iterator())


def _ids(dataset):
    return [row["metadata"]["example_id"].decode() for row in _rows(dataset)]


def _identity(sample, **kwargs):
    return sample


def test_offline_ordered_snapshot_survives_source_removal(tmp_path, monkeypatch):
    source = _source(tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("Local admission attempted networking or legacy source loading")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr("tensorflow_datasets.load", forbidden)
    monkeypatch.setattr("justdata.core.loader.fetch_ds", forbidden)
    admitted = _admit(tmp_path, source)
    for record in source.splits["test"]:
        Path(record.assets["value"].path).unlink()
    reopened = open_inventory(admitted.path)
    assert _ids(reopened.dataset) == ["row-0", "row-1", "row-2"]
    assert [row["x"] for row in _rows(reopened.dataset)] == [7, 3, 9]
    assert reopened.dataset.cardinality().numpy() == 3
    assert reopened.report == admitted.report
    assert reopened.report["complete"] is True
    assert reopened.report["requested_count"] == reopened.report["retained_count"] == 3
    assert reopened.report["excluded_count"] == 0
    assert reopened.report["failures"] == []
    assert reopened.report["splits"][0]["requested_ids"] == _ids(reopened.dataset)


def test_requested_source_split_and_record_order(tmp_path):
    first = _source(tmp_path, [1, 2], prefix="first")
    second = _source(tmp_path, [3, 4], prefix="second")
    first = replace(
        first, splits={"a": first.splits["test"][:1], "b": first.splits["test"][1:]}
    )
    admitted = admit_inventory(
        {"first": first, "second": second},
        {"second": ["test"], "first": ["b", "a"]},
        inventory_id="ordered",
        snapshot_dir=tmp_path / "snapshot",
        output_signature={"x": tf.TensorSpec([], tf.int32)},
    )
    assert _ids(admitted.dataset) == ["second-0", "second-1", "first-1", "first-0"]


def test_two_sources_one_unavailable_fails_before_reading(tmp_path):
    def unread(*args):
        pytest.fail("Preflight must validate all sources before reading")

    source = _source(tmp_path, reader=unread)
    with pytest.raises(InventoryLoadError) as caught:
        admit_inventory(
            {"available": source},
            {"available": ["test"], "unavailable": ["test"]},
            inventory_id="two-sources",
            snapshot_dir=tmp_path / "snapshot",
            output_signature={"x": tf.TensorSpec([], tf.int32)},
        )
    assert caught.value.issue["code"] == "missing_source"
    assert caught.value.issue["source"] == "unavailable"
    assert caught.value.report["complete"] is False
    assert not (tmp_path / "snapshot").exists()


def test_second_source_failure_never_returns_successful_subset(tmp_path):
    first = _source(tmp_path, [1], prefix="first")
    second = _source(tmp_path, [2], prefix="second")
    Path(second.splits["test"][0].assets["value"].path).unlink()
    with pytest.raises(InventoryLoadError) as caught:
        admit_inventory(
            {"first": first, "second": second},
            {"first": ["test"], "second": ["test"]},
            inventory_id="two-sources",
            snapshot_dir=tmp_path / "snapshot",
            output_signature={"x": tf.TensorSpec([], tf.int32)},
        )
    assert caught.value.issue["source"] == "second"
    assert caught.value.issue["record_id"] == "second-0"
    assert caught.value.report["retained_count"] == 1
    assert caught.value.report["requested_count"] == 2
    assert not (tmp_path / "snapshot").exists()


@pytest.mark.parametrize(
    ("splits", "code"),
    [
        ([], "invalid_splits"),
        (["missing"], "missing_split"),
        (["test", "test"], "invalid_splits"),
        ([""], "invalid_splits"),
        ("test", "invalid_splits"),
    ],
)
def test_missing_or_empty_split_declarations_fail(tmp_path, splits, code):
    with pytest.raises(InventoryLoadError) as caught:
        admit_inventory(
            {"local": _source(tmp_path)},
            {"local": splits},
            inventory_id="splits",
            snapshot_dir=tmp_path / "snapshot",
            output_signature={"x": tf.TensorSpec([], tf.int32)},
        )
    assert caught.value.issue["code"] == code
    assert caught.value.issue["source"] == "local"
    assert not (tmp_path / "snapshot").exists()


def test_explicitly_empty_split_succeeds(tmp_path):
    source = InventorySource({"test": ()}, lambda *_: pytest.fail("Empty source read"))
    admitted = _admit(tmp_path, source)
    assert _rows(admitted.dataset) == []
    assert _rows(open_inventory(admitted.path).dataset) == []
    assert admitted.report["complete"] is True
    assert admitted.report["splits"][0]["requested_count"] == 0
    assert admitted.dataset.cardinality().numpy() == 0
    batches, count = load_inventory(
        admitted,
        "validation",
        2,
        0,
        preprocess_fn=_identity,
        postprocess_fn=_identity,
        private_threadpool_size=1,
    )
    assert count == 0
    assert _rows(batches) == []


@pytest.mark.parametrize("across_sources", [False, True])
def test_duplicate_ids_rejected_before_payload_io(tmp_path, across_sources):
    source = _source(tmp_path, [1])
    if across_sources:
        sources, splits = {"a": source, "b": source}, {"a": ["test"], "b": ["test"]}
    else:
        source = replace(source, splits={"test": source.splits["test"] * 2})
        sources, splits = {"a": source}, {"a": ["test"]}
    Path(source.splits["test"][0].assets["value"].path).unlink()
    with pytest.raises(InventoryLoadError) as caught:
        admit_inventory(
            sources,
            splits,
            inventory_id="duplicates",
            snapshot_dir=tmp_path / "snapshot",
            output_signature={"x": tf.TensorSpec([], tf.int32)},
        )
    assert caught.value.issue["code"] == "duplicate_id"
    assert caught.value.issue["record_id"] == "row-0"
    assert not (tmp_path / "snapshot").exists()


def test_changed_source_bytes_fail_with_expected_actual_digests(tmp_path):
    source = _source(tmp_path)
    path = Path(source.splits["test"][1].assets["value"].path)
    path.write_bytes(b"different content")
    with pytest.raises(InventoryLoadError) as caught:
        _admit(tmp_path, source)
    issue = caught.value.issue
    assert issue["code"] == "digest_mismatch"
    assert issue["record_id"] == "row-1"
    assert issue["path"] == str(path)
    assert issue["expected"] == hashlib.sha256(b"3").hexdigest()
    assert issue["actual"] == hashlib.sha256(b"different content").hexdigest()
    assert caught.value.report["retained_count"] == 1
    assert not (tmp_path / "snapshot").exists()


def test_reader_receives_the_verified_bytes_without_reopening(tmp_path):
    source = _source(tmp_path, [17])
    path = Path(source.splits["test"][0].assets["value"].path)

    def reader(payloads, metadata):
        path.write_bytes(b"99")
        return _reader(payloads, metadata)

    admitted = _admit(tmp_path, replace(source, reader=reader))
    assert _rows(admitted.dataset)[0]["x"] == 17
    with pytest.raises(InventoryLoadError, match="digest_mismatch"):
        _admit(tmp_path, source, snapshot_dir=tmp_path / "readmission")
    assert _rows(open_inventory(admitted.path).dataset)[0]["x"] == 17


def test_iteration_time_decode_failure_is_attributed_and_unpublished(tmp_path):
    records = []
    for index, content in enumerate(
        [tf.io.encode_png(tf.ones([2, 3, 3], tf.uint8)).numpy(), b"corrupt png"]
    ):
        path = tmp_path / f"{index}.png"
        path.write_bytes(content)
        records.append(
            InventoryRecord(
                f"image-{index}",
                {"image": InventoryAsset(path, hashlib.sha256(content).hexdigest())},
            )
        )

    def decode(payloads, metadata):
        # Failure occurs in tf.data iteration, not during dataset construction.
        dataset = tf.data.Dataset.from_tensors(payloads["image"]).map(
            lambda b: tf.io.decode_png(b, channels=3)
        )
        options = tf.data.Options()
        options.threading.private_threadpool_size = 1
        return {"image": next(iter(dataset.with_options(options)))}

    with pytest.raises(InventoryLoadError) as caught:
        _admit(
            tmp_path,
            InventorySource({"test": records}, decode),
            output_signature={"image": tf.TensorSpec([None, None, 3], tf.uint8)},
        )
    assert caught.value.issue["code"] == "reader_failed"
    assert caught.value.issue["source"] == "local"
    assert caught.value.issue["split"] == "test"
    assert caught.value.issue["record_id"] == "image-1"
    assert isinstance(caught.value.__cause__, tf.errors.OpError)
    assert caught.value.report["retained_count"] == 1
    assert caught.value.report["complete"] is False
    json.dumps(caught.value.to_dict(), allow_nan=False)
    assert not (tmp_path / "snapshot").exists()


@pytest.mark.parametrize(
    "reader",
    [
        lambda *_: None,
        lambda *_: {"other": 1},
        lambda *_: {"x": tf.constant(1, tf.int64)},
        lambda *_: {"x": [1]},
        lambda *_: {"x": 1, "metadata": {"example_id": "wrong"}},
    ],
)
def test_invalid_records_are_not_coerced_or_dropped(tmp_path, reader):
    with pytest.raises(InventoryLoadError) as caught:
        _admit(tmp_path, _source(tmp_path, [1], reader=reader))
    assert caught.value.issue["code"] == "invalid_record"
    assert caught.value.issue["record_id"] == "row-0"
    assert not (tmp_path / "snapshot").exists()


def test_required_adapter_lookup_preserves_legacy_fallback(tmp_path, monkeypatch):
    from justdata.core.adapters import _ADAPTERS, _default_adapter

    unknown = "jd01-unregistered:variant"
    assert get_adapter(unknown) is _default_adapter
    with pytest.raises(ValueError, match="No adapter registered"):
        get_adapter(unknown, required=True)
    source = replace(_source(tmp_path, [3]), adapter=unknown)
    with pytest.raises(InventoryLoadError, match="unknown_adapter"):
        _admit(tmp_path, source)

    monkeypatch.setitem(_ADAPTERS, "jd01:", lambda sample: {"x": sample["x"] + 1})
    admitted = _admit(tmp_path, replace(source, adapter="jd01:reviewed"))
    assert _rows(admitted.dataset)[0]["x"] == 4
    assert _ids(admitted.dataset) == ["row-0"]


@pytest.mark.parametrize("phase", ["adapter", "validate"])
def test_callback_failures_preserve_cause_and_record(tmp_path, phase):
    cause = RuntimeError("domain validation failed")

    def fail(sample):
        raise cause

    source = replace(_source(tmp_path, [1]), **{phase: fail})
    with pytest.raises(InventoryLoadError) as caught:
        _admit(tmp_path, source)
    assert caught.value.__cause__ is cause
    assert caught.value.issue["code"] == (
        "adapter_failed" if phase == "adapter" else "invalid_record"
    )
    assert caught.value.issue["record_id"] == "row-0"


@pytest.mark.parametrize("keep_none", [False, True])
def test_declared_selection_reconciles_actual_identities(tmp_path, keep_none):
    selection = InventoryFilter(
        "value-above-five:v1",
        lambda sample: tf.greater(sample["x"], 100 if keep_none else 5),
    )
    admitted = _admit(tmp_path, selection=selection)
    expected = [] if keep_none else ["row-0", "row-2"]
    group = admitted.report["splits"][0]
    assert group["requested_ids"] == ["row-0", "row-1", "row-2"]
    assert group["retained_ids"] == _ids(admitted.dataset) == expected
    assert group["excluded_ids"] == (
        ["row-0", "row-1", "row-2"] if keep_none else ["row-1"]
    )
    assert admitted.report["retained_count"] == len(expected)
    assert admitted.report["excluded_count"] == 3 - len(expected)
    assert admitted.report["selection"] == {
        "inventory_id": "fixture-revision-1",
        "transformation_id": "value-above-five:v1",
    }
    assert open_inventory(admitted.path).report == admitted.report


def test_filter_cannot_hide_bad_record_or_mutate_snapshot(tmp_path):
    source = _source(tmp_path, [1, "invalid"])
    with pytest.raises(InventoryLoadError, match="reader_failed"):
        _admit(
            tmp_path, source, selection=InventoryFilter("exclude-all", lambda _: False)
        )

    def mutating_filter(sample):
        sample["x"] = tf.constant(99)
        sample["metadata"]["example_id"] = tf.constant("wrong")
        return True

    admitted = _admit(
        tmp_path,
        _source(tmp_path, [1]),
        selection=InventoryFilter("keep", mutating_filter),
    )
    assert _ids(admitted.dataset) == ["row-0"]
    assert _rows(admitted.dataset)[0]["x"] == 1


@pytest.mark.parametrize(
    "predicate",
    [
        lambda _: 1,
        lambda _: [True],
        lambda _: (_ for _ in ()).throw(RuntimeError("bad filter")),
    ],
)
def test_filter_failure_is_not_exclusion(tmp_path, predicate):
    with pytest.raises(InventoryLoadError) as caught:
        _admit(tmp_path, selection=InventoryFilter("broken", predicate))
    assert caught.value.issue["code"] == "filter_failed"
    assert caught.value.issue["record_id"] == "row-0"
    assert caught.value.report["excluded_count"] == 0


def test_variable_shapes_and_unicode_ids_roundtrip(tmp_path):
    records = [
        InventoryRecord(f"échantillon-{i}", metadata={"size": i + 1}) for i in range(3)
    ]
    source = InventorySource(
        {"test": records}, lambda _, metadata: {"x": tf.range(metadata["size"])}
    )
    admitted = _admit(
        tmp_path, source, output_signature={"x": tf.TensorSpec([None], tf.int32)}
    )
    assert _ids(open_inventory(admitted.path).dataset) == [r.record_id for r in records]
    assert [row["x"].tolist() for row in _rows(admitted.dataset)] == [
        [0],
        [0, 1],
        [0, 1, 2],
    ]


@pytest.mark.parametrize("artifact", ["records.tfrecord", "schema.json", "report.json"])
def test_snapshot_artifact_corruption_is_detected(tmp_path, artifact):
    admitted = _admit(tmp_path)
    path = admitted.path / artifact
    path.write_bytes(path.read_bytes() + b"bad")
    with pytest.raises(InventoryLoadError) as caught:
        open_inventory(admitted.path)
    assert caught.value.issue["code"] == "snapshot_digest_mismatch"
    assert caught.value.issue["path"] == str(path)


def test_incomplete_snapshot_cannot_be_opened_and_is_not_deleted(tmp_path):
    path = tmp_path / "snapshot"
    path.mkdir()
    (path / "manifest.pending").write_text("{}")
    with pytest.raises(InventoryLoadError, match="invalid_snapshot"):
        open_inventory(path)
    assert (path / "manifest.pending").read_text() == "{}"


def test_existing_destination_is_never_overwritten(tmp_path):
    source = _source(tmp_path)
    admitted = _admit(tmp_path, source)
    contents = {p.name: p.read_bytes() for p in admitted.path.iterdir()}
    with pytest.raises(InventoryLoadError, match="snapshot_exists"):
        _admit(tmp_path, source)
    assert contents == {p.name: p.read_bytes() for p in admitted.path.iterdir()}


def test_snapshot_write_failure_cleans_owned_directory(tmp_path, monkeypatch):
    original = Path.write_bytes

    def fail(path, content):
        if path.name == "schema.json":
            raise OSError("disk full")
        return original(path, content)

    source = _source(tmp_path)
    monkeypatch.setattr(Path, "write_bytes", fail)
    with pytest.raises(InventoryLoadError) as caught:
        _admit(tmp_path, source)
    assert caught.value.issue["code"] == "snapshot_io"
    assert caught.value.report["complete"] is False
    assert not (tmp_path / "snapshot").exists()
    assert all(Path(r.assets["value"].path).exists() for r in source.splits["test"])


def test_snapshot_readback_detects_silently_missing_record(tmp_path, monkeypatch):
    real_writer = tf.io.TFRecordWriter

    class DroppingWriter:
        def __init__(self, path):
            self.writer = real_writer(path)
            self.writes = 0

        def __enter__(self):
            return self

        def write(self, value):
            self.writes += 1
            if self.writes != 2:
                self.writer.write(value)

        def __exit__(self, *_):
            self.writer.close()

    monkeypatch.setattr(tf.io, "TFRecordWriter", DroppingWriter)
    with pytest.raises(InventoryLoadError, match="snapshot_io") as caught:
        _admit(tmp_path)
    assert caught.value.report["complete"] is False
    assert not (tmp_path / "snapshot").exists()


@pytest.mark.parametrize(
    "change", ["disposition", "missing", "count", "selection", "failures"]
)
def test_reopen_rejects_inconsistent_report_even_with_updated_digest(tmp_path, change):
    admitted = _admit(tmp_path)
    report = admitted.report
    if change == "disposition":
        report["records"][0]["disposition"] = "excluded"
    elif change == "missing":
        report["records"].pop()
    elif change == "count":
        report["retained_count"] -= 1
    elif change == "selection":
        report["selection"] = {"inventory_id": "wrong", "transformation_id": "rule"}
    else:
        report["failures"] = "invalid failure list"
    payload = json.dumps(report).encode()
    (admitted.path / "report.json").write_bytes(payload)
    manifest_path = admitted.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["report.json"] = hashlib.sha256(payload).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(InventoryLoadError, match="invalid_snapshot"):
        open_inventory(admitted.path)


@pytest.mark.parametrize(
    "kind", ["digest", "metadata", "schema", "identity_schema", "selection"]
)
def test_invalid_declarations_fail_before_reading(tmp_path, kind):
    def unread(*_):
        pytest.fail("Invalid declarations must fail before payload I/O")

    source = _source(tmp_path, [1], reader=unread)
    options = {}
    if kind == "digest":
        record = source.splits["test"][0]
        record = replace(
            record, assets={"value": replace(record.assets["value"], sha256="short")}
        )
        source = replace(source, splits={"test": [record]})
    elif kind == "metadata":
        source = replace(
            source,
            splits={
                "test": [
                    replace(source.splits["test"][0], metadata={"bad": float("nan")})
                ]
            },
        )
    elif kind == "schema":
        options["output_signature"] = {"x": tf.TensorSpec(None, tf.int32)}
    elif kind == "identity_schema":
        options["output_signature"] = {
            "x": tf.TensorSpec([], tf.int32),
            "metadata": {"example_id": tf.TensorSpec([], tf.int64)},
        }
    else:
        options["selection"] = InventoryFilter("", lambda _: True)
    with pytest.raises(InventoryLoadError) as caught:
        _admit(tmp_path, source, **options)
    assert caught.value.issue["code"] in {
        "invalid_record",
        "invalid_schema",
        "invalid_declaration",
    }
    assert not (tmp_path / "snapshot").exists()


def test_validator_mutations_do_not_change_validated_snapshot(tmp_path):
    def validate(sample):
        sample["metadata"]["example_id"] = tf.constant("wrong")
        sample["x"] = tf.constant("wrong dtype")

    source = replace(_source(tmp_path, [1]), validate=validate)
    admitted = _admit(tmp_path, source)
    assert _ids(admitted.dataset) == ["row-0"]
    assert _rows(admitted.dataset)[0]["x"] == 1


def test_load_inventory_normal_raw_numpy_and_epoch_paths(tmp_path):
    admitted = _admit(tmp_path)
    options = dict(
        preprocess_fn=_identity,
        augment_fn=_identity,
        late_augment_fn=_identity,
        postprocess_fn=_identity,
        map_parallel_calls=1,
        private_threadpool_size=1,
        deterministic=True,
    )
    batched, count = load_inventory(admitted, "validation", 2, 7, **options)
    assert count == 2
    rows = _rows(batched)
    assert rows[0]["metadata"]["example_id"].tolist() == [b"row-0", b"row-1"]
    assert rows[1]["metadata"]["example_id"].tolist() == [b"row-2", b""]
    np.testing.assert_array_equal(rows[1]["padding_mask"], [True, False])
    raw, tools = load_inventory(admitted, "train", 2, 7, return_raw_ds=True, **options)
    epoch_a, _ = tools["finalize_epoch"](raw, seed=11, as_numpy=True)
    epoch_b, _ = tools["finalize_epoch"](raw, seed=11, as_numpy=True)
    for a, b in zip(epoch_a, epoch_b, strict=True):
        for left, right in zip(tf.nest.flatten(a), tf.nest.flatten(b), strict=True):
            np.testing.assert_array_equal(left, right)
    numpy_rows, _ = load_inventory(
        admitted, "validation", 2, 7, as_numpy=True, **options
    )
    assert isinstance(next(numpy_rows)["x"], np.ndarray)


@pytest.mark.parametrize("argument", ["source_filter_fn", "filter_fn"])
def test_strict_loader_rejects_undeclared_filters(tmp_path, argument):
    with pytest.raises(ValueError, match="InventoryFilter"):
        load_inventory(
            _admit(tmp_path), "validation", 2, 0, **{argument: lambda _: True}
        )


@pytest.mark.parametrize("modality", ["vision", "acoustic"])
def test_strict_loader_supports_both_modality_pipelines(tmp_path, modality):
    from justdata.core.registry import get_pipeline
    import justdata.vision  # noqa: F401
    import justdata.acoustic  # noqa: F401

    if modality == "vision":
        sample = {
            "image": tf.ones([8, 10, 3], tf.uint8),
            "label": tf.constant(1, tf.int64),
        }
        pipeline = get_pipeline(
            "vision/classification",
            apply_presets=False,
            aug_kwargs={"image_size": 8},
            postproc_kwargs={"image_size": 8, "num_classes": 2},
        )
    else:
        sample = {
            "waveform": tf.ones([64, 1], tf.float32),
            "sample_rate": tf.constant(16000, tf.int32),
            "label": tf.constant(1, tf.int64),
        }
        pipeline = get_pipeline("acoustic/identity", apply_presets=False)
    source = InventorySource(
        {"test": [InventoryRecord(f"{modality}-{i}") for i in range(3)]},
        lambda *_: sample,
    )
    signature = tf.nest.map_structure(
        lambda value: tf.TensorSpec(value.shape, value.dtype), sample
    )
    admitted = _admit(tmp_path, source, output_signature=signature)
    ds, count = load_inventory(
        admitted,
        "validation",
        2,
        0,
        pipeline=pipeline,
        metadata_mode="full",
        map_parallel_calls=1,
        private_threadpool_size=1,
    )
    assert count == 2
    batches = _rows(ds)
    assert batches[0]["metadata"]["example_id"].tolist() == [
        f"{modality}-0".encode(),
        f"{modality}-1".encode(),
    ]
    np.testing.assert_array_equal(batches[-1]["padding_mask"], [True, False])


def test_strict_inventory_loader_exports_executed_configuration(tmp_path):
    from justdata.core.registry import get_pipeline
    import justdata.acoustic  # noqa: F401

    admitted = _admit(tmp_path)
    pipeline = get_pipeline("acoustic/identity", apply_presets=False, overrides={})
    _dataset, count, config = load_inventory(
        admitted,
        "validation",
        2,
        7,
        pipeline=pipeline,
        return_config=True,
        map_parallel_calls=1,
        private_threadpool_size=1,
    )

    assert count == 2
    snapshot = config.to_dict()
    assert snapshot["schema_version"] == 1
    assert snapshot["pipeline"]["name"] == "acoustic/identity"
    assert snapshot["execution"]["batching"]["batch_size"] == 2

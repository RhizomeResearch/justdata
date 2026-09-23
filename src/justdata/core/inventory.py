"""Strict, offline admission of finite inventories into self-contained snapshots."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tensorflow as tf

from justdata.core.adapters import DatasetAdapter, get_adapter
from justdata.core.loader import _prepare_ds


_VERSION = "justdata.inventory.v1"
_ARTIFACTS = ("records.tfrecord", "schema.json", "report.json")
_SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")


@dataclass(frozen=True)
class InventoryAsset:
    """A local file whose complete bytes must match the supplied SHA-256."""

    path: str | os.PathLike
    sha256: str


@dataclass(frozen=True)
class InventoryRecord:
    """An expected record with a globally unique ID and JSON-compatible metadata."""

    record_id: str
    assets: Mapping[str, InventoryAsset] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InventorySource:
    """Resolved splits and a reader accepting ``(verified_bytes, metadata)``.

    The reader must return one self-contained sample, eagerly decoding external
    content. ``adapter=None`` explicitly means the reader already returns the
    canonical schema. A string selects a required registered adapter. The optional
    validator receives the canonical tensor sample and must raise on invalid data.
    """

    splits: Mapping[str, Sequence[InventoryRecord]]
    reader: Callable[[Mapping[str, bytes], Mapping[str, Any]], dict]
    adapter: str | DatasetAdapter | None = None
    validate: Callable[[dict], None] | None = None


@dataclass(frozen=True)
class InventoryFilter:
    """A named selection applied to fully validated canonical records."""

    transformation_id: str
    predicate: Callable[[dict], bool | tf.Tensor]


class InventoryLoadError(ValueError):
    """An attributable failure; ``issue`` and ``report`` are JSON-compatible.

    ``issue`` has a stable ``code`` plus available source, split, record_id, path,
    expected and actual details. ``report`` is a detached partial progress snapshot.
    The original exception is retained as ``__cause__`` when there is one.
    """

    def __init__(self, issue: dict, report: dict):
        self.issue = copy.deepcopy(issue)
        self.report = copy.deepcopy(report)
        super().__init__(f"{issue['code']}: {issue['message']}")

    def to_dict(self) -> dict:
        return copy.deepcopy({"issue": self.issue, "report": self.report})


@dataclass(frozen=True)
class AdmittedInventory:
    """A verified snapshot. Its directory must remain immutable while in use."""

    dataset: tf.data.Dataset
    report: dict
    path: Path


def _fail(report, code, message, *, cause=None, **context):
    issue = {"code": code, "message": message, **context}
    report["complete"] = False
    failures = report.get("failures", [])
    report["failures"] = [*(failures if isinstance(failures, list) else []), issue]
    raise InventoryLoadError(issue, report) from cause


def _json_bytes(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _schema(signature):
    """Encode only nested dictionaries and ordinary, known-rank TensorSpecs."""
    leaves = []

    def visit(value, path):
        if isinstance(value, dict):
            if not value or any(not isinstance(k, str) or not k for k in value):
                raise ValueError("Schema dictionaries need nonempty string keys.")
            for key in sorted(value):
                visit(value[key], [*path, key])
        elif isinstance(value, tf.TensorSpec) and value.shape.rank is not None:
            dtype = value.dtype
            if not (
                dtype.is_integer
                or dtype.is_floating
                or dtype.is_complex
                or dtype in (tf.bool, tf.string)
            ):
                raise ValueError(f"Unsupported snapshot dtype: {dtype.name}.")
            leaves.append(
                {"path": path, "dtype": dtype.name, "shape": value.shape.as_list()}
            )
        else:
            raise ValueError("Schema leaves must be TensorSpecs with known rank.")

    if not isinstance(signature, dict):
        raise ValueError("output_signature must be a nested dictionary.")
    signature = copy.deepcopy(signature)
    metadata = signature.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a dictionary.")
    identity = metadata.setdefault("example_id", tf.TensorSpec([], tf.string))
    if (
        not isinstance(identity, tf.TensorSpec)
        or identity.shape != tf.TensorShape([])
        or identity.dtype != tf.string
    ):
        raise ValueError("metadata.example_id must be a scalar string TensorSpec.")
    visit(signature, [])
    return signature, leaves


def _decode_schema(leaves):
    signature = {}
    for leaf in leaves:
        node = signature
        for key in leaf["path"][:-1]:
            node = node.setdefault(key, {})
        key = leaf["path"][-1]
        if key in node:
            raise ValueError("Duplicate schema path.")
        node[key] = tf.TensorSpec(leaf["shape"], tf.as_dtype(leaf["dtype"]))
    signature, canonical = _schema(signature)
    if canonical != leaves:
        raise ValueError("Noncanonical snapshot schema.")
    return signature


def _with_identity(sample, record_id):
    if not isinstance(sample, dict):
        raise ValueError("A reader or adapter must return one sample dictionary.")
    metadata = sample.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("Sample metadata must be a dictionary.")
    if "example_id" in metadata:
        identity = tf.convert_to_tensor(metadata["example_id"])
        if (
            identity.dtype != tf.string
            or identity.shape != tf.TensorShape([])
            or identity.numpy().decode("utf-8") != record_id
        ):
            raise ValueError("metadata.example_id conflicts with the inventory ID.")
    return sample | {"metadata": metadata | {"example_id": tf.constant(record_id)}}


def _tensorize(sample, signature):
    tf.nest.assert_same_structure(signature, sample)

    def tensor(spec, value):
        value = tf.convert_to_tensor(value)
        if value.dtype != spec.dtype or not spec.shape.is_compatible_with(value.shape):
            raise ValueError(f"Expected {spec}, received {value.dtype} {value.shape}.")
        return value

    return tf.nest.map_structure(tensor, signature, sample)


def _serialize(sample):
    features = {
        str(i): tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(value).numpy()])
        )
        for i, value in enumerate(tf.nest.flatten(sample))
    }
    return tf.train.Example(
        features=tf.train.Features(feature=features)
    ).SerializeToString()


def _snapshot_dataset(path, signature, count):
    specs = tf.nest.flatten(signature)
    fields = {str(i): tf.io.FixedLenFeature([], tf.string) for i in range(len(specs))}

    def parse(serialized):
        values = tf.io.parse_single_example(serialized, fields)
        tensors = [
            tf.ensure_shape(tf.io.parse_tensor(values[str(i)], spec.dtype), spec.shape)
            for i, spec in enumerate(specs)
        ]
        return tf.nest.pack_sequence_as(signature, tensors)

    dataset = tf.data.TFRecordDataset(os.fspath(path / "records.tfrecord"))
    return dataset.map(parse, num_parallel_calls=1, deterministic=True).apply(
        tf.data.experimental.assert_cardinality(count)
    )


def _preflight(sources, splits_info, report):
    if (
        not isinstance(sources, Mapping)
        or not isinstance(splits_info, Mapping)
        or not splits_info
    ):
        _fail(
            report,
            "invalid_declaration",
            "sources and nonempty splits_info must be mappings.",
        )
    plans = []
    seen_ids = set()
    for name, splits in splits_info.items():
        context = {"source": str(name)}
        if not isinstance(name, str) or not name:
            _fail(
                report,
                "invalid_declaration",
                "Source names must be nonempty strings.",
                **context,
            )
        if name not in sources:
            _fail(
                report, "missing_source", "Requested source is unavailable.", **context
            )
        if not isinstance(splits, (list, tuple)) or not splits:
            _fail(
                report,
                "invalid_splits",
                "Declare a nonempty list of required splits.",
                **context,
            )
        if any(not isinstance(s, str) or not s for s in splits) or len(
            set(splits)
        ) != len(splits):
            _fail(
                report,
                "invalid_splits",
                "Split names must be unique nonempty strings.",
                **context,
            )
        source = sources[name]
        if not isinstance(source, InventorySource) or not isinstance(
            source.splits, Mapping
        ):
            _fail(
                report,
                "invalid_declaration",
                "Expected an InventorySource with explicit splits.",
                **context,
            )
        if not callable(source.reader) or (
            source.validate is not None and not callable(source.validate)
        ):
            _fail(
                report,
                "invalid_declaration",
                "Reader and validator must be callable.",
                **context,
            )
        adapter = source.adapter
        if isinstance(adapter, str):
            try:
                adapter = get_adapter(adapter, required=True)
            except ValueError as exc:
                _fail(report, "unknown_adapter", str(exc), cause=exc, **context)
        if adapter is not None and not callable(adapter):
            _fail(
                report,
                "invalid_declaration",
                "Adapter must be callable, registered name, or None.",
                **context,
            )
        for split in splits:
            context = {"source": name, "split": split}
            if split not in source.splits:
                _fail(
                    report,
                    "missing_split",
                    "Requested split is not declared.",
                    **context,
                )
            records = source.splits[split]
            if not isinstance(records, (list, tuple)):
                _fail(
                    report,
                    "invalid_declaration",
                    "Split records must be a finite list or tuple.",
                    **context,
                )
            group = {
                **context,
                "requested_ids": [],
                "retained_ids": [],
                "excluded_ids": [],
                "requested_count": len(records),
                "retained_count": 0,
                "excluded_count": 0,
            }
            report["splits"].append(group)
            frozen_records = []
            for record in records:
                if (
                    not isinstance(record, InventoryRecord)
                    or not isinstance(record.record_id, str)
                    or not record.record_id
                ):
                    _fail(
                        report,
                        "invalid_record",
                        "Expected an InventoryRecord with a nonempty string ID.",
                        **context,
                    )
                record_context = {**context, "record_id": record.record_id}
                if record.record_id in seen_ids:
                    _fail(
                        report,
                        "duplicate_id",
                        "Record IDs must be unique across the requested inventory.",
                        **record_context,
                    )
                seen_ids.add(record.record_id)
                group["requested_ids"].append(record.record_id)
                report["requested_count"] += 1
                try:
                    if not isinstance(record.metadata, Mapping) or not isinstance(
                        record.assets, Mapping
                    ):
                        raise ValueError("Record metadata and assets must be mappings.")
                    metadata = json.loads(_json_bytes(dict(record.metadata)))
                    assets = {}
                    for role, asset in record.assets.items():
                        if (
                            not isinstance(role, str)
                            or not role
                            or not isinstance(asset, InventoryAsset)
                        ):
                            raise ValueError(
                                "Assets need nonempty string roles and InventoryAsset values."
                            )
                        if not isinstance(asset.sha256, str) or not _SHA256.fullmatch(
                            asset.sha256
                        ):
                            raise ValueError(
                                "Each asset requires a full SHA-256 hexadecimal digest."
                            )
                        assets[role] = InventoryAsset(
                            os.fspath(Path(asset.path).absolute()), asset.sha256.lower()
                        )
                except (TypeError, ValueError) as exc:
                    _fail(
                        report, "invalid_record", str(exc), cause=exc, **record_context
                    )
                frozen_records.append(
                    InventoryRecord(record.record_id, assets, metadata)
                )
            plans.append((source, adapter, group, frozen_records))
    return plans


def admit_inventory(
    sources: Mapping[str, InventorySource],
    splits_info: Mapping[str, Sequence[str]],
    *,
    inventory_id: str,
    snapshot_dir: str | os.PathLike,
    output_signature: dict,
    selection: InventoryFilter | None = None,
) -> AdmittedInventory:
    """Validate all requested records and publish a new offline disk snapshot.

    Source order follows ``splits_info``, then its split lists, then record order.
    Every requested record is validated before selection, including excluded ones.
    An explicitly empty split is valid; an absent split is not. No permissive
    mode, network loader, deferred media I/O, or implicit adapter is used.

    ``output_signature`` describes canonical tensors. A scalar string
    ``metadata.example_id`` is added automatically. Reader metadata is a detached
    copy of InventoryRecord.metadata; identity is supplied by the inventory.
    The parent of ``snapshot_dir`` must exist and the destination must not exist.
    """
    report = {
        "schema": _VERSION,
        "inventory_id": inventory_id if isinstance(inventory_id, str) else None,
        "complete": False,
        "selection": None,
        "requested_count": 0,
        "retained_count": 0,
        "excluded_count": 0,
        "splits": [],
        "records": [],
        "failures": [],
    }
    if not isinstance(inventory_id, str) or not inventory_id:
        _fail(report, "invalid_declaration", "inventory_id must be a nonempty string.")
    if selection is not None:
        if (
            not isinstance(selection, InventoryFilter)
            or not isinstance(selection.transformation_id, str)
            or not selection.transformation_id
            or not callable(selection.predicate)
        ):
            _fail(
                report,
                "invalid_declaration",
                "Selection requires a named InventoryFilter and callable predicate.",
            )
        report["selection"] = {
            "inventory_id": inventory_id,
            "transformation_id": selection.transformation_id,
        }
    try:
        signature, schema = _schema(output_signature)
    except (TypeError, ValueError) as exc:
        _fail(report, "invalid_schema", str(exc), cause=exc)
    plans = _preflight(sources, splits_info, report)
    try:
        path = Path(snapshot_dir).absolute()
    except (TypeError, ValueError) as exc:
        _fail(report, "invalid_declaration", str(exc), cause=exc)
    try:
        path.mkdir()
    except OSError as exc:
        _fail(
            report,
            "snapshot_exists" if isinstance(exc, FileExistsError) else "snapshot_io",
            str(exc),
            cause=exc,
            path=os.fspath(path),
        )

    published = False
    try:
        with tf.io.TFRecordWriter(os.fspath(path / "records.tfrecord")) as writer:
            for source, adapter, group, records in plans:
                for record in records:
                    context = {
                        "source": group["source"],
                        "split": group["split"],
                        "record_id": record.record_id,
                    }
                    payloads = {}
                    evidence = []
                    for role, asset in record.assets.items():
                        asset_context = {
                            **context,
                            "asset": role,
                            "path": os.fspath(asset.path),
                        }
                        try:
                            payload = Path(asset.path).read_bytes()
                        except OSError as exc:
                            _fail(
                                report,
                                "asset_read",
                                str(exc),
                                cause=exc,
                                **asset_context,
                            )
                        actual = hashlib.sha256(payload).hexdigest()
                        if actual != asset.sha256:
                            _fail(
                                report,
                                "digest_mismatch",
                                "Source bytes do not match the expected SHA-256.",
                                expected=asset.sha256,
                                actual=actual,
                                **asset_context,
                            )
                        payloads[role] = payload
                        evidence.append(
                            {
                                "role": role,
                                "path": os.fspath(asset.path),
                                "sha256": actual,
                                "size_bytes": len(payload),
                            }
                        )
                    try:
                        sample = source.reader(payloads, copy.deepcopy(record.metadata))
                    except Exception as exc:
                        _fail(report, "reader_failed", str(exc), cause=exc, **context)
                    try:
                        sample = _with_identity(sample, record.record_id)
                    except Exception as exc:
                        _fail(report, "invalid_record", str(exc), cause=exc, **context)
                    if adapter is not None:
                        try:
                            sample = adapter(sample)
                        except Exception as exc:
                            _fail(
                                report, "adapter_failed", str(exc), cause=exc, **context
                            )
                    try:
                        sample = _tensorize(
                            _with_identity(sample, record.record_id), signature
                        )
                        if source.validate is not None:
                            validation_sample = tf.nest.map_structure(
                                lambda x: x, sample
                            )
                            if source.validate(validation_sample) is not None:
                                raise ValueError(
                                    "Validators must return None and raise on invalid data."
                                )
                            del validation_sample
                        # Freeze the tensors before selection can mutate its dictionary.
                        serialized = _serialize(sample)
                    except Exception as exc:
                        _fail(report, "invalid_record", str(exc), cause=exc, **context)
                    keep = True
                    if selection is not None:
                        try:
                            decision = tf.convert_to_tensor(selection.predicate(sample))
                            if (
                                decision.dtype != tf.bool
                                or decision.shape != tf.TensorShape([])
                            ):
                                raise ValueError(
                                    "Selection must return a scalar boolean."
                                )
                            keep = bool(decision.numpy())
                        except Exception as exc:
                            _fail(
                                report, "filter_failed", str(exc), cause=exc, **context
                            )
                    disposition = "retained" if keep else "excluded"
                    if keep:
                        writer.write(serialized)
                    group[f"{disposition}_ids"].append(record.record_id)
                    group[f"{disposition}_count"] += 1
                    report[f"{disposition}_count"] += 1
                    report["records"].append(
                        {
                            **context,
                            "disposition": disposition,
                            "assets": evidence,
                            "metadata": record.metadata,
                        }
                    )
                    # Do not retain payload tensors across records.
                    del payloads, sample, serialized

        _check_report(report, complete=False)
        (path / "schema.json").write_bytes(_json_bytes(schema))
        report["snapshot"] = {
            name: _file_digest(path / name)
            for name in ("records.tfrecord", "schema.json")
        }
        report["complete"] = True
        (path / "report.json").write_bytes(_json_bytes(report))
        manifest = {
            "schema": _VERSION,
            "artifacts": {name: _file_digest(path / name) for name in _ARTIFACTS},
        }
        # Read back actual records before committing the manifest.
        dataset = _snapshot_dataset(path, signature, report["retained_count"])
        _verify_rows(dataset, report)
        (path / "manifest.pending").write_bytes(_json_bytes(manifest))
        (path / "manifest.pending").replace(path / "manifest.json")
        published = True
        return AdmittedInventory(dataset, report, path)
    except InventoryLoadError:
        raise
    except Exception as exc:
        _fail(report, "snapshot_io", str(exc), cause=exc, path=os.fspath(path))
    finally:
        if not published:
            shutil.rmtree(path)


def _check_report(report, *, complete=True):
    if (
        report["schema"] != _VERSION
        or report["complete"] is not complete
        or report["failures"] != []
    ):
        raise ValueError("Snapshot report is incomplete or unsupported.")
    requested = []
    retained = []
    excluded = []
    expected_evidence = []
    seen_splits = set()
    for group in report["splits"]:
        split_key = (group["source"], group["split"])
        if split_key in seen_splits:
            raise ValueError("Duplicate source/split declaration.")
        seen_splits.add(split_key)
        ids = group["requested_ids"]
        kept, dropped = group["retained_ids"], group["excluded_ids"]
        kept_set, dropped_set = set(kept), set(dropped)
        if kept_set & dropped_set or set(ids) != kept_set | dropped_set:
            raise ValueError("Requested identities do not reconcile.")
        for status, values in (
            ("requested", ids),
            ("retained", kept),
            ("excluded", dropped),
        ):
            if group[f"{status}_count"] != len(values) or len(values) != len(
                set(values)
            ):
                raise ValueError("Split counts or identities do not reconcile.")
        if [value for value in ids if value in kept_set] != kept or [
            value for value in ids if value in dropped_set
        ] != dropped:
            raise ValueError("Selection changed record order.")
        requested.extend(ids)
        retained.extend(kept)
        excluded.extend(dropped)
        expected_evidence.extend(
            (*split_key, value, "retained" if value in kept_set else "excluded")
            for value in ids
        )
    if len(requested) != len(set(requested)):
        raise ValueError("Duplicate inventory identities.")
    for status, values in (
        ("requested", requested),
        ("retained", retained),
        ("excluded", excluded),
    ):
        if report[f"{status}_count"] != len(values):
            raise ValueError("Inventory counts do not reconcile.")
    actual_evidence = [
        (record["source"], record["split"], record["record_id"], record["disposition"])
        for record in report["records"]
    ]
    if actual_evidence != expected_evidence:
        raise ValueError("Inventory evidence is incomplete.")
    selection = report["selection"]
    if excluded and selection is None:
        raise ValueError("Excluded records require a declared selection.")
    if selection is not None and (
        selection["inventory_id"] != report["inventory_id"]
        or not isinstance(selection["transformation_id"], str)
        or not selection["transformation_id"]
    ):
        raise ValueError("Selection is not bound to the input inventory.")
    return retained


def _verify_rows(dataset, report):
    expected = iter(
        record["record_id"]
        for record in report["records"]
        if record["disposition"] == "retained"
    )
    # Bound verification to one reader thread; pipeline controls remain configurable.
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    for sample in dataset.with_options(options):
        if sample["metadata"]["example_id"].numpy().decode("utf-8") != next(
            expected, None
        ):
            raise ValueError("Snapshot identities do not match the admission report.")
    if next(expected, None) is not None:
        raise ValueError("Snapshot is missing expected records.")


def open_inventory(snapshot_dir: str | os.PathLike) -> AdmittedInventory:
    """Verify snapshot artifacts, reconcile identities, and reopen without sources.

    Manifest digests detect changes relative to the local manifest; they are not
    an authenticity signature. Treat admitted directories as immutable while used.
    """
    report = {"schema": _VERSION, "complete": False, "failures": []}
    path = None
    try:
        path = Path(snapshot_dir).absolute()
        manifest = json.loads((path / "manifest.json").read_bytes())
        if manifest["schema"] != _VERSION or set(manifest["artifacts"]) != set(
            _ARTIFACTS
        ):
            raise ValueError("Unsupported or incomplete snapshot manifest.")
        for name in _ARTIFACTS:
            expected = manifest["artifacts"][name]
            actual = _file_digest(path / name)
            if expected != actual:
                _fail(
                    report,
                    "snapshot_digest_mismatch",
                    "Snapshot artifact digest mismatch.",
                    path=os.fspath(path / name),
                    expected=expected,
                    actual=actual,
                )
        signature = _decode_schema(json.loads((path / "schema.json").read_bytes()))
        report = json.loads((path / "report.json").read_bytes())
        _check_report(report)
        if report["snapshot"] != {
            name: manifest["artifacts"][name]
            for name in ("records.tfrecord", "schema.json")
        }:
            raise ValueError("Report snapshot digests do not match the manifest.")
        dataset = _snapshot_dataset(path, signature, report["retained_count"])
        _verify_rows(dataset, report)
        return AdmittedInventory(dataset, report, path)
    except InventoryLoadError:
        raise
    except Exception as exc:
        _fail(
            report if isinstance(report, dict) else {},
            "invalid_snapshot",
            str(exc),
            cause=exc,
            path=os.fspath(path) if path is not None else None,
        )


def load_inventory(
    admitted: AdmittedInventory,
    dataset_type: str,
    batch_size: int,
    seed: int,
    **pipeline_options,
):
    """Prepare admitted data using the same pipeline controls as ``load_ds``.

    Source arguments and filters are not accepted. Selection must be declared at
    admission. Return values match ``load_ds``, including ``return_raw_ds=True``
    and its addressable ``finalize_epoch``. The admission report counts source
    records, independently of training batching, padding, and view expansion.
    """
    if not isinstance(admitted, AdmittedInventory):
        raise TypeError("admitted must be an AdmittedInventory.")
    for name in ("filter_fn", "source_filter_fn"):
        if name in pipeline_options:
            raise ValueError(
                f"{name} is not supported; declare an InventoryFilter at admission."
            )
    cache_identity = hashlib.sha256(
        (admitted.path / "manifest.json").read_bytes()
    ).hexdigest()
    return _prepare_ds(
        lambda: admitted.dataset,
        dataset_type=dataset_type,
        batch_size=batch_size,
        seed=seed,
        _cache_input_identity=cache_identity,
        **pipeline_options,
    )

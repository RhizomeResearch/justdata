from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from collections.abc import Mapping
from numbers import Integral
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import tensorflow as tf


def python_value(value: Any) -> Any:
    if hasattr(value, "numpy"):
        value = value.numpy()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "tolist"):
        return python_value(value.tolist())
    if isinstance(value, (list, tuple)):
        return [python_value(child) for child in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def get_metadata_value(sample: dict, key: str) -> Any | None:
    metadata = sample.get("metadata")
    if isinstance(metadata, dict) and key in metadata:
        return metadata[key]
    return sample.get(key)


def _is_numeric_metadata_value(value: Any) -> bool:
    return hasattr(value, "dtype") and value.dtype != tf.string


def numeric_metadata(metadata: dict) -> dict:
    numeric = {}
    for key, value in metadata.items():
        if isinstance(value, dict):
            child = numeric_metadata(value)
            if child:
                numeric[key] = child
        elif _is_numeric_metadata_value(value):
            numeric[key] = value
    return numeric


def apply_metadata_mode(
    sample: dict,
    metadata_mode: Literal["full", "numeric_only", "none"],
) -> dict:
    if metadata_mode == "full" or "metadata" not in sample:
        return sample

    result = dict(sample)
    if metadata_mode == "none":
        result.pop("metadata", None)
        return result

    metadata = result.get("metadata")
    if isinstance(metadata, dict):
        metadata = numeric_metadata(metadata)
        if metadata:
            result["metadata"] = metadata
        else:
            result.pop("metadata", None)
    return result


def _flatten_metadata(metadata: dict, prefix: tuple[str, ...] = ()):
    leaves = []
    for key, value in metadata.items():
        path = prefix + (key,)
        if isinstance(value, dict):
            leaves.extend(_flatten_metadata(value, path))
        else:
            leaves.append((path, value))
    return leaves


def _identity_structure(value):
    if isinstance(value, dict):
        return {key: _identity_structure(child) for key, child in value.items()}
    return tf.identity(value)


def _attach_metadata(sample: dict, fields: dict[str, tf.Tensor]) -> dict:
    """Merge numeric transport fields into metadata once they have been computed.

    Every output tensor depends on ``fields`` so a sidecar callback cannot be
    pruned even when a later stage drops the transport fields.
    """
    with tf.control_dependencies(list(fields.values())):
        result = _identity_structure(sample)
        result["metadata"] = result["metadata"] | fields
        return result


def json_value(value: Any) -> Any:
    value = python_value(value)
    if isinstance(value, Mapping):
        return {str(key): json_value(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(child) for child in value]
    return value


def stable_int64_hash(value: Any) -> int:
    """Return a process-stable non-negative int64 hash for metadata ids."""
    value = python_value(value)
    if value is None:
        value = ""
    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False) & ((1 << 63) - 1)


def _assign_nested(target: dict, path: tuple[str, ...], value: Any) -> None:
    current = target
    for key in path[:-1]:
        current = current.setdefault(key, {})
    current[path[-1]] = value


def _nested_metadata(values_by_path: dict[tuple[str, ...], Any]) -> dict:
    """Rebuild nested metadata from flattened leaf paths."""
    nested = {}
    for path, value in values_by_path.items():
        _assign_nested(nested, path, value)
    return nested


def _sidecar_example_id(values_by_path: dict[tuple[str, ...], Any]) -> int:
    by_name = {
        path[0]: value for path, value in values_by_path.items() if len(path) == 1
    }
    composite_values = [
        python_value(by_name.get(key)) for key in ("dataset", "split", "clip_id")
    ]
    if all(value is not None for value in composite_values):
        return stable_int64_hash(
            f"{composite_values[0]}::{composite_values[1]}::{composite_values[2]}"
        )

    value = python_value(by_name.get("example_id"))
    if isinstance(value, int) and not isinstance(value, bool):
        if not 0 <= value < 2**63:
            raise ValueError("Sidecar example_id must fit in a non-negative int64.")
        return value
    if isinstance(value, str):
        return stable_int64_hash(value)
    raise ValueError(
        "Sidecar metadata requires either non-null 'dataset', 'split', and "
        "'clip_id' fields or an integer/string 'example_id'."
    )


def _identity(metadata: Mapping[str, Any]) -> dict:
    if all(metadata.get(key) is not None for key in ("dataset", "split", "clip_id")):
        return {
            key: json_value(metadata[key]) for key in ("dataset", "split", "clip_id")
        }
    if "example_id" in metadata:
        return {"example_id": json_value(metadata["example_id"])}
    raise ValueError("Sidecar metadata requires an example identity.")


_path_locks: dict[Path, threading.Lock] = {}
_path_locks_guard = threading.Lock()


def _path_lock(path: Path) -> threading.Lock:
    with _path_locks_guard:
        return _path_locks.setdefault(path.resolve(), threading.Lock())


def _record_fingerprint(metadata: dict, identity: dict | None) -> np.ndarray:
    encoded = json.dumps(
        {"metadata": metadata, "identity": identity},
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).digest()
    return np.asarray(
        [
            int.from_bytes(digest[start : start + 8], "big", signed=True)
            for start in range(0, 32, 8)
        ],
        dtype=np.int64,
    )


_TRANSPORT_FIELDS = {"row_id", "row_fingerprint", "view_id", "view_fingerprint"}


class MetadataSidecar:
    def __init__(self) -> None:
        self.records: dict[int, dict] = {}
        self.identities: dict[int, dict] = {}
        self.view_records: dict[tuple[int, int], dict] = {}

    def add(
        self, example_id: int, metadata: dict, *, identity: dict | None = None
    ) -> None:
        if isinstance(example_id, bool) or not isinstance(example_id, Integral):
            raise TypeError("Sidecar key must be an integer.")
        example_id = int(example_id)
        if not 0 <= example_id < 2**63:
            raise ValueError("Sidecar key must fit in a non-negative int64.")
        if not isinstance(metadata, Mapping):
            raise TypeError("Sidecar metadata must be a mapping.")
        if identity is not None and not isinstance(identity, Mapping):
            raise TypeError("Sidecar identity must be a mapping.")
        metadata = json_value(metadata)
        identity = None if identity is None else json_value(identity)
        existing = self.records.get(example_id)
        previous_identity = self.identities.get(example_id)
        if existing is not None and previous_identity is None and identity is not None:
            raise ValueError(
                f"Sidecar ID {example_id} lacks full identity evidence; rebuild the artifact."
            )
        if existing is not None and existing != metadata:
            raise ValueError(f"Conflicting metadata for sidecar ID {example_id}.")
        if previous_identity is not None and identity != previous_identity:
            raise ValueError(f"Conflicting identity for sidecar ID {example_id}.")
        self.records[example_id] = metadata
        if identity is not None:
            self.identities[example_id] = identity

    def add_view(self, example_id: int, view_id: int, metadata: dict) -> None:
        """Register dynamic metadata under a source and view pair."""
        if not all(
            isinstance(value, Integral) and not isinstance(value, bool)
            for value in (example_id, view_id)
        ):
            raise TypeError("Sidecar source and view keys must be integers.")
        key = (int(example_id), int(view_id))
        if not 0 <= key[0] < 2**63 or not 0 < key[1] < 2**63:
            raise ValueError("Sidecar source and view keys must fit in int64.")
        if not isinstance(metadata, Mapping):
            raise TypeError("View metadata must be a mapping.")
        metadata = json_value(metadata)
        existing = self.view_records.get(key)
        if existing is not None and existing != metadata:
            raise ValueError(f"Conflicting view metadata for sidecar ID {key}.")
        self.view_records[key] = metadata

    @staticmethod
    def view_key(metadata: Mapping[str, Any]) -> int:
        """Derive a stable numeric key from complete dynamic view metadata."""
        encoded = json.dumps(
            json_value(metadata),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        )
        return stable_int64_hash(encoded) or 1

    def add_view_metadata(self, example_id: int, metadata: dict) -> int:
        view_id = self.view_key(metadata)
        self.add_view(example_id, view_id, metadata)
        return view_id

    @classmethod
    def from_metadata(cls, records: Iterable[Mapping[str, Any]]) -> "MetadataSidecar":
        """Build a complete source mapping before an iterator is consumed."""
        sidecar = cls()
        for record in records:
            metadata = json_value(record)
            if _TRANSPORT_FIELDS & metadata.keys():
                raise ValueError("Sidecar transport field names are reserved.")
            key = _sidecar_example_id(
                {(name,): value for name, value in metadata.items()}
            )
            sidecar.add(key, metadata, identity=_identity(metadata))
        return sidecar

    @classmethod
    def from_inventory(cls, admitted: Any) -> "MetadataSidecar":
        """Bind admitted source records to complete IDs and verified assets."""
        from justdata.core.inventory import AdmittedInventory

        if not isinstance(admitted, AdmittedInventory):
            raise TypeError("admitted must be an AdmittedInventory.")
        report = {
            record["record_id"]: record
            for record in admitted.report["records"]
            if record["disposition"] == "retained"
        }
        sidecar = cls()
        for sample in admitted.dataset.as_numpy_iterator():
            metadata = json_value(sample["metadata"])
            if _TRANSPORT_FIELDS & metadata.keys():
                raise ValueError("Sidecar transport field names are reserved.")
            if "source_record" in metadata:
                raise ValueError("source_record is reserved for inventory provenance.")
            record_id = metadata["example_id"]
            record = report[record_id]
            key = _sidecar_example_id({("example_id",): record_id})
            sidecar.add(
                key,
                metadata
                | {
                    "source_record": {
                        "source": record["source"],
                        "split": record["split"],
                        "record_id": record_id,
                        "assets": record["assets"],
                        "metadata": record["metadata"],
                    }
                },
                identity={"example_id": record_id},
            )
        if len(sidecar.records) != admitted.report["retained_count"]:
            raise ValueError(
                "Sidecar inventory count differs from the admitted snapshot."
            )
        return sidecar

    def write_jsonl(
        self, path: str, *, policy: Literal["resume", "create", "overwrite"] = "resume"
    ) -> None:
        """Persist a validated snapshot without replacing accepted records on failure."""
        if policy not in {"resume", "create", "overwrite"}:
            raise ValueError("Unknown sidecar metadata policy.")
        output = Path(path)
        if output.parent != Path("."):
            output.parent.mkdir(parents=True, exist_ok=True)
        with _path_lock(output):
            if output.exists():
                if policy == "create":
                    raise FileExistsError(f"Sidecar already exists: {output}")
                merged = (
                    self.read_jsonl(str(output)) if policy == "resume" else type(self)()
                )
            else:
                merged = type(self)()
            previous_records = merged.records.copy()
            previous_identities = merged.identities.copy()
            previous_views = merged.view_records.copy()
            for example_id, metadata in self.records.items():
                merged.add(
                    example_id, metadata, identity=self.identities.get(example_id)
                )
            for (example_id, view_id), metadata in self.view_records.items():
                if example_id not in merged.records:
                    raise ValueError(
                        f"Unknown source ID {example_id} for sidecar view."
                    )
                merged.add_view(example_id, view_id, metadata)
            if (
                output.exists()
                and policy == "resume"
                and merged.records == previous_records
                and merged.identities == previous_identities
                and merged.view_records == previous_views
            ):
                return
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", dir=output.parent, delete=False
                ) as handle:
                    temporary = Path(handle.name)
                    for example_id in sorted(merged.records):
                        entry = {
                            "example_id": example_id,
                            "metadata": merged.records[example_id],
                        }
                        if example_id in merged.identities:
                            entry["identity"] = merged.identities[example_id]
                        handle.write(
                            json.dumps(entry, sort_keys=True, allow_nan=False) + "\n"
                        )
                    for (example_id, view_id), metadata in sorted(
                        merged.view_records.items()
                    ):
                        handle.write(
                            json.dumps(
                                {
                                    "example_id": example_id,
                                    "view_id": view_id,
                                    "metadata": metadata,
                                },
                                sort_keys=True,
                                allow_nan=False,
                            )
                            + "\n"
                        )
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, output)
                if hasattr(os, "O_DIRECTORY"):
                    directory_fd = os.open(output.parent, os.O_DIRECTORY)
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)

    @classmethod
    def read_jsonl(cls, path: str) -> "MetadataSidecar":
        sidecar = cls()
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if "view_id" in record:
                    sidecar.add_view(
                        record["example_id"], record["view_id"], record["metadata"]
                    )
                else:
                    sidecar.add(
                        record["example_id"],
                        record.get("metadata", {}),
                        identity=record.get("identity"),
                    )
        if any(key[0] not in sidecar.records for key in sidecar.view_records):
            raise ValueError("Sidecar view refers to an unknown source ID.")
        return sidecar

    def digest(self) -> str:
        entries = {
            "sources": [
                (key, self.identities.get(key), self.records[key])
                for key in sorted(self.records)
            ],
            "views": [
                (source, view, self.view_records[(source, view)])
                for source, view in sorted(self.view_records)
            ],
        }
        return hashlib.sha256(
            json.dumps(
                entries, sort_keys=True, allow_nan=False, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()


def attach_sidecar_writer(
    ds: tf.data.Dataset,
    path: str,
    *,
    policy: Literal["resume", "create", "overwrite"] = "resume",
    map_parallel_calls: int = tf.data.AUTOTUNE,
) -> tf.data.Dataset:
    output = Path(path)
    if policy not in {"resume", "create", "overwrite"}:
        raise ValueError("Unknown sidecar metadata policy.")
    if policy == "create" and output.exists():
        raise FileExistsError(f"Sidecar already exists: {output}")
    if policy == "overwrite":
        MetadataSidecar().write_jsonl(path, policy="overwrite")
    elif not output.exists():
        MetadataSidecar().write_jsonl(path, policy="create")

    def add_writer(sample):
        metadata = sample.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError("Sidecar transport requires sample metadata.")
        if _TRANSPORT_FIELDS & metadata.keys():
            raise ValueError("Sidecar transport field names are reserved.")

        leaves = _flatten_metadata(metadata)
        all_paths = [leaf_path for leaf_path, _ in leaves]
        all_values = [value for _, value in leaves]

        def write_record(*values):
            values_by_path = {
                leaf_path: python_value(value)
                for leaf_path, value in zip(all_paths, values)
            }
            source_metadata = _nested_metadata(values_by_path)
            example_id = _sidecar_example_id(values_by_path)
            pending = MetadataSidecar()
            pending.add(
                example_id, source_metadata, identity=_identity(source_metadata)
            )
            pending.write_jsonl(str(output), policy="resume")
            return example_id, _record_fingerprint(
                source_metadata, pending.identities[example_id]
            )

        marker, fingerprint = tf.py_function(
            write_record, all_values, Tout=(tf.int64, tf.int64)
        )
        marker.set_shape([])
        fingerprint.set_shape([4])
        return _attach_metadata(
            sample, {"row_id": marker, "row_fingerprint": fingerprint}
        )

    return ds.map(add_writer, num_parallel_calls=map_parallel_calls)


def attach_sidecar_mapping(
    ds: tf.data.Dataset,
    sidecar: MetadataSidecar,
    *,
    map_parallel_calls: int = tf.data.AUTOTUNE,
) -> tf.data.Dataset:
    """Attach verified numeric keys using a detached, complete source mapping."""
    records = {key: json_value(value) for key, value in sidecar.records.items()}
    identities = {key: json_value(value) for key, value in sidecar.identities.items()}
    if len(records) != len(identities):
        raise ValueError(
            "Immutable sidecar needs full identity evidence for every row."
        )

    def bind(sample):
        metadata = sample.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError("Sidecar transport requires sample metadata.")
        if _TRANSPORT_FIELDS & metadata.keys():
            raise ValueError("Sidecar transport field names are reserved.")
        leaves = _flatten_metadata(metadata)
        paths = [path for path, _ in leaves]

        def lookup(*values):
            by_path = {path: python_value(value) for path, value in zip(paths, values)}
            key = _sidecar_example_id(by_path)
            supplied = _nested_metadata(by_path)
            if key not in records:
                raise ValueError(f"Unknown sidecar ID {key}.")
            if _identity(supplied) != identities[key]:
                raise ValueError(f"Conflicting identity for sidecar ID {key}.")
            for name, value in json_value(supplied).items():
                if records[key].get(name) != value:
                    raise ValueError(f"Conflicting metadata for sidecar ID {key}.")
            return key, _record_fingerprint(records[key], identities[key])

        row_id, fingerprint = tf.py_function(
            lookup, [value for _, value in leaves], Tout=(tf.int64, tf.int64)
        )
        row_id.set_shape([])
        fingerprint.set_shape([4])
        return _attach_metadata(
            sample, {"row_id": row_id, "row_fingerprint": fingerprint}
        )

    return ds.map(bind, num_parallel_calls=map_parallel_calls)


def _contains_string(value: Any) -> bool:
    if isinstance(value, str):
        return True
    if isinstance(value, Mapping):
        return any(_contains_string(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_string(child) for child in value)
    return False


def _require_one_sidecar_source(
    sidecar: MetadataSidecar | None, path: str | None
) -> None:
    if (sidecar is None) == (path is None):
        raise ValueError(
            "Choose either a sidecar path or an immutable metadata sidecar."
        )


def attach_sidecar_views(
    ds: tf.data.Dataset,
    *,
    sidecar: MetadataSidecar | None = None,
    path: str | None = None,
    map_parallel_calls: int = tf.data.AUTOTUNE,
) -> tf.data.Dataset:
    """Persist string view metadata under a distinct numeric view key."""
    _require_one_sidecar_source(sidecar, path)
    known = (
        {}
        if sidecar is None
        else {key: json_value(value) for key, value in sidecar.records.items()}
    )
    known_views = (
        {}
        if sidecar is None
        else {key: json_value(value) for key, value in sidecar.view_records.items()}
    )

    def bind(sample):
        metadata = sample.get("metadata")
        if not isinstance(metadata, dict) or "row_id" not in metadata:
            raise ValueError("Sidecar views require metadata.row_id.")
        if {"view_id", "view_fingerprint"} & metadata.keys():
            raise ValueError("view_id and view_fingerprint are reserved.")
        leaves = _flatten_metadata(metadata)
        paths = [leaf_path for leaf_path, _ in leaves]

        def register(*values):
            full = _nested_metadata(
                {
                    leaf_path: python_value(value)
                    for leaf_path, value in zip(paths, values)
                }
            )
            source_id = int(full["row_id"])
            source_records = (
                known if path is None else MetadataSidecar.read_jsonl(path).records
            )
            if source_id not in source_records:
                raise ValueError(f"Unknown sidecar ID {source_id}.")
            source = source_records[source_id]
            dynamic = {
                name: value
                for name, value in full.items()
                if name not in {"row_id", "row_fingerprint"}
                and (name not in source or source[name] != value)
            }
            if not _contains_string(dynamic):
                return 0, np.zeros(4, dtype=np.int64)
            view_id = MetadataSidecar.view_key(dynamic)
            if path is not None:
                pending = MetadataSidecar()
                pending.add_view(source_id, view_id, dynamic)
                pending.write_jsonl(path)
            elif known_views.get((source_id, view_id)) != dynamic:
                raise ValueError(
                    f"Unknown or conflicting sidecar view {(source_id, view_id)}."
                )
            return view_id, _record_fingerprint(
                dynamic, {"source_id": source_id, "view_id": view_id}
            )

        view_id, fingerprint = tf.py_function(
            register, [value for _, value in leaves], Tout=(tf.int64, tf.int64)
        )
        view_id.set_shape([])
        fingerprint.set_shape([4])
        return _attach_metadata(
            sample, {"view_id": view_id, "view_fingerprint": fingerprint}
        )

    return ds.map(bind, num_parallel_calls=map_parallel_calls)


def _sidecar_fingerprints(
    sidecar: MetadataSidecar,
) -> tuple[dict[int, np.ndarray], dict[tuple[int, int], np.ndarray]]:
    """Fingerprint every source row and view of a sidecar mapping."""
    rows = {
        key: _record_fingerprint(value, sidecar.identities.get(key))
        for key, value in sidecar.records.items()
    }
    views = {
        key: _record_fingerprint(value, {"source_id": key[0], "view_id": key[1]})
        for key, value in sidecar.view_records.items()
    }
    return rows, views


def verify_sidecar_keys(
    ds: tf.data.Dataset,
    *,
    sidecar: MetadataSidecar | None = None,
    path: str | None = None,
    map_parallel_calls: int = tf.data.AUTOTUNE,
) -> tf.data.Dataset:
    """Check joins after caches that can skip the source mapping callback."""
    _require_one_sidecar_source(sidecar, path)
    known = ({}, {}) if sidecar is None else _sidecar_fingerprints(sidecar)

    def verify(sample):
        metadata = sample.get("metadata")
        if (
            not isinstance(metadata, dict)
            or not {
                "row_id",
                "row_fingerprint",
            }
            <= metadata.keys()
        ):
            raise ValueError("Sidecar-backed samples need row_id and row_fingerprint.")

        def check(value, fingerprint, view_value, view_fingerprint):
            key = int(python_value(value))
            available, available_views = (
                known
                if path is None
                else _sidecar_fingerprints(MetadataSidecar.read_jsonl(path))
            )
            if key not in available:
                raise ValueError(f"Unknown sidecar ID {key}; rebuild the mapping.")
            if not np.array_equal(fingerprint, available[key]):
                raise ValueError(f"Conflicting sidecar mapping for ID {key}.")
            view_id = int(python_value(view_value))
            if view_id:
                view_key = (key, view_id)
                if view_key not in available_views:
                    raise ValueError(f"Unknown sidecar view {view_key}.")
                if not np.array_equal(view_fingerprint, available_views[view_key]):
                    raise ValueError(f"Conflicting sidecar view {view_key}.")
            elif not np.array_equal(view_fingerprint, np.zeros(4, dtype=np.int64)):
                raise ValueError("Invalid empty sidecar view fingerprint.")
            return key

        marker = tf.py_function(
            check,
            [
                metadata["row_id"],
                metadata["row_fingerprint"],
                metadata["view_id"],
                metadata["view_fingerprint"],
            ],
            Tout=tf.int64,
        )
        marker.set_shape([])
        return _attach_metadata(sample, {"row_id": marker})

    return ds.map(verify, num_parallel_calls=map_parallel_calls)


__all__ = [
    "apply_metadata_mode",
    "attach_sidecar_mapping",
    "attach_sidecar_views",
    "verify_sidecar_keys",
    "attach_sidecar_writer",
    "get_metadata_value",
    "json_value",
    "MetadataSidecar",
    "numeric_metadata",
    "python_value",
    "stable_int64_hash",
]

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Literal

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


def _stable_int64_hash(value: Any) -> int:
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


def _sidecar_example_id(values_by_path: dict[tuple[str, ...], Any]) -> int:
    by_name = {path[-1]: value for path, value in values_by_path.items()}
    if {"dataset", "split", "clip_id"}.issubset(by_name):
        return _stable_int64_hash(
            f"{python_value(by_name['dataset'])}::"
            f"{python_value(by_name['split'])}::"
            f"{python_value(by_name['clip_id'])}"
        )

    example_id = by_name.get("example_id")
    value = python_value(example_id)
    if isinstance(value, int):
        return value
    return _stable_int64_hash(value)


def attach_sidecar_writer(ds: tf.data.Dataset, path: str) -> tf.data.Dataset:
    output = Path(path)
    if output.parent != Path("."):
        output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("", encoding="utf-8")
    lock = threading.Lock()

    def add_writer(sample):
        metadata = sample.get("metadata")
        if not isinstance(metadata, dict):
            return sample

        leaves = _flatten_metadata(metadata)
        string_leaves = [
            (leaf_path, value)
            for leaf_path, value in leaves
            if value.dtype == tf.string
        ]
        if not string_leaves:
            return sample

        all_paths = [leaf_path for leaf_path, _ in leaves]
        all_values = [value for _, value in leaves]
        string_paths = {leaf_path for leaf_path, _ in string_leaves}

        def write_record(*values):
            values_by_path = {
                leaf_path: python_value(value)
                for leaf_path, value in zip(all_paths, values)
            }
            string_metadata = {}
            for leaf_path in string_paths:
                _assign_nested(
                    string_metadata,
                    leaf_path,
                    values_by_path[leaf_path],
                )

            record = {
                "example_id": _sidecar_example_id(values_by_path),
                "metadata": string_metadata,
            }
            with lock:
                with output.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
            return 0

        marker = tf.py_function(write_record, all_values, Tout=tf.int64)
        with tf.control_dependencies([marker]):
            return _identity_structure(sample)

    return ds.map(add_writer, num_parallel_calls=tf.data.AUTOTUNE)


__all__ = [
    "apply_metadata_mode",
    "attach_sidecar_writer",
    "get_metadata_value",
    "numeric_metadata",
    "python_value",
]

from __future__ import annotations

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


__all__ = [
    "apply_metadata_mode",
    "get_metadata_value",
    "numeric_metadata",
    "python_value",
]

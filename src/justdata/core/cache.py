"""Bounded, identity-checked caches for deterministic pipeline stages."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

import tensorflow as tf

from justdata.core._records import (
    RECORDS_FILE,
    file_sha256,
    parse_example_fn,
    read_examples,
    serialize_example,
)
from justdata.core.config_resolution import is_positive_int
from justdata.core.executed_config import canonical_config_json


class CacheError(RuntimeError):
    """A protected cache cannot be built or safely reused."""

    def __init__(self, code: str, path: str, message: str):
        self.code = code
        self.path = path
        super().__init__(message)


class CachePolicy:
    """Opt in to finite, deterministic file caches.

    ``input_identity`` must identify the ordered source contents. Inventory
    loading supplies its verified manifest identity when this is omitted.
    """

    def __init__(
        self,
        *,
        input_identity: str | None = None,
        max_bytes: int,
        max_examples: int,
        materialization: Literal["lazy", "eager"] = "lazy",
        callbacks_are_deterministic: bool = False,
    ):
        for name, value in (("max_bytes", max_bytes), ("max_examples", max_examples)):
            if not is_positive_int(value):
                raise ValueError(f"{name} must be a positive integer")
        if input_identity is not None and (
            not isinstance(input_identity, str) or not input_identity
        ):
            raise ValueError("input_identity must be a nonempty string")
        if materialization not in {"lazy", "eager"}:
            raise ValueError("materialization must be 'lazy' or 'eager'")
        if not isinstance(callbacks_are_deterministic, bool):
            raise TypeError("callbacks_are_deterministic must be boolean")
        self.input_identity = input_identity
        self.max_bytes = max_bytes
        self.max_examples = max_examples
        self.materialization = materialization
        self.callbacks_are_deterministic = callbacks_are_deterministic
        self._paths: dict[str, Path] = {}
        self._failures: dict[str, dict[str, Any]] = {}

    def status(self, stage: str) -> dict[str, Any] | None:
        """Inspect the most recently bound path for a cache stage."""
        path = self._paths.get(stage)
        if path is None:
            return None
        status = inspect_cache(path)
        if stage in self._failures:
            return status | self._failures[stage]
        return status


def _canonical_bytes(value: dict) -> bytes:
    return canonical_config_json(value).encode("utf-8")


def inspect_cache(path: str | os.PathLike) -> dict[str, Any]:
    """Report persisted completion without opening source datasets."""
    root = Path(path)
    if not root.exists():
        return {"state": "missing", "path": os.fspath(root)}
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        result = {"state": "incomplete", "path": os.fspath(root)}
        failure_path = root / "failure.json"
        if failure_path.is_file():
            try:
                failure = json.loads(failure_path.read_text())
                result.update(
                    {
                        name: failure[name]
                        for name in ("failure_code", "reason")
                        if isinstance(failure.get(name), str)
                    }
                )
            except (OSError, ValueError, TypeError):
                result["failure_code"] = "unreadable_failure_record"
        data = root / RECORDS_FILE
        if data.is_file():
            result["bytes"] = data.stat().st_size
        return result
    try:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema") != "justdata.cache.v1":
            raise ValueError("unknown manifest schema")
        if not isinstance(manifest.get("count"), int) or manifest["count"] < 0:
            raise ValueError("invalid record count")
        if not isinstance(manifest.get("bytes"), int) or manifest["bytes"] < 0:
            raise ValueError("invalid byte count")
        data = root / RECORDS_FILE
        if (
            data.stat().st_size != manifest["bytes"]
            or file_sha256(data) != manifest["sha256"]
        ):
            raise ValueError("record checksum mismatch")
        return {
            "state": "complete",
            "path": os.fspath(root),
            "fingerprint": manifest["fingerprint"],
            "count": manifest["count"],
            "bytes": manifest["bytes"],
        }
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"state": "corrupt", "path": os.fspath(root), "reason": str(exc)}


def protected_cache(
    ds: tf.data.Dataset,
    path: str | os.PathLike,
    *,
    stage: str,
    policy: CachePolicy,
    fingerprint: str,
) -> tf.data.Dataset:
    """Cache one deterministic stage with exclusive, atomic completion."""
    root = Path(path).absolute()
    policy._paths[stage] = root
    policy._failures.pop(stage, None)
    if not root.parent.is_dir():
        raise CacheError("invalid_path", os.fspath(root), "Cache parent must exist.")
    signature = ds.element_spec
    if not all(
        isinstance(spec, tf.TensorSpec) and spec.shape.rank is not None
        for spec in tf.nest.flatten(signature)
    ):
        raise CacheError(
            "invalid_schema", os.fspath(root), "Cache needs known-rank tensor leaves."
        )
    schema = tf.nest.map_structure(
        lambda spec: {"dtype": spec.dtype.name, "shape": spec.shape.as_list()},
        signature,
    )
    schema_bytes = _canonical_bytes({"element_spec": schema})

    def check_compatible(complete: dict[str, Any]) -> None:
        if (
            complete["fingerprint"] != fingerprint
            or (root / "schema.json").read_bytes() != schema_bytes
        ):
            raise CacheError(
                "incompatible", os.fspath(root), "Cache identity or schema changed."
            )

    existing = inspect_cache(root)
    if existing["state"] != "missing":
        if existing["state"] != "complete":
            raise CacheError(
                existing["state"], os.fspath(root), "Cache is incomplete or corrupt."
            )
        check_compatible(existing)
        if (
            existing["count"] > policy.max_examples
            or existing["bytes"] > policy.max_bytes
        ):
            raise CacheError(
                "quota_exceeded", os.fspath(root), "Existing cache exceeds limits."
            )
        return read_examples(root, signature, existing["count"])

    def records():
        current = inspect_cache(root)
        if current["state"] == "complete":
            check_compatible(current)
            for serialized in tf.data.TFRecordDataset(os.fspath(root / RECORDS_FILE)):
                yield serialized.numpy()
            return
        try:
            root.mkdir()
        except FileExistsError as exc:
            error = CacheError(
                "incomplete", os.fspath(root), "Another writer owns this path."
            )
            policy._failures[stage] = {
                "failure_code": error.code,
                "reason": str(error),
            }
            raise error from exc
        count = 0
        size = 0
        try:
            (root / "schema.json").write_bytes(schema_bytes)
            with tf.io.TFRecordWriter(os.fspath(root / RECORDS_FILE)) as writer:
                for sample in ds:
                    serialized = serialize_example(sample)
                    count += 1
                    size += len(serialized) + 16
                    if count > policy.max_examples or size > policy.max_bytes:
                        raise CacheError(
                            "quota_exceeded", os.fspath(root), "Cache limit exceeded."
                        )
                    writer.write(serialized)
                    yield serialized
            data = root / RECORDS_FILE
            actual_size = data.stat().st_size
            if actual_size > policy.max_bytes:
                raise CacheError(
                    "quota_exceeded", os.fspath(root), "Cache byte limit exceeded."
                )
            manifest = {
                "schema": "justdata.cache.v1",
                "fingerprint": fingerprint,
                "count": count,
                "bytes": actual_size,
                "sha256": file_sha256(data),
            }
            pending = root / "manifest.pending"
            pending.write_bytes(_canonical_bytes(manifest))
            pending.replace(root / "manifest.json")
        except Exception as exc:
            code = (
                exc.code
                if isinstance(exc, CacheError)
                else (
                    "resource_exhausted"
                    if isinstance(exc, tf.errors.ResourceExhaustedError)
                    else "source_or_io_failed"
                )
            )
            policy._failures[stage] = {"failure_code": code, "reason": str(exc)}
            try:
                (root / "failure.json").write_bytes(
                    _canonical_bytes(policy._failures[stage])
                )
            except OSError:
                pass
            if isinstance(exc, CacheError):
                raise
            raise CacheError(
                code, os.fspath(root), "Cache construction failed."
            ) from exc

    if policy.materialization == "eager":
        for _ in records():
            pass
        return read_examples(root, signature, count=inspect_cache(root)["count"])

    encoded = tf.data.Dataset.from_generator(
        records, output_signature=tf.TensorSpec([], tf.string)
    )
    return encoded.map(
        parse_example_fn(signature), num_parallel_calls=1, deterministic=True
    )


__all__ = ["CacheError", "CachePolicy", "inspect_cache"]

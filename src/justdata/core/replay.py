"""Addressable replay of admitted inventories at committed batch boundaries."""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass, replace
from importlib.metadata import version
from typing import Any

import tensorflow as tf

from justdata.core.executed_config import ExecutedConfig
from justdata.core.config_resolution import is_positive_int
from justdata.core.inventory import (
    AdmittedInventory,
    _manifest_digest,
    load_inventory,
    open_inventory,
)
from justdata.core.registry import DataPipeline


_SCHEMA = "justdata.replay.v1"
_STATE_FIELDS = {
    "schema",
    "inventory_digest",
    "config_digest",
    "run_seed",
    "epoch",
    "view",
    "batch_size",
    "drop_remainder",
    "total_batches",
    "next_batch",
    "python_version",
    "tensorflow_version",
    "justdata_version",
}


class ReplayError(ValueError):
    """A replay request that cannot be qualified or resumed."""

    def __init__(self, status: str, reason: str, message: str):
        self.status = status
        self.reason = reason
        super().__init__(message)


@dataclass(frozen=True)
class ReplayState:
    """The next batch committed by the caller, not the iterator's read position."""

    schema: str
    inventory_digest: str
    config_digest: str
    run_seed: int
    epoch: int
    view: int
    batch_size: int
    drop_remainder: bool
    total_batches: int
    next_batch: int
    python_version: str
    tensorflow_version: str
    justdata_version: str

    def __post_init__(self):
        if self.schema != _SCHEMA:
            raise ReplayError("incompatible_replay", "schema", "Unknown replay schema.")
        for name in ("inventory_digest", "config_digest"):
            digest = getattr(self, name)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ReplayError(
                    "incompatible_replay", "state_format", f"Invalid {name}."
                )
        for name in ("run_seed", "epoch", "view", "total_batches", "next_batch"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ReplayError(
                    "invalid_replay_cursor", "state_format", f"Invalid {name}."
                )
        if (
            not is_positive_int(self.batch_size)
            or not isinstance(self.drop_remainder, bool)
            or self.next_batch > self.total_batches
        ):
            raise ReplayError(
                "invalid_replay_cursor", "state_format", "Invalid batch position."
            )
        for name in ("python_version", "tensorflow_version", "justdata_version"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ReplayError(
                    "incompatible_replay", "state_format", f"Invalid {name}."
                )

    def with_next_batch(self, index: int) -> ReplayState:
        """Return state to persist after all preceding batches are committed."""
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < self.next_batch
            or index > self.total_batches
        ):
            raise ReplayError(
                "invalid_replay_cursor",
                "batch_position",
                "Committed batch position must advance within this epoch.",
            )
        return replace(self, next_batch=index)

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in sorted(_STATE_FIELDS)}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ReplayState:
        if not isinstance(value, dict) or set(value) != _STATE_FIELDS:
            raise ReplayError(
                "incompatible_replay", "state_format", "Invalid replay state fields."
            )
        return cls(**value)

    @classmethod
    def from_json(cls, value: str) -> ReplayState:
        try:
            return cls.from_dict(json.loads(value))
        except (TypeError, ValueError) as exc:
            if isinstance(exc, ReplayError):
                raise
            raise ReplayError(
                "incompatible_replay", "state_format", "Invalid replay state JSON."
            ) from exc


@dataclass(frozen=True)
class ReplayEpoch:
    batches: Any
    remaining_batches: int
    remaining_examples: int
    config: ExecutedConfig
    state: ReplayState


def count_real_examples(batch: dict[str, Any]) -> tf.Tensor:
    """Count true rows in a batch's shared padding mask."""
    return tf.reduce_sum(tf.cast(batch["padding_mask"], tf.int64))


def _nonnegative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _epoch_seed(run_seed: int, epoch: int, view: int) -> int:
    encoded = json.dumps([_SCHEMA, run_seed, epoch, view], separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _runtime() -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "justdata_version": version("justdata"),
    }


def _unsupported(reason: str, message: str) -> None:
    raise ReplayError("unsupported_replay", reason, message)


def load_replay_epoch(
    admitted: AdmittedInventory,
    dataset_type: str,
    batch_size: int,
    seed: int,
    *,
    pipeline: DataPipeline,
    epoch: int,
    view: int = 0,
    state: ReplayState | None = None,
    callbacks_are_stateless: bool = False,
    as_numpy: bool = False,
    map_parallel_calls: int = 1,
    private_threadpool_size: int = 1,
    max_intra_op_parallelism: int = 1,
    prefetch: bool | int = 1,
    **pipeline_options,
) -> ReplayEpoch:
    """Build and resume one complete deterministic epoch from a verified snapshot.

    The caller certifies that every pipeline callback is pure, preserves rows,
    and draws randomness only from its supplied seed. Progress changes only when
    the caller saves ``state.with_next_batch(...)`` after a successful update.
    """
    if not isinstance(admitted, AdmittedInventory):
        raise TypeError("admitted must be an AdmittedInventory")
    for name, value in (("seed", seed), ("epoch", epoch), ("view", view)):
        _nonnegative_int(name, value)
    if seed >= 1 << 63:
        raise ValueError("seed must fit a signed 64-bit integer")
    if not isinstance(as_numpy, bool):
        raise TypeError("as_numpy must be boolean")
    for name, value in (
        ("map_parallel_calls", map_parallel_calls),
        ("private_threadpool_size", private_threadpool_size),
        ("max_intra_op_parallelism", max_intra_op_parallelism),
    ):
        if not is_positive_int(value):
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(callbacks_are_stateless, bool):
        raise TypeError("callbacks_are_stateless must be boolean")
    if prefetch is not False and not is_positive_int(prefetch):
        raise ValueError("Replay prefetch must be disabled or a positive integer")
    if not callbacks_are_stateless:
        _unsupported(
            "callbacks_unqualified",
            "Replay requires an explicit purity declaration for every pipeline stage.",
        )
    if not isinstance(pipeline, DataPipeline):
        _unsupported(
            "pipeline_unqualified", "Replay requires a registered DataPipeline."
        )
    try:
        resolved = pipeline.resolve_config(dataset_type == "train")
    except ValueError as exc:
        if "config_resolver" in str(exc):
            _unsupported("configuration_unavailable", str(exc))
        raise
    try:
        ExecutedConfig.from_dict({"schema_version": 1, **resolved})
    except (TypeError, ValueError) as exc:
        _unsupported("configuration_unavailable", str(exc))
    forbidden = {
        "return_raw_ds",
        "return_config",
        "deterministic",
        "as_numpy",
        "map_parallel_calls",
        "private_threadpool_size",
        "max_intra_op_parallelism",
        "filter_fn",
        "source_filter_fn",
        "prefetch",
    }.intersection(pipeline_options)
    if forbidden:
        _unsupported(
            "option_unqualified",
            f"Replay cannot override {', '.join(sorted(forbidden))}.",
        )
    if pipeline_options.get("sidecar_metadata_path") is not None:
        _unsupported(
            "streaming_sidecar", "Replay requires an immutable metadata sidecar."
        )
    if pipeline_options.get("cache_policy") is None and (
        (pipeline_options.get("cache_dataset") and pipeline_options.get("cache_path"))
        or (
            pipeline_options.get("cache_model_inputs")
            and pipeline_options.get("model_input_cache_path")
        )
    ):
        _unsupported(
            "persistent_cache", "Replay does not qualify persistent pipeline caches."
        )
    if pipeline_options.get("cache_model_inputs") and (
        dataset_type == "train" or pipeline.kwargs.get("augment_eval", False)
    ):
        _unsupported(
            "augmented_model_input_cache",
            "Replay cannot cache model inputs produced with augmentation.",
        )
    drop_remainder = pipeline_options.get("drop_remainder", False)
    if not isinstance(drop_remainder, bool):
        raise TypeError("drop_remainder must be boolean")
    if dataset_type != "train" and drop_remainder:
        _unsupported(
            "evaluation_drop", "Evaluation must retain its final partial batch."
        )

    # Reopening verifies the snapshot artifacts and ordered identities before
    # each new epoch construction, including a restart in another process.
    verified = open_inventory(admitted.path)
    raw, tools, _ = load_inventory(
        verified,
        dataset_type,
        batch_size,
        seed,
        pipeline=pipeline,
        return_raw_ds=True,
        return_config=True,
        deterministic=True,
        as_numpy=False,
        map_parallel_calls=map_parallel_calls,
        private_threadpool_size=private_threadpool_size,
        max_intra_op_parallelism=max_intra_op_parallelism,
        prefetch=False,
        **pipeline_options,
    )
    epoch_seed = _epoch_seed(seed, epoch, view)
    batches, _, config = tools["finalize_epoch"](
        raw,
        seed=epoch_seed,
        augment_is_stateless=True,
        prefetch=False,
        return_config=True,
    )
    config_data = config.to_dict()
    config_data["execution"]["limits"]["prefetch"] = prefetch or "disabled"
    config_data["execution"]["replay"] = {
        "schema": _SCHEMA,
        "epoch": epoch,
        "view": view,
        "prefetch_after_skip": prefetch or "disabled",
    }
    config = ExecutedConfig.from_dict(config_data)
    total_examples = verified.report["retained_count"]
    total_batches = (
        total_examples // batch_size
        if drop_remainder
        else (total_examples + batch_size - 1) // batch_size
    )
    expected = ReplayState(
        schema=_SCHEMA,
        inventory_digest=_manifest_digest(verified),
        config_digest=hashlib.sha256(config.to_bytes()).hexdigest(),
        run_seed=seed,
        epoch=epoch,
        view=view,
        batch_size=batch_size,
        drop_remainder=drop_remainder,
        total_batches=total_batches,
        next_batch=0,
        **_runtime(),
    )
    if state is not None:
        if not isinstance(state, ReplayState):
            raise TypeError("state must be a ReplayState")
        mismatches = [
            name
            for name in _STATE_FIELDS - {"next_batch"}
            if getattr(state, name) != getattr(expected, name)
        ]
        if mismatches:
            raise ReplayError(
                "incompatible_replay",
                "state_mismatch",
                "Replay state differs in: " + ", ".join(sorted(mismatches)),
            )
    cursor = 0 if state is None else state.next_batch
    # Assert the full epoch's batch count before skipping. The snapshot route
    # has one output row per admitted input; callback purity is caller-owned.
    batches = batches.apply(tf.data.experimental.assert_cardinality(total_batches))
    batches = batches.skip(cursor)
    if prefetch:
        batches = batches.prefetch(prefetch)
    remaining_batches = total_batches - cursor
    consumed_examples = min(cursor * batch_size, total_examples)
    remaining_examples = (
        total_batches * batch_size - cursor * batch_size
        if drop_remainder
        else total_examples - consumed_examples
    )
    return ReplayEpoch(
        batches=batches.as_numpy_iterator() if as_numpy else batches,
        remaining_batches=remaining_batches,
        remaining_examples=remaining_examples,
        config=config,
        state=expected.with_next_batch(cursor),
    )


__all__ = [
    "ReplayEpoch",
    "ReplayError",
    "ReplayState",
    "count_real_examples",
    "load_replay_epoch",
]

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import tensorflow as tf
from loguru import logger

from justdata.core.metadata import apply_metadata_mode, get_metadata_value, python_value


def make_stats_iterator(
    dataset: tf.data.Dataset,
    *,
    groupby: str | list[str] | None = None,
    batch_size: int,
    deterministic: bool = True,
    augment: bool = False,
    metadata_mode: Literal["full", "numeric_only", "none"] = "numeric_only",
):
    if augment:
        logger.warning(
            "Statistics iterator requested augment=True; no augmentation is applied."
        )
    if metadata_mode not in {"full", "numeric_only", "none"}:
        raise ValueError(
            "metadata_mode must be one of 'full', 'numeric_only', or 'none'."
        )

    options = tf.data.Options()
    options.deterministic = deterministic
    ds = dataset.with_options(options)
    if metadata_mode != "full":
        ds = ds.map(
            lambda sample: _apply_stats_metadata_mode(
                sample, metadata_mode=metadata_mode, groupby=groupby
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=deterministic,
        )
    ds = ds.batch(batch_size)
    return ds.as_numpy_iterator()


def _apply_stats_metadata_mode(
    sample: dict,
    *,
    metadata_mode: Literal["full", "numeric_only", "none"],
    groupby: str | list[str] | None,
) -> dict:
    result = apply_metadata_mode(sample, metadata_mode)
    if metadata_mode != "numeric_only" or groupby is None:
        return result

    metadata = sample.get("metadata")
    if not isinstance(metadata, dict):
        return result

    keys = (groupby,) if isinstance(groupby, str) else tuple(groupby)
    grouping_metadata = {key: metadata[key] for key in keys if key in metadata}
    if not grouping_metadata:
        return result

    result = dict(result)
    result["metadata"] = {**result.get("metadata", {}), **grouping_metadata}
    return result


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _axis_index(axis: str | int, rank: int) -> int:
    if isinstance(axis, int):
        return axis if axis >= 0 else rank + axis
    if axis == "time":
        return 0
    if axis in {"frequency", "freq"}:
        return 1 if rank > 1 else 0
    if axis == "channel":
        return rank - 1
    raise ValueError(f"Unknown feature axis: {axis!r}.")


def _observations(features: np.ndarray, axes: tuple[str | int, ...]) -> np.ndarray:
    if not axes:
        return features.reshape((1,) + features.shape)

    rank = features.ndim
    reduce_axes = sorted({_axis_index(axis, rank) for axis in axes})
    remaining_axes = [axis for axis in range(rank) if axis not in reduce_axes]
    moved = np.moveaxis(features, reduce_axes + remaining_axes, range(rank))
    reduce_size = int(
        np.prod([features.shape[axis] for axis in reduce_axes], dtype=np.int64)
    )
    remaining_shape = tuple(features.shape[axis] for axis in remaining_axes)
    return moved.reshape((reduce_size,) + remaining_shape)


def _group_value(metadata: dict, groupby: str | list[str] | None) -> Any:
    if groupby is None:
        return "all"
    if isinstance(groupby, str):
        return python_value(metadata.get(groupby))
    return tuple(python_value(metadata.get(key)) for key in groupby)


def _hashable_group_value(value: Any) -> Any:
    value = python_value(value)
    if isinstance(value, list):
        return tuple(_hashable_group_value(child) for child in value)
    return value


def _iter_group_feature_pairs(
    dataset,
    *,
    groupby: str | list[str] | None,
    feature_key: str,
):
    for sample in dataset:
        features = _to_numpy(sample[feature_key])
        metadata = sample.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        group_value = _group_value(metadata, groupby)
        if isinstance(group_value, list):
            group_value = tuple(group_value)

        batched_group = None
        if groupby is not None:
            keys = (groupby,) if isinstance(groupby, str) else tuple(groupby)
            values = [python_value(get_metadata_value(sample, key)) for key in keys]
            if values and all(hasattr(np.asarray(value), "shape") for value in values):
                arrays = [np.asarray(value) for value in values]
                if features.ndim > 0 and all(
                    array.shape[:1] == features.shape[:1] for array in arrays
                ):
                    batched_group = (
                        arrays[0] if len(arrays) == 1 else list(zip(*arrays))
                    )

        if batched_group is not None:
            for idx, value in enumerate(batched_group):
                yield _hashable_group_value(value), features[idx]
        else:
            yield group_value, features


def _partial_state(
    observations: np.ndarray,
) -> tuple[int, np.ndarray | None, np.ndarray | None]:
    count = int(observations.shape[0])
    if count == 0:
        return 0, None, None

    mean = np.mean(observations, axis=0, dtype=np.float64)
    centered = observations - mean
    np.multiply(centered, centered, out=centered)
    m2 = np.sum(centered, axis=0, dtype=np.float64)
    return count, mean, m2


def _merge_partial_state(
    state: dict[str, Any],
    count_b: int,
    mean_b: np.ndarray | None,
    m2_b: np.ndarray | None,
) -> None:
    if count_b == 0:
        return
    if state["count"] == 0:
        state["count"] = count_b
        state["mean"] = mean_b
        state["m2"] = m2_b
        return

    count_a = state["count"]
    count = count_a + count_b
    delta = mean_b - state["mean"]
    state["mean"] = state["mean"] + delta * count_b / count
    state["m2"] = state["m2"] + m2_b + delta * delta * count_a * count_b / count
    state["count"] = count


def compute_feature_stats(
    dataset,
    axes: tuple[str | int, ...] = ("time",),
    groupby: str | list[str] | None = "device",
    feature_key: str = "inputs",
) -> dict:
    states: dict[Any, dict[str, Any]] = {}
    axes = tuple(axes)

    for group_value, features in _iter_group_feature_pairs(
        dataset,
        groupby=groupby,
        feature_key=feature_key,
    ):
        observations = _observations(np.asarray(features, dtype=np.float64), axes)
        state = states.setdefault(
            group_value,
            {"count": 0, "mean": None, "m2": None},
        )
        _merge_partial_state(state, *_partial_state(observations))

    result = {}
    for group_value, state in states.items():
        count = int(state["count"])
        mean = state["mean"] if state["mean"] is not None else np.asarray(0.0)
        variance = state["m2"] / count if count else np.zeros_like(mean)
        result[group_value] = {
            "mean": tf.convert_to_tensor(mean, dtype=tf.float32),
            "std": tf.convert_to_tensor(np.sqrt(variance), dtype=tf.float32),
            "count": count,
            "axes": axes,
            "feature_key": feature_key,
        }
    return result


__all__ = [
    "compute_feature_stats",
    "make_stats_iterator",
]

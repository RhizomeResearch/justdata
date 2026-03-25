from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import tensorflow as tf

from justdata.core.metadata import get_metadata_value, python_value


def _criterion_parts(name: str) -> tuple[str, bool]:
    if name.endswith("_in"):
        return name[:-3], True
    return name, False


def _as_option_tensor(value: Any, dtype: tf.DType) -> tf.Tensor:
    if dtype == tf.string and not isinstance(value, (bytes, str)):
        value = str(value)
    return tf.convert_to_tensor(value, dtype=dtype)


def _matches(value: Any, expected: Any, *, membership: bool) -> tf.Tensor:
    value = tf.convert_to_tensor(value)
    if membership:
        options = [_as_option_tensor(item, value.dtype) for item in expected]
        if not options:
            return tf.constant(False)
        return tf.reduce_any(tf.equal(value, tf.stack(options)))
    return tf.equal(value, _as_option_tensor(expected, value.dtype))


def metadata_filter_predicate(**criteria: Any):
    def predicate(sample: dict) -> tf.Tensor:
        checks = []
        for criterion, expected in criteria.items():
            key, membership = _criterion_parts(criterion)
            value = get_metadata_value(sample, key)
            if value is None:
                checks.append(tf.constant(False))
            else:
                checks.append(_matches(value, expected, membership=membership))
        if not checks:
            return tf.constant(True)
        return tf.reduce_all(tf.stack([tf.cast(check, tf.bool) for check in checks]))

    return predicate


def filter_by_metadata(ds: tf.data.Dataset, **criteria: Any) -> tf.data.Dataset:
    return ds.filter(metadata_filter_predicate(**criteria))


def _normalize_group_keys(keys: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(keys, str):
        return (keys,)
    if not isinstance(keys, Iterable):
        raise TypeError("keys must be a string or a list of strings.")
    return tuple(keys)


def _group_value(sample: dict, keys: tuple[str, ...]) -> Any:
    values = tuple(python_value(get_metadata_value(sample, key)) for key in keys)
    return values[0] if len(values) == 1 else values


def groupby_metadata(
    ds: tf.data.Dataset,
    keys: str | list[str] | tuple[str, ...],
) -> dict[Any, tf.data.Dataset]:
    normalized_keys = _normalize_group_keys(keys)
    values = []
    seen = set()
    for sample in ds:
        value = _group_value(sample, normalized_keys)
        if value not in seen:
            seen.add(value)
            values.append(value)

    groups = {}
    for value in values:
        if len(normalized_keys) == 1:
            criteria = {normalized_keys[0]: value}
        else:
            criteria = dict(zip(normalized_keys, value))
        groups[value] = filter_by_metadata(ds, **criteria)
    return groups


__all__ = [
    "filter_by_metadata",
    "groupby_metadata",
    "metadata_filter_predicate",
]

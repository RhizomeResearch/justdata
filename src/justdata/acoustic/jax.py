from __future__ import annotations

from typing import Any

import numpy as np
import tensorflow as tf


def to_numpy(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: to_numpy(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(to_numpy(child) for child in value)
    if isinstance(value, tf.Tensor):
        return value.numpy()
    return np.asarray(value) if np.isscalar(value) else value


def as_numpy_iterator(ds: tf.data.Dataset):
    return ds.as_numpy_iterator()


__all__ = ["as_numpy_iterator", "to_numpy"]

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def _labels(values: Any) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim > 1:
        return np.argmax(values, axis=-1)
    return values.astype(np.int64)


def _predictions(values: Any) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim > 1:
        return np.argmax(values, axis=-1)
    return values.astype(np.int64)


def macro_classwise_accuracy(y_true, y_pred, num_classes: int = 10) -> float:
    true = _labels(y_true)
    pred = _predictions(y_pred)

    class_acc = []
    for class_id in range(num_classes):
        mask = true == class_id
        if np.any(mask):
            class_acc.append(float(np.mean(pred[mask] == class_id)))
    if not class_acc:
        return 0.0
    return float(np.mean(class_acc))


def _group_accuracy(y_true, y_pred, groups: Sequence[Any]) -> dict[Any, float]:
    true = _labels(y_true)
    pred = _predictions(y_pred)
    groups = np.asarray(groups)
    result = {}
    for group in sorted(set(groups.tolist()), key=str):
        mask = groups == group
        result[group] = float(np.mean(pred[mask] == true[mask]))
    return result


def per_device_accuracy(y_true, y_pred, devices) -> dict[Any, float]:
    return _group_accuracy(y_true, y_pred, devices)


def per_scene_accuracy(y_true, y_pred, scenes) -> dict[Any, float]:
    return _group_accuracy(y_true, y_pred, scenes)


def confusion_matrix(y_true, y_pred, num_classes: int = 10) -> np.ndarray:
    true = _labels(y_true)
    pred = _predictions(y_pred)
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, prediction in zip(true, pred):
        if 0 <= target < num_classes and 0 <= prediction < num_classes:
            matrix[int(target), int(prediction)] += 1
    return matrix


__all__ = [
    "confusion_matrix",
    "macro_classwise_accuracy",
    "per_device_accuracy",
    "per_scene_accuracy",
]

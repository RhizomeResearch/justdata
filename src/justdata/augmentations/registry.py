import threading
from typing import Dict, Protocol

import tensorflow as tf


class CropStrategyFn(Protocol):
    def __call__(self, image: tf.Tensor, *, seed: tf.Tensor, **kwargs) -> tf.Tensor: ...


class AugmentStrategyFn(Protocol):
    def __call__(self, image: tf.Tensor, *, seed: tf.Tensor, **kwargs) -> tf.Tensor: ...


_CROP_STRATEGIES: Dict[str, CropStrategyFn] = {}
_AUGMENT_STRATEGIES: Dict[str, AugmentStrategyFn] = {}
_REGISTRY_LOCK = threading.Lock()


def register_crop_strategy(name: str):
    def decorator(fn: CropStrategyFn) -> CropStrategyFn:
        with _REGISTRY_LOCK:
            if name in _CROP_STRATEGIES:
                raise ValueError(
                    f"Crop strategy '{name}' already registered by "
                    f"{_CROP_STRATEGIES[name].__module__}.{_CROP_STRATEGIES[name].__qualname__}"
                )
            _CROP_STRATEGIES[name] = fn
        return fn

    return decorator


def register_augment_strategy(name: str):
    def decorator(fn: AugmentStrategyFn) -> AugmentStrategyFn:
        with _REGISTRY_LOCK:
            if name in _AUGMENT_STRATEGIES:
                raise ValueError(
                    f"Augment strategy '{name}' already registered by "
                    f"{_AUGMENT_STRATEGIES[name].__module__}.{_AUGMENT_STRATEGIES[name].__qualname__}"
                )
            _AUGMENT_STRATEGIES[name] = fn
        return fn

    return decorator


def get_crop_strategy(name: str) -> CropStrategyFn:
    if name not in _CROP_STRATEGIES:
        raise ValueError(
            f"Crop strategy '{name}' not found. "
            f"Available: {list(_CROP_STRATEGIES.keys())}"
        )
    return _CROP_STRATEGIES[name]


def get_augment_strategy(name: str) -> AugmentStrategyFn:
    if name not in _AUGMENT_STRATEGIES:
        raise ValueError(
            f"Augment strategy '{name}' not found. "
            f"Available: {list(_AUGMENT_STRATEGIES.keys())}"
        )
    return _AUGMENT_STRATEGIES[name]

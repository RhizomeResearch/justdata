import threading
from typing import Dict, Protocol

import tensorflow as tf

from justdata.core.augmentations import AugmentationMetadata, attach_augmentation_metadata


class CropStrategyFn(Protocol):
    def __call__(self, image: tf.Tensor, *, seed: tf.Tensor, **kwargs) -> tf.Tensor: ...


class AugmentStrategyFn(Protocol):
    def __call__(self, image: tf.Tensor, *, seed: tf.Tensor, **kwargs) -> tf.Tensor: ...


_CROP_STRATEGIES: Dict[str, CropStrategyFn] = {}
_AUGMENT_STRATEGIES: Dict[str, AugmentStrategyFn] = {}
_CROP_STRATEGY_METADATA: Dict[str, AugmentationMetadata] = {}
_AUGMENT_STRATEGY_METADATA: Dict[str, AugmentationMetadata] = {}
_REGISTRY_LOCK = threading.Lock()


def register_crop_strategy(
    name: str,
    *,
    is_training_only: bool = True,
    requires_labels: bool = False,
):
    metadata = AugmentationMetadata(
        name=name,
        domain="image",
        is_training_only=is_training_only,
        requires_labels=requires_labels,
    )

    def decorator(fn: CropStrategyFn) -> CropStrategyFn:
        with _REGISTRY_LOCK:
            if name in _CROP_STRATEGIES:
                raise ValueError(
                    f"Crop strategy '{name}' already registered by "
                    f"{_CROP_STRATEGIES[name].__module__}.{_CROP_STRATEGIES[name].__qualname__}"
                )
            _CROP_STRATEGIES[name] = fn
            _CROP_STRATEGY_METADATA[name] = metadata
        attach_augmentation_metadata(fn, metadata)
        return fn

    return decorator


def register_augment_strategy(
    name: str,
    *,
    is_training_only: bool = True,
    requires_labels: bool = False,
):
    metadata = AugmentationMetadata(
        name=name,
        domain="image",
        is_training_only=is_training_only,
        requires_labels=requires_labels,
    )

    def decorator(fn: AugmentStrategyFn) -> AugmentStrategyFn:
        with _REGISTRY_LOCK:
            if name in _AUGMENT_STRATEGIES:
                raise ValueError(
                    f"Augment strategy '{name}' already registered by "
                    f"{_AUGMENT_STRATEGIES[name].__module__}.{_AUGMENT_STRATEGIES[name].__qualname__}"
                )
            _AUGMENT_STRATEGIES[name] = fn
            _AUGMENT_STRATEGY_METADATA[name] = metadata
        attach_augmentation_metadata(fn, metadata)
        return fn

    return decorator


def get_crop_strategy(name: str) -> CropStrategyFn:
    if name not in _CROP_STRATEGIES:
        raise ValueError(
            f"Crop strategy '{name}' not found. "
            f"Available: {list(_CROP_STRATEGIES.keys())}"
        )
    return attach_augmentation_metadata(
        _CROP_STRATEGIES[name],
        _CROP_STRATEGY_METADATA[name],
    )


def get_augment_strategy(name: str) -> AugmentStrategyFn:
    if name not in _AUGMENT_STRATEGIES:
        raise ValueError(
            f"Augment strategy '{name}' not found. "
            f"Available: {list(_AUGMENT_STRATEGIES.keys())}"
        )
    return attach_augmentation_metadata(
        _AUGMENT_STRATEGIES[name],
        _AUGMENT_STRATEGY_METADATA[name],
    )


def list_crop_strategies() -> tuple[str, ...]:
    return tuple(sorted(_CROP_STRATEGIES))


def list_augment_strategies() -> tuple[str, ...]:
    return tuple(sorted(_AUGMENT_STRATEGIES))


def get_crop_strategy_metadata(name: str) -> AugmentationMetadata:
    if name not in _CROP_STRATEGY_METADATA:
        raise ValueError(
            f"Crop strategy metadata '{name}' not found. "
            f"Available: {list(_CROP_STRATEGY_METADATA.keys())}"
        )
    return _CROP_STRATEGY_METADATA[name]


def get_augment_strategy_metadata(name: str) -> AugmentationMetadata:
    if name not in _AUGMENT_STRATEGY_METADATA:
        raise ValueError(
            f"Augment strategy metadata '{name}' not found. "
            f"Available: {list(_AUGMENT_STRATEGY_METADATA.keys())}"
        )
    return _AUGMENT_STRATEGY_METADATA[name]


def list_crop_strategy_metadata() -> tuple[AugmentationMetadata, ...]:
    return tuple(_CROP_STRATEGY_METADATA[name] for name in sorted(_CROP_STRATEGY_METADATA))


def list_augment_strategy_metadata() -> tuple[AugmentationMetadata, ...]:
    return tuple(
        _AUGMENT_STRATEGY_METADATA[name] for name in sorted(_AUGMENT_STRATEGY_METADATA)
    )

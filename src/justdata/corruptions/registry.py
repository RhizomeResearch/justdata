from typing import Callable, Dict

import tensorflow as tf

_CORRUPTION_REGISTRY: Dict[str, Callable] = {}


def register_corruption(name: str):
    def decorator(fn):
        _CORRUPTION_REGISTRY[name] = fn
        return fn

    return decorator


def apply_minic_corruption(
    image: tf.Tensor, corruption: str, severity: int, seed: tf.Tensor
) -> tf.Tensor:
    """
    Apply a specific Mini-C corruption to an image.

    Args:
        image: Input image tensor (uint8 or float).
        corruption: One of ["noise", "blur", "weather", "digital"].
        severity: Severity level [1-5].
        seed: Random seed.

    Returns:
        Corrupted image as uint8 tensor.
    """
    if corruption not in _CORRUPTION_REGISTRY:
        raise ValueError(
            f"Unknown corruption '{corruption}'. "
            f"Available: {list(_CORRUPTION_REGISTRY.keys())}"
        )

    # Ensure input is float for processing
    if image.dtype != tf.float32:
        image = tf.cast(image, tf.float32)

    fn = _CORRUPTION_REGISTRY[corruption]
    corrupted = fn(image, severity, seed)

    # Ensure output is uint8 [0, 255] for consistency with standard pipeline
    corrupted = tf.clip_by_value(corrupted, 0.0, 255.0)
    return tf.cast(corrupted, tf.uint8)


def _get_severity_index(severity: int) -> int:
    """Clamps severity to 1-5 and returns 0-based index."""
    return tf.clip_by_value(severity, 1, 5) - 1

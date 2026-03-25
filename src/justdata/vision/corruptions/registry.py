import threading
from typing import Any, Callable, Dict

import tensorflow as tf

_CORRUPTION_REGISTRY: Dict[str, Callable] = {}
_CORRUPTION_LOCK = threading.Lock()


def register_corruption(name: str):
    def decorator(fn):
        with _CORRUPTION_LOCK:
            if name in _CORRUPTION_REGISTRY:
                raise ValueError(
                    f"Corruption '{name}' already registered by "
                    f"{_CORRUPTION_REGISTRY[name].__module__}.{_CORRUPTION_REGISTRY[name].__qualname__}"
                )
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
    _validate_severity(severity)
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


def list_corruptions() -> tuple[str, ...]:
    return tuple(sorted(_CORRUPTION_REGISTRY))


def _get_severity_index(severity: int) -> int:
    """Validate severity 1-5 and return a 0-based Tensor index."""
    return _validate_severity(severity) - 1


def _validate_severity(severity: int | tf.Tensor) -> tf.Tensor:
    if not tf.is_tensor(severity):
        value = int(severity)
        if value < 1 or value > 5:
            raise ValueError("Mini-C corruption severity must be in the range 1..5.")

    severity_tensor = tf.cast(tf.convert_to_tensor(severity), tf.int32)
    if tf.executing_eagerly():
        try:
            value = int(severity_tensor.numpy())
        except (TypeError, ValueError):
            value = None
        if value is not None and (value < 1 or value > 5):
            raise ValueError("Mini-C corruption severity must be in the range 1..5.")

    with tf.control_dependencies(
        [
            tf.debugging.assert_greater_equal(
                severity_tensor,
                tf.constant(1, dtype=tf.int32),
                message="Mini-C corruption severity must be in the range 1..5.",
            ),
            tf.debugging.assert_less_equal(
                severity_tensor,
                tf.constant(5, dtype=tf.int32),
                message="Mini-C corruption severity must be in the range 1..5.",
            ),
        ]
    ):
        return tf.identity(severity_tensor)


def _metadata_with_corruption(
    sample: dict[str, Any],
    *,
    name: str,
    severity: int,
    domain: str = "image",
) -> dict[str, Any]:
    result = dict(sample)
    current = result.get("metadata")
    metadata = dict(current) if isinstance(current, dict) else {}
    metadata["corruption"] = tf.constant(name)
    metadata["severity"] = tf.cast(severity, tf.int32)
    metadata["corruption_domain"] = tf.constant(domain)
    result["metadata"] = metadata
    return result

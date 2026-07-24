from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict

import tensorflow as tf

from justdata.core.metadata import stable_int64_hash

PlainValue = str | int | float | bool | None | tuple["PlainValue", ...]
SeverityParameters = tuple[
    tuple[int, tuple[tuple[str, PlainValue], ...]],
    ...,
]
CorruptionFn = Callable[[tf.Tensor, int | tf.Tensor, tf.Tensor], tf.Tensor]

_SEMANTIC_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
_CORRUPTION_REGISTRY: Dict[tuple[str, str | None], CorruptionFn] = {}
_CORRUPTION_DESCRIPTOR_REGISTRY: Dict[tuple[str, str], CorruptionDescriptor] = {}
_CORRUPTION_LOCK = threading.Lock()


def _plain_value(value: PlainValue) -> Any:
    if isinstance(value, tuple):
        return [_plain_value(child) for child in value]
    return value


def _validate_plain_value(value: PlainValue, *, field: str) -> None:
    if isinstance(value, tuple):
        for child in value:
            _validate_plain_value(child, field=field)
        return
    if value is not None and not isinstance(value, (str, int, float, bool)):
        raise TypeError(f"{field} must contain only immutable plain Python values.")


@dataclass(frozen=True, slots=True)
class CorruptionDescriptor:
    """Immutable, serializable semantics for one corruption implementation."""

    name: str
    version: str
    input_domain: str
    input_dtypes: tuple[str, ...]
    output_domain: str
    output_shape: str
    output_dtype: str
    severity_values: tuple[int, ...]
    severity_parameters: SeverityParameters
    uses_randomness: bool
    seed_contract: str
    clipping_policy: str
    rounding_policy: str
    implementation: str
    implementation_parameters: tuple[tuple[str, PlainValue], ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Corruption descriptor name must be non-empty.")
        if _SEMANTIC_VERSION.fullmatch(self.version) is None:
            raise ValueError(
                "Corruption descriptor version must use semantic X.Y.Z form."
            )
        if not self.input_domain or not self.input_dtypes:
            raise ValueError("Corruption input domain and dtypes must be declared.")
        if not self.output_domain or not self.output_shape or not self.output_dtype:
            raise ValueError(
                "Corruption output domain, shape, and dtype must be declared."
            )
        if not self.severity_values or len(set(self.severity_values)) != len(
            self.severity_values
        ):
            raise ValueError("Corruption severity values must be non-empty and unique.")
        table_levels = tuple(level for level, _params in self.severity_parameters)
        if table_levels != self.severity_values:
            raise ValueError(
                "Corruption severity parameter rows must follow severity_values."
            )
        for level, parameters in self.severity_parameters:
            names = tuple(name for name, _value in parameters)
            if len(set(names)) != len(names):
                raise ValueError(
                    f"Corruption severity {level} has duplicate parameter names."
                )
            for name, value in parameters:
                if not name:
                    raise ValueError(
                        "Corruption severity parameter names must be non-empty."
                    )
                _validate_plain_value(
                    value,
                    field=f"severity_parameters[{level}].{name}",
                )
        implementation_names = tuple(
            name for name, _value in self.implementation_parameters
        )
        if len(set(implementation_names)) != len(implementation_names):
            raise ValueError(
                "Corruption implementation parameter names must be unique."
            )
        for name, value in self.implementation_parameters:
            if not name:
                raise ValueError(
                    "Corruption implementation parameter names must be non-empty."
                )
            _validate_plain_value(
                value,
                field=f"implementation_parameters.{name}",
            )
        for field_name in (
            "seed_contract",
            "clipping_policy",
            "rounding_policy",
            "implementation",
        ):
            if not getattr(self, field_name):
                raise ValueError(f"Corruption descriptor {field_name} is required.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "input_domain": self.input_domain,
            "input_dtypes": list(self.input_dtypes),
            "output_domain": self.output_domain,
            "output_shape": self.output_shape,
            "output_dtype": self.output_dtype,
            "severity_values": list(self.severity_values),
            "severity_parameters": [
                {
                    "severity": level,
                    "parameters": {
                        name: _plain_value(value) for name, value in parameters
                    },
                }
                for level, parameters in self.severity_parameters
            ],
            "uses_randomness": self.uses_randomness,
            "seed_contract": self.seed_contract,
            "clipping_policy": self.clipping_policy,
            "rounding_policy": self.rounding_policy,
            "implementation": self.implementation,
            "implementation_parameters": {
                name: _plain_value(value)
                for name, value in self.implementation_parameters
            },
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @property
    def identity(self) -> str:
        digest = hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()
        return f"{self.name}@{self.version}:sha256:{digest}"


def register_corruption(
    name: str,
    *,
    descriptor: CorruptionDescriptor | None = None,
):
    """Register an unversioned legacy extension or an exact versioned corruption."""
    if descriptor is not None and descriptor.name != name:
        raise ValueError(
            f"Corruption descriptor name '{descriptor.name}' does not match "
            f"registration name '{name}'."
        )
    version = descriptor.version if descriptor is not None else None
    key = (name, version)

    def decorator(fn: CorruptionFn) -> CorruptionFn:
        with _CORRUPTION_LOCK:
            if key in _CORRUPTION_REGISTRY:
                registered = _CORRUPTION_REGISTRY[key]
                identity = name if version is None else f"{name}@{version}"
                raise ValueError(
                    f"Corruption '{identity}' already registered by "
                    f"{registered.__module__}.{registered.__qualname__}"
                )
            _CORRUPTION_REGISTRY[key] = fn
            if descriptor is not None:
                _CORRUPTION_DESCRIPTOR_REGISTRY[key] = descriptor
        return fn

    return decorator


def get_corruption_descriptor(name: str, version: str) -> CorruptionDescriptor:
    key = (name, version)
    with _CORRUPTION_LOCK:
        descriptor = _CORRUPTION_DESCRIPTOR_REGISTRY.get(key)
        available = tuple(sorted(_CORRUPTION_DESCRIPTOR_REGISTRY))
    if descriptor is None:
        raise ValueError(
            f"Corruption descriptor '{name}@{version}' not found. "
            f"Available: {[f'{item[0]}@{item[1]}' for item in available]}"
        )
    return descriptor


def list_corruption_descriptors() -> tuple[CorruptionDescriptor, ...]:
    with _CORRUPTION_LOCK:
        return tuple(
            _CORRUPTION_DESCRIPTOR_REGISTRY[key]
            for key in sorted(_CORRUPTION_DESCRIPTOR_REGISTRY)
        )


def list_corruption_versions(name: str) -> tuple[str, ...]:
    with _CORRUPTION_LOCK:
        return tuple(
            version
            for registered_name, version in sorted(_CORRUPTION_DESCRIPTOR_REGISTRY)
            if registered_name == name
        )


def apply_corruption(
    image: tf.Tensor,
    *,
    name: str,
    version: str,
    severity: int | tf.Tensor,
    seed: tf.Tensor,
) -> tf.Tensor:
    """Apply an exact versioned corruption using a complete stateless seed."""
    descriptor = get_corruption_descriptor(name, version)
    image = _validate_image(image, descriptor)
    severity_tensor = _validate_descriptor_severity(severity, descriptor)
    seed_tensor = _validate_seed(seed, descriptor)
    with _CORRUPTION_LOCK:
        fn = _CORRUPTION_REGISTRY[(name, version)]
    corrupted = fn(image, severity_tensor, seed_tensor)
    corrupted = _finalize_corruption(corrupted, descriptor)

    with tf.control_dependencies(
        [
            tf.debugging.assert_equal(
                tf.shape(corrupted),
                tf.shape(image),
                message=(
                    f"Corruption '{name}@{version}' must preserve the input shape."
                ),
            )
        ]
    ):
        corrupted = tf.identity(corrupted)
    corrupted.set_shape(image.shape)
    return corrupted


_MINIC_VERSIONS = {
    "blur": "1.0.0",
    "digital": "1.0.0",
    "noise": "1.0.0",
    "weather": "1.0.0",
}


def apply_minic_corruption(
    image: tf.Tensor, corruption: str, severity: int, seed: tf.Tensor
) -> tf.Tensor:
    """Apply a legacy Mini-C family without changing its version-1 semantics."""
    if corruption in _MINIC_VERSIONS:
        return apply_corruption(
            image,
            name=corruption,
            version=_MINIC_VERSIONS[corruption],
            severity=severity,
            seed=seed,
        )

    key = (corruption, None)
    with _CORRUPTION_LOCK:
        fn = _CORRUPTION_REGISTRY.get(key)
    if fn is None:
        raise ValueError(
            f"Unknown corruption '{corruption}' for the Mini-C wrapper. "
            f"Available: {sorted(_MINIC_VERSIONS)}"
        )

    severity_tensor = _validate_severity(severity)
    image = tf.cast(image, tf.float32)
    corrupted = fn(image, severity_tensor, seed)
    corrupted = tf.clip_by_value(corrupted, 0.0, 255.0)
    return tf.cast(corrupted, tf.uint8)


def list_corruptions() -> tuple[str, ...]:
    with _CORRUPTION_LOCK:
        return tuple(sorted({name for name, _version in _CORRUPTION_REGISTRY}))


def _validate_image(
    image: tf.Tensor,
    descriptor: CorruptionDescriptor,
) -> tf.Tensor:
    image = tf.convert_to_tensor(image)
    if image.dtype.name not in descriptor.input_dtypes:
        raise TypeError(
            f"Corruption '{descriptor.name}@{descriptor.version}' requires "
            f"dtype in {descriptor.input_dtypes}, got {image.dtype.name}."
        )
    if image.shape.rank is not None and image.shape.rank != 3:
        raise ValueError("Corruption input must be an HWC rank-3 tensor.")
    if image.shape.rank == 3 and image.shape[-1] not in (3, None):
        raise ValueError("Corruption input must have exactly three RGB channels.")

    with tf.control_dependencies(
        [
            tf.debugging.assert_rank(
                image,
                3,
                message="Corruption input must be an HWC rank-3 tensor.",
            ),
            tf.debugging.assert_equal(
                tf.shape(image)[-1],
                3,
                message="Corruption input must have exactly three RGB channels.",
            ),
            tf.debugging.assert_positive(
                tf.shape(image)[:2],
                message="Corruption input spatial dimensions must be positive.",
            ),
        ]
    ):
        image = tf.identity(image)
    return tf.ensure_shape(image, [None, None, 3])


def _validate_seed(
    seed: tf.Tensor,
    descriptor: CorruptionDescriptor,
) -> tf.Tensor:
    seed = tf.convert_to_tensor(seed)
    if seed.dtype != tf.int32:
        raise TypeError(
            f"Corruption '{descriptor.name}@{descriptor.version}' requires "
            "an int32 stateless seed with shape [2]."
        )
    if seed.shape.rank is not None and seed.shape.rank != 1:
        raise ValueError("Corruption seed must have shape [2].")
    if seed.shape.rank == 1 and seed.shape[0] not in (2, None):
        raise ValueError("Corruption seed must have shape [2].")
    with tf.control_dependencies(
        [
            tf.debugging.assert_equal(
                tf.shape(seed),
                tf.constant([2], dtype=tf.int32),
                message="Corruption seed must have shape [2].",
            )
        ]
    ):
        return tf.ensure_shape(tf.identity(seed), [2])


def _validate_descriptor_severity(
    severity: int | tf.Tensor,
    descriptor: CorruptionDescriptor,
) -> tf.Tensor:
    if not tf.is_tensor(severity):
        value = int(severity)
        if value not in descriptor.severity_values:
            raise ValueError(
                f"Corruption '{descriptor.name}@{descriptor.version}' severity "
                f"must be one of {descriptor.severity_values}."
            )

    severity_tensor = tf.cast(tf.convert_to_tensor(severity), tf.int32)
    if severity_tensor.shape.rank not in (0, None):
        raise ValueError("Corruption severity must be a scalar.")
    allowed = tf.constant(descriptor.severity_values, dtype=tf.int32)
    with tf.control_dependencies(
        [
            tf.debugging.assert_equal(
                tf.reduce_any(tf.equal(severity_tensor, allowed)),
                True,
                message=(
                    f"Corruption '{descriptor.name}@{descriptor.version}' severity "
                    f"must be one of {descriptor.severity_values}."
                ),
            )
        ]
    ):
        return tf.ensure_shape(tf.identity(severity_tensor), [])


def _finalize_corruption(
    corrupted: tf.Tensor,
    descriptor: CorruptionDescriptor,
) -> tf.Tensor:
    if descriptor.rounding_policy == "clip_then_truncate_toward_zero":
        corrupted = tf.clip_by_value(tf.cast(corrupted, tf.float32), 0.0, 255.0)
        return tf.cast(corrupted, tf.uint8)
    if descriptor.rounding_policy == "clip_then_round_half_to_even":
        corrupted = tf.clip_by_value(tf.cast(corrupted, tf.float32), 0.0, 255.0)
        return tf.cast(tf.math.rint(corrupted), tf.uint8)
    if descriptor.rounding_policy == "codec_native_uint8_decode":
        if corrupted.dtype != tf.uint8:
            raise TypeError("Codec-native corruption output must have dtype uint8.")
        return corrupted
    raise ValueError(
        f"Unsupported rounding policy '{descriptor.rounding_policy}' for "
        f"'{descriptor.name}@{descriptor.version}'."
    )


def _get_severity_index(severity: int | tf.Tensor) -> tf.Tensor:
    """Validate legacy Mini-C severity 1-5 and return a zero-based index."""
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
    descriptor: CorruptionDescriptor,
    severity: int,
    domain: str = "image",
) -> dict[str, Any]:
    result = dict(sample)
    current = result.get("metadata")
    metadata = dict(current) if isinstance(current, dict) else {}
    metadata["corruption"] = tf.constant(descriptor.name)
    metadata["corruption_version"] = tf.constant(descriptor.version)
    metadata["corruption_identity"] = tf.constant(descriptor.identity)
    metadata["corruption_identity_hash"] = tf.constant(
        stable_int64_hash(descriptor.identity),
        dtype=tf.int64,
    )
    metadata["severity"] = tf.cast(severity, tf.int32)
    metadata["corruption_domain"] = tf.constant(domain)
    result["metadata"] = metadata
    return result

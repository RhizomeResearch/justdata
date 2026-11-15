from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable, Dict, cast

import tensorflow as tf

from justdata.core.metadata import stable_int64_hash
from justdata.core.registry import qualified_name

PlainValue = str | int | float | bool | None | tuple["PlainValue", ...]
SeverityParameters = tuple[
    tuple[int, tuple[tuple[str, PlainValue], ...]],
    ...,
]
# Severity-only implementations are called as ``fn(image, severity, seed)``.
# Implementations of descriptors with a continuous-strength contract are called
# as ``fn(image01, parameters, *, frame_seed, episode_seed, mode, **auxiliary)``
# on float32 RGB in [0, 1], with one float32 tensor per declared parameter and
# one keyword (possibly None) per declared auxiliary input.
CorruptionFn = Callable[..., tf.Tensor]

INTERPOLATION_RULES = ("linear", "log", "nearest")
SPATIAL_SCALE_POLICIES = ("absolute_pixels", "relative_to_width")
AUXILIARY_INPUTS = ("depth_m", "horizon", "region")
EPISODE_SEED_SALT = 0x45504953
_STRENGTH_ROUNDING_POLICY = "clip_then_round_half_to_even"
_STRENGTH_FIELD_DEFAULTS: dict[str, Any] = {
    "interpolation": (),
    "spatial_scale_policy": "absolute_pixels",
    "reference_width": None,
    "spatial_parameters": (),
    "episode_draws": (),
    "frame_draws": (),
    "auxiliary_inputs": (),
    "modes": (),
    "label_effect": None,
}

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
    """Immutable, serializable semantics for one corruption implementation.

    A descriptor that declares ``identity_parameters`` also accepts continuous
    strengths in ``[0, max severity]``: strength 0 is the identity row, integer
    strengths are the severity rows, and each parameter is interpolated between
    adjacent rows by its ``interpolation`` rule. The remaining continuous-strength
    fields declare how pixel parameters follow the image width, which random
    draws come from the episode and frame seeds, which auxiliary inputs are
    accepted, which mode they select (first match wins), and the effect on
    labels. These fields are serialized only when declared, so descriptors
    without them keep their original identity.
    """

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
    identity_parameters: tuple[tuple[str, int | float], ...] = ()
    interpolation: tuple[tuple[str, str], ...] = ()
    spatial_scale_policy: str = "absolute_pixels"
    reference_width: int | None = None
    spatial_parameters: tuple[str, ...] = ()
    episode_draws: tuple[str, ...] = ()
    frame_draws: tuple[str, ...] = ()
    auxiliary_inputs: tuple[str, ...] = ()
    modes: tuple[tuple[str, tuple[str, ...]], ...] = ()
    label_effect: str | None = None

    @property
    def supports_strength(self) -> bool:
        return bool(self.identity_parameters)

    @property
    def max_strength(self) -> int:
        return max(self.severity_values)

    def strength_rows(self) -> tuple[dict[str, float], ...]:
        """The identity row followed by severity rows 1..N, as floats."""
        rows = (self.identity_parameters,) + tuple(
            parameters for _level, parameters in self.severity_parameters
        )
        return tuple(
            {name: float(cast(float, value)) for name, value in row} for row in rows
        )

    def _validate_strength_contract(self) -> None:
        key = f"{self.name}@{self.version}"
        if not self.supports_strength:
            declared = [
                field
                for field, default in _STRENGTH_FIELD_DEFAULTS.items()
                if getattr(self, field) != default
            ]
            if declared:
                raise ValueError(
                    f"Corruption '{key}' declares {declared} without "
                    "identity_parameters."
                )
            return

        if self.severity_values != tuple(range(1, len(self.severity_values) + 1)):
            raise ValueError(
                f"Corruption '{key}' severity values must be 1..N for continuous "
                "strength."
            )
        if self.rounding_policy != _STRENGTH_ROUNDING_POLICY:
            raise ValueError(
                f"Corruption '{key}' continuous strength requires rounding policy "
                f"'{_STRENGTH_ROUNDING_POLICY}'."
            )
        if not self.label_effect:
            raise ValueError(f"Corruption '{key}' must declare its label_effect.")

        names = tuple(name for name, _value in self.identity_parameters)
        if tuple(name for name, _rule in self.interpolation) != names:
            raise ValueError(
                f"Corruption '{key}' interpolation must list every identity "
                "parameter in the same order."
            )
        rows = (self.identity_parameters,) + tuple(
            parameters for _level, parameters in self.severity_parameters
        )
        for row in rows:
            if tuple(name for name, _value in row) != names:
                raise ValueError(
                    f"Corruption '{key}' severity rows must list the identity "
                    "parameters in the same order."
                )
            for name, value in row:
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                ):
                    raise ValueError(
                        f"Corruption '{key}' parameter '{name}' must be a finite "
                        "number in every row."
                    )
        for name, rule in self.interpolation:
            if rule not in INTERPOLATION_RULES:
                raise ValueError(
                    f"Corruption '{key}' parameter '{name}' interpolation must be "
                    f"one of {INTERPOLATION_RULES}."
                )
            if rule == "log" and any(row[name] <= 0 for row in self.strength_rows()):
                raise ValueError(
                    f"Corruption '{key}' parameter '{name}' uses log interpolation "
                    "and must be positive in every row."
                )

        if self.spatial_scale_policy not in SPATIAL_SCALE_POLICIES:
            raise ValueError(
                f"Corruption '{key}' spatial_scale_policy must be one of "
                f"{SPATIAL_SCALE_POLICIES}."
            )
        if self.spatial_scale_policy == "absolute_pixels":
            if self.reference_width is not None or self.spatial_parameters:
                raise ValueError(
                    f"Corruption '{key}' absolute_pixels takes no reference_width "
                    "or spatial_parameters."
                )
        elif (
            isinstance(self.reference_width, bool)
            or not isinstance(self.reference_width, int)
            or self.reference_width <= 0
        ):
            raise ValueError(
                f"Corruption '{key}' relative_to_width needs a positive integer "
                "reference_width."
            )
        unknown_spatial = set(self.spatial_parameters) - set(names)
        if unknown_spatial:
            raise ValueError(
                f"Corruption '{key}' spatial parameters {sorted(unknown_spatial)} "
                "are not declared parameters."
            )

        if len(set(self.auxiliary_inputs)) != len(self.auxiliary_inputs) or (
            set(self.auxiliary_inputs) - set(AUXILIARY_INPUTS)
        ):
            raise ValueError(
                f"Corruption '{key}' auxiliary inputs must be unique names from "
                f"{AUXILIARY_INPUTS}."
            )
        if bool(self.auxiliary_inputs) != bool(self.modes):
            raise ValueError(
                f"Corruption '{key}' declares modes exactly when it accepts "
                "auxiliary inputs."
            )
        mode_names = [mode for mode, _requires in self.modes]
        if len(set(mode_names)) != len(mode_names):
            raise ValueError(f"Corruption '{key}' mode names must be unique.")
        for mode, requires in self.modes:
            if set(requires) - set(self.auxiliary_inputs):
                raise ValueError(
                    f"Corruption '{key}' mode '{mode}' requires undeclared inputs."
                )
        if self.modes and self.modes[-1][1]:
            raise ValueError(
                f"Corruption '{key}' last mode must be the fallback that requires "
                "no auxiliary input."
            )

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
        self._validate_strength_contract()

    def to_dict(self) -> dict[str, Any]:
        result = {
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
        if self.supports_strength:
            result |= {
                "identity_parameters": dict(self.identity_parameters),
                "interpolation": dict(self.interpolation),
                "spatial_scale_policy": self.spatial_scale_policy,
                "reference_width": self.reference_width,
                "spatial_parameters": list(self.spatial_parameters),
                "episode_draws": list(self.episode_draws),
                "frame_draws": list(self.frame_draws),
                "auxiliary_inputs": list(self.auxiliary_inputs),
                "modes": [
                    {"mode": mode, "requires": list(requires)}
                    for mode, requires in self.modes
                ],
                "label_effect": self.label_effect,
            }
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @property
    def identity(self) -> str:
        digest = hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()
        return f"{self.name}@{self.version}:sha256:{digest}"


def severity_table(columns: Mapping[str, Sequence[PlainValue]]) -> SeverityParameters:
    """Build severity rows 1..N from one sequence of values per parameter."""
    lengths = {len(values) for values in columns.values()}
    if len(lengths) != 1:
        raise ValueError("Every severity column must have the same length.")
    (count,) = lengths
    return tuple(
        (
            level,
            tuple((name, values[level - 1]) for name, values in columns.items()),
        )
        for level in range(1, count + 1)
    )


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
                    f"{qualified_name(registered)}"
                )
            _CORRUPTION_REGISTRY[key] = fn
            if descriptor is not None:
                _CORRUPTION_DESCRIPTOR_REGISTRY[(name, descriptor.version)] = descriptor
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
    """Apply an exact versioned corruption using a complete stateless seed.

    Descriptors with a continuous-strength contract run their severity row with
    the episode seed derived from ``seed`` and no auxiliary input; use
    ``apply_corruption_at`` for fractional strengths, episode seeds, or maps.
    """
    descriptor = get_corruption_descriptor(name, version)
    image = _validate_image(image, descriptor)
    severity_tensor = _validate_descriptor_severity(severity, descriptor)
    seed_tensor = _validate_seed(seed, descriptor)
    fn = _registered_corruption(name, version)
    if descriptor.supports_strength:
        rendered = render_unit_image(
            fn,
            descriptor,
            unit_image(image),
            severity_row_parameters(descriptor, severity_tensor),
            frame_seed=seed_tensor,
            episode_seed=None,
            mode=select_mode(descriptor, frozenset()),
            auxiliary={},
        )
        corrupted = _finalize_corruption(rendered * 255.0, descriptor)
    else:
        corrupted = _finalize_corruption(
            fn(image, severity_tensor, seed_tensor),
            descriptor,
        )
    return _with_input_shape(corrupted, image, descriptor)


def _registered_corruption(name: str, version: str) -> CorruptionFn:
    with _CORRUPTION_LOCK:
        return _CORRUPTION_REGISTRY[(name, version)]


def _with_input_shape(
    corrupted: tf.Tensor,
    image: tf.Tensor,
    descriptor: CorruptionDescriptor,
) -> tf.Tensor:
    with tf.control_dependencies(
        [
            tf.debugging.assert_equal(
                tf.shape(corrupted),
                tf.shape(image),
                message=(
                    f"Corruption '{descriptor.name}@{descriptor.version}' must "
                    "preserve the input shape."
                ),
            )
        ]
    ):
        corrupted = tf.identity(corrupted)
    corrupted.set_shape(image.shape)
    return corrupted


def draw_seed(seed: tf.Tensor, index: int) -> tf.Tensor:
    """Seed of the ``index``-th random draw taken from ``seed``."""
    return tf.random.experimental.stateless_fold_in(seed, index, alg="philox")


def unit_image(image: tf.Tensor) -> tf.Tensor:
    """Map a validated ``[0, 255]`` image to float32 RGB in ``[0, 1]``."""
    return tf.clip_by_value(tf.cast(image, tf.float32) / 255.0, 0.0, 1.0)


def severity_row_parameters(
    descriptor: CorruptionDescriptor,
    severity: tf.Tensor,
) -> dict[str, tf.Tensor]:
    """Gather one float32 tensor per parameter from a validated severity."""
    rows = descriptor.strength_rows()[1:]
    index = tf.cast(severity, tf.int32) - 1
    return {
        name: tf.gather(
            tf.constant([row[name] for row in rows], dtype=tf.float32),
            index,
        )
        for name, _value in descriptor.identity_parameters
    }


def select_mode(
    descriptor: CorruptionDescriptor,
    provided: frozenset[str],
) -> str | None:
    """Return the first declared mode whose required inputs were all provided."""
    for mode, requires in descriptor.modes:
        if set(requires) <= provided:
            return mode
    return None


def render_unit_image(
    fn: CorruptionFn,
    descriptor: CorruptionDescriptor,
    image01: tf.Tensor,
    parameters: Mapping[str, tf.Tensor],
    *,
    frame_seed: tf.Tensor,
    episode_seed: tf.Tensor | None,
    mode: str | None,
    auxiliary: Mapping[str, Any],
) -> tf.Tensor:
    """Run a continuous-strength implementation and clip its output to [0, 1].

    Spatial parameters are scaled by ``width / reference_width`` here, so each
    implementation receives values in pixels of the image it renders.
    """
    if episode_seed is None:
        episode_seed = draw_seed(frame_seed, EPISODE_SEED_SALT)
    scaled = dict(parameters)
    if descriptor.spatial_scale_policy == "relative_to_width":
        assert descriptor.reference_width is not None
        width = tf.cast(tf.shape(image01)[1], tf.float32)
        scale = width / float(descriptor.reference_width)
        for name in descriptor.spatial_parameters:
            scaled[name] = scaled[name] * scale
    arguments = {name: auxiliary.get(name) for name in descriptor.auxiliary_inputs}
    rendered = fn(
        image01,
        scaled,
        frame_seed=frame_seed,
        episode_seed=episode_seed,
        mode=mode,
        **arguments,
    )
    return tf.clip_by_value(rendered, 0.0, 1.0)


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

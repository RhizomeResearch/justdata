"""Continuous strengths, episode seeds, auxiliary maps, and ordered stacks.

These entry points accept only descriptors that declare a continuous-strength
contract (``identity_parameters``). Strength is a Python number so that the
parameters resolved here, rounded to float32, are exactly the ones rendered and
recorded.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np
import tensorflow as tf

from justdata.core.metadata import stable_int64_hash
from justdata.vision.corruptions.registry import (
    CorruptionDescriptor,
    _finalize_corruption,
    _registered_corruption,
    _validate_image,
    _validate_seed,
    _with_input_shape,
    draw_seed,
    get_corruption_descriptor,
    render_unit_image,
    select_mode,
    unit_image,
)
from justdata.vision.horizon import Horizon

StackStep = tuple[str, str, float]


@dataclass(frozen=True, slots=True)
class CorruptionRecord:
    """What one continuous-strength corruption call resolved and ran.

    ``parameters`` are float32-rounded values in the descriptor's units. Under
    ``relative_to_width`` the descriptor's spatial parameters were multiplied by
    ``image_width / reference_width``; ``image_width`` is ``None`` when the
    width was not static. ``mode`` is the descriptor mode selected by the
    provided ``auxiliary_inputs``.
    """

    name: str
    version: str
    identity: str
    strength: float
    parameters: tuple[tuple[str, float], ...]
    spatial_scale_policy: str
    reference_width: int | None
    image_width: int | None
    mode: str | None
    auxiliary_inputs: tuple[str, ...]
    episode_seed: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "identity": self.identity,
            "strength": self.strength,
            "parameters": dict(self.parameters),
            "spatial_scale_policy": self.spatial_scale_policy,
            "reference_width": self.reference_width,
            "image_width": self.image_width,
            "mode": self.mode,
            "auxiliary_inputs": list(self.auxiliary_inputs),
            "episode_seed": self.episode_seed,
        }


@dataclass(frozen=True, slots=True)
class CorruptionResult:
    image: tf.Tensor
    record: CorruptionRecord


@dataclass(frozen=True, slots=True)
class CorruptionStackRecord:
    """Ordered step records; the identity covers each step's identity and strength."""

    steps: tuple[CorruptionRecord, ...]

    @property
    def identity(self) -> str:
        return corruption_stack_identity(
            [(step.identity, step.strength) for step in self.steps]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass(frozen=True, slots=True)
class CorruptionStackResult:
    image: tf.Tensor
    record: CorruptionStackRecord


def corruption_stack_identity(steps: Sequence[tuple[str, float]]) -> str:
    """Digest of ordered ``(descriptor identity, strength)`` pairs."""
    payload = json.dumps(
        [
            {"identity": identity, "strength": float(strength)}
            for identity, strength in steps
        ],
        separators=(",", ":"),
    )
    return f"stack:sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def resolve_parameters(name: str, version: str, strength: float) -> dict[str, float]:
    """Return the float32-rounded parameters that ``strength`` renders with.

    Strength 0 is the identity row and integer strengths are severity rows. In
    between, each parameter follows its declared rule between the two adjacent
    rows: ``linear``, ``log`` (geometric), or ``nearest`` (the lower row below
    the midpoint).
    """
    descriptor = _strength_descriptor(name, version)
    return _interpolate(descriptor, _validate_strength(strength, descriptor))


def apply_corruption_at(
    image: tf.Tensor,
    *,
    name: str,
    version: str,
    strength: float,
    seed: tf.Tensor,
    episode_seed: tf.Tensor | None = None,
    depth_m: tf.Tensor | None = None,
    horizon: Horizon | None = None,
    region: tf.Tensor | None = None,
) -> CorruptionResult:
    """Apply a corruption at a continuous strength and return its record.

    ``seed`` drives per-frame draws and ``episode_seed`` (same type and shape)
    the draws that persist across an episode; when it is omitted it is derived
    from ``seed``. ``depth_m`` (float32 HW metres), ``horizon``, and ``region``
    (bool HW) must be declared by the descriptor. Strength 0 returns the input
    after the rounding policy.
    """
    descriptor = _strength_descriptor(name, version)
    strength = _validate_strength(strength, descriptor)
    image = _validate_image(image, descriptor)
    seed_tensor = _validate_seed(seed, descriptor)
    episode_tensor = (
        None if episode_seed is None else _validate_seed(episode_seed, descriptor)
    )
    provided = _provided_auxiliary(depth_m=depth_m, horizon=horizon, region=region)
    _reject_undeclared(provided, descriptor.auxiliary_inputs, descriptor.name)
    auxiliary = _validate_auxiliary(provided, image)
    parameters = _interpolate(descriptor, strength)

    if strength == 0:
        rendered = tf.cast(image, tf.float32)
    else:
        rendered = 255.0 * _render(
            descriptor,
            unit_image(image),
            parameters,
            frame_seed=seed_tensor,
            episode_seed=episode_tensor,
            auxiliary=auxiliary,
        )
    corrupted = _with_input_shape(
        _finalize_corruption(rendered, descriptor),
        image,
        descriptor,
    )
    record = _record(
        descriptor,
        strength,
        parameters,
        image,
        provided=frozenset(auxiliary),
        episode_provided=episode_tensor is not None,
    )
    return CorruptionResult(image=corrupted, record=record)


def apply_corruption_stack(
    image: tf.Tensor,
    *,
    steps: Sequence[StackStep],
    seed: tf.Tensor,
    episode_seed: tf.Tensor | None = None,
    depth_m: tf.Tensor | None = None,
    horizon: Horizon | None = None,
    region: tf.Tensor | None = None,
) -> CorruptionStackResult:
    """Apply ordered ``(name, version, strength)`` steps in float, rounding once.

    Each step clips to ``[0, 1]`` and receives the auxiliary inputs its
    descriptor declares; every given input must be declared by some step. Step
    ``i`` draws from ``stateless_fold_in(seed, salt_i)`` and, when given,
    ``stateless_fold_in(episode_seed, salt_i)``, where ``salt_i`` is
    ``stable_int64_hash(f"{i}:{identity}") mod 2**31``.
    """
    if not steps:
        raise ValueError("A corruption stack needs at least one step.")
    descriptors = [
        _strength_descriptor(name, version) for name, version, _strength in steps
    ]
    strengths = [
        _validate_strength(strength, descriptor)
        for (_name, _version, strength), descriptor in zip(
            steps, descriptors, strict=True
        )
    ]
    image = _validate_image(image, descriptors[0])
    seed_tensor = _validate_seed(seed, descriptors[0])
    episode_tensor = (
        None if episode_seed is None else _validate_seed(episode_seed, descriptors[0])
    )
    provided = _provided_auxiliary(depth_m=depth_m, horizon=horizon, region=region)
    declared = {name for d in descriptors for name in d.auxiliary_inputs}
    _reject_undeclared(provided, tuple(sorted(declared)), "this stack")
    auxiliary = _validate_auxiliary(provided, image)

    image01 = unit_image(image)
    records = []
    for index, (descriptor, strength) in enumerate(
        zip(descriptors, strengths, strict=True)
    ):
        step_auxiliary = {
            name: value
            for name, value in auxiliary.items()
            if name in descriptor.auxiliary_inputs
        }
        parameters = _interpolate(descriptor, strength)
        if strength > 0:
            salt = stable_int64_hash(f"{index}:{descriptor.identity}") % 2**31
            image01 = _render(
                descriptor,
                image01,
                parameters,
                frame_seed=draw_seed(seed_tensor, salt),
                episode_seed=(
                    None if episode_tensor is None else draw_seed(episode_tensor, salt)
                ),
                auxiliary=step_auxiliary,
            )
        records.append(
            _record(
                descriptor,
                strength,
                parameters,
                image,
                provided=frozenset(step_auxiliary),
                episode_provided=episode_tensor is not None,
            )
        )

    rendered = 255.0 * image01 if any(strengths) else tf.cast(image, tf.float32)
    corrupted = _with_input_shape(
        _finalize_corruption(rendered, descriptors[0]),
        image,
        descriptors[0],
    )
    return CorruptionStackResult(
        image=corrupted,
        record=CorruptionStackRecord(steps=tuple(records)),
    )


def _strength_descriptor(name: str, version: str) -> CorruptionDescriptor:
    descriptor = get_corruption_descriptor(name, version)
    if not descriptor.supports_strength:
        raise ValueError(
            f"Corruption '{name}@{version}' declares only integer severities; "
            "continuous strengths and stacks need identity_parameters."
        )
    return descriptor


def _validate_strength(strength: Any, descriptor: CorruptionDescriptor) -> float:
    if tf.is_tensor(strength) or isinstance(strength, bool):
        raise TypeError("Corruption strength must be a Python number.")
    if not isinstance(strength, Real):
        raise TypeError("Corruption strength must be a Python number.")
    value = float(strength)
    if not math.isfinite(value) or not 0.0 <= value <= descriptor.max_strength:
        raise ValueError(
            f"Corruption '{descriptor.name}@{descriptor.version}' strength must be "
            f"in [0, {descriptor.max_strength}]."
        )
    return value


def _interpolate(
    descriptor: CorruptionDescriptor,
    strength: float,
) -> dict[str, float]:
    rows = descriptor.strength_rows()
    lower = math.floor(strength)
    upper = min(lower + 1, len(rows) - 1)
    fraction = strength - lower
    resolved = {}
    for name, rule in descriptor.interpolation:
        start = rows[lower][name]
        end = rows[upper][name]
        if fraction == 0:
            value = start
        elif rule == "linear":
            value = start + fraction * (end - start)
        elif rule == "log":
            value = math.exp(
                math.log(start) + fraction * (math.log(end) - math.log(start))
            )
        else:
            value = start if fraction < 0.5 else end
        resolved[name] = float(np.float32(value))
    return resolved


def _render(
    descriptor: CorruptionDescriptor,
    image01: tf.Tensor,
    parameters: Mapping[str, float],
    *,
    frame_seed: tf.Tensor,
    episode_seed: tf.Tensor | None,
    auxiliary: Mapping[str, Any],
) -> tf.Tensor:
    return render_unit_image(
        _registered_corruption(descriptor.name, descriptor.version),
        descriptor,
        image01,
        {name: tf.constant(value, tf.float32) for name, value in parameters.items()},
        frame_seed=frame_seed,
        episode_seed=episode_seed,
        mode=select_mode(descriptor, frozenset(auxiliary)),
        auxiliary=auxiliary,
    )


def _record(
    descriptor: CorruptionDescriptor,
    strength: float,
    parameters: Mapping[str, float],
    image: tf.Tensor,
    *,
    provided: frozenset[str],
    episode_provided: bool,
) -> CorruptionRecord:
    return CorruptionRecord(
        name=descriptor.name,
        version=descriptor.version,
        identity=descriptor.identity,
        strength=strength,
        parameters=tuple(parameters.items()),
        spatial_scale_policy=descriptor.spatial_scale_policy,
        reference_width=descriptor.reference_width,
        image_width=image.shape[1],
        mode=select_mode(descriptor, provided),
        auxiliary_inputs=tuple(
            name for name in descriptor.auxiliary_inputs if name in provided
        ),
        episode_seed="provided" if episode_provided else "derived_from_seed",
    )


def _provided_auxiliary(**inputs: Any) -> dict[str, Any]:
    return {name: value for name, value in inputs.items() if value is not None}


def _reject_undeclared(
    provided: Mapping[str, Any],
    declared: tuple[str, ...],
    owner: str,
) -> None:
    undeclared = sorted(set(provided) - set(declared))
    if undeclared:
        raise ValueError(
            f"Auxiliary inputs {undeclared} are not used by {owner}; "
            f"accepted: {list(declared)}."
        )


def _validate_auxiliary(
    provided: Mapping[str, Any],
    image: tf.Tensor,
) -> dict[str, Any]:
    validated: dict[str, Any] = {}
    if "depth_m" in provided:
        depth = _validate_map(provided["depth_m"], image, "depth_m", tf.float32)
        with tf.control_dependencies(
            [
                tf.debugging.assert_greater_equal(
                    depth,
                    0.0,
                    message="depth_m must be non-negative metres (inf is allowed).",
                )
            ]
        ):
            validated["depth_m"] = tf.identity(depth)
    if "region" in provided:
        validated["region"] = _validate_map(
            provided["region"], image, "region", tf.bool
        )
    if "horizon" in provided:
        horizon = provided["horizon"]
        if not isinstance(horizon, Horizon):
            raise TypeError("horizon must be a justdata.vision.horizon.Horizon.")
        validated["horizon"] = _horizon_tensors(horizon)
    return validated


def _validate_map(
    value: Any,
    image: tf.Tensor,
    name: str,
    dtype: tf.DType,
) -> tf.Tensor:
    tensor = tf.convert_to_tensor(value)
    if tensor.dtype != dtype:
        raise TypeError(
            f"{name} must have dtype {dtype.name}, got {tensor.dtype.name}."
        )
    if tensor.shape.rank is not None and tensor.shape.rank != 2:
        raise ValueError(f"{name} must be an HW rank-2 tensor.")
    with tf.control_dependencies(
        [
            tf.debugging.assert_equal(
                tf.shape(tensor),
                tf.shape(image)[:2],
                message=f"{name} must match the image height and width.",
            )
        ]
    ):
        return tf.identity(tensor)


def _horizon_tensors(horizon: Horizon) -> dict[str, tf.Tensor | None]:
    def scalar(value: float | None) -> tf.Tensor | None:
        return None if value is None else tf.constant(value, tf.float32)

    return {
        "rows": tf.constant([horizon.left_row, horizon.right_row], tf.float32),
        "camera_height_m": scalar(horizon.camera_height_m),
        "focal_length_px": scalar(horizon.focal_length_px),
    }


__all__ = [
    "CorruptionRecord",
    "CorruptionResult",
    "CorruptionStackRecord",
    "CorruptionStackResult",
    "apply_corruption_at",
    "apply_corruption_stack",
    "corruption_stack_identity",
    "resolve_parameters",
]

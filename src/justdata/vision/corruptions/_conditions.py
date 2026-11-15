"""Descriptor construction shared by the continuous-strength photometric conditions."""

from collections.abc import Mapping

from justdata.vision.corruptions.registry import (
    EPISODE_SEED_SALT,
    CorruptionDescriptor,
    PlainValue,
    severity_table,
)

_DOMAIN = "decoded_rgb_hwc_0_255"
_RANDOM_SEED_CONTRACT = (
    "caller_supplied_int32_shape_2_frame_seed_and_optional_episode_seed; "
    f"missing_episode_seed_is_stateless_fold_in_seed_{EPISODE_SEED_SALT}; "
    "draw_i_uses_stateless_fold_in_source_seed_i_philox; "
    "stable_for_fixed_tensorflow_version_and_supported_hardware"
)
_DETERMINISTIC_SEED_CONTRACT = (
    "caller_supplied_int32_shape_2; accepted_validated_and_ignored"
)


def photometric_condition(
    name: str,
    *,
    severity: Mapping[str, tuple[int | float, ...]],
    identity: Mapping[str, int | float],
    interpolation: Mapping[str, str],
    implementation: str,
    implementation_parameters: tuple[tuple[str, PlainValue], ...] = (),
    spatial_parameters: tuple[str, ...] = (),
    reference_width: int | None = None,
    episode_draws: tuple[str, ...] = (),
    frame_draws: tuple[str, ...] = (),
    auxiliary_inputs: tuple[str, ...] = (),
    modes: tuple[tuple[str, tuple[str, ...]], ...] = (),
) -> CorruptionDescriptor:
    """Describe a version-1.0.0 condition that never moves pixels or labels."""
    names = tuple(identity)
    uses_randomness = bool(episode_draws or frame_draws)
    return CorruptionDescriptor(
        name=name,
        version="1.0.0",
        input_domain=_DOMAIN,
        input_dtypes=("uint8", "float32"),
        output_domain=_DOMAIN,
        output_shape="same_as_input_hwc",
        output_dtype="uint8",
        severity_values=tuple(range(1, len(next(iter(severity.values()))) + 1)),
        severity_parameters=severity_table({key: severity[key] for key in names}),
        uses_randomness=uses_randomness,
        seed_contract=(
            _RANDOM_SEED_CONTRACT if uses_randomness else _DETERMINISTIC_SEED_CONTRACT
        ),
        clipping_policy="clip_input_and_output_to_closed_interval_0_255",
        rounding_policy="clip_then_round_half_to_even",
        implementation=implementation,
        implementation_parameters=implementation_parameters
        + (("compute_dtype", "float32"),),
        identity_parameters=tuple(identity.items()),
        interpolation=tuple((key, interpolation[key]) for key in names),
        spatial_scale_policy=(
            "relative_to_width" if reference_width is not None else "absolute_pixels"
        ),
        reference_width=reference_width,
        spatial_parameters=spatial_parameters,
        episode_draws=episode_draws,
        frame_draws=frame_draws,
        auxiliary_inputs=auxiliary_inputs,
        modes=modes,
        label_effect="none",
    )

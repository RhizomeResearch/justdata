# justdata.vision.corruptions

## Overview

Versioned discrete-severity and continuous-strength image corruption descriptors, thread-safe corruption registries,
composable corruption stacks, and deterministic or stateless Philox-seeded photometric, atmospheric, optical, motion,
weather, and Mini-C corruptions.

## Interface

- `[__init__.py:__all__]`: Re-exports corruption submodules (`atmosphere`, `blur`, `digital`, `exposure`, `lens`,
  `motion`, `noise`, `precipitation`, `standard`, `water`, `weather`), continuous evaluation APIs from `continuous.py`,
  registry APIs from `registry.py`, and external `Horizon` (`justdata.vision.horizon.Horizon`).
- `[registry.py:CorruptionDescriptor]`, `[registry.py:severity_table]`, `[registry.py:PlainValue]`,
  `[registry.py:SeverityParameters]`, `[registry.py:CorruptionFn]`, `[registry.py:INTERPOLATION_RULES]`,
  `[registry.py:SPATIAL_SCALE_POLICIES]`, `[registry.py:AUXILIARY_INPUTS]`, `[registry.py:EPISODE_SEED_SALT]`: Immutable
  versioned corruption contract (`identity` formatted as `"{name}@{version}:sha256:{digest}"`), type aliases, and
  parameter-table helpers.
- `[registry.py:register_corruption]`, `[registry.py:get_corruption_descriptor]`,
  `[registry.py:list_corruption_descriptors]`, `[registry.py:list_corruption_versions]`,
  `[registry.py:list_corruptions]`, `[registry.py:apply_corruption]`, `[registry.py:apply_minic_corruption]`: Versioned
  and legacy Mini-C corruption registration, lookup, and execution entry points.
- `[registry.py:draw_seed]`, `[registry.py:unit_image]`, `[registry.py:severity_row_parameters]`,
  `[registry.py:select_mode]`, `[registry.py:render_unit_image]`, `[_conditions.py:photometric_condition]`: Shared
  stateless Philox seed derivation, unit-range `[0, 1]` conversion, mode selection, spatial parameter scaling, and
  continuous condition descriptor factory.
- `[continuous.py:StackStep]`, `[continuous.py:CorruptionRecord]`, `[continuous.py:CorruptionResult]`,
  `[continuous.py:CorruptionStackRecord]`, `[continuous.py:CorruptionStackResult]`,
  `[continuous.py:corruption_stack_identity]`, `[continuous.py:resolve_parameters]`,
  `[continuous.py:apply_corruption_at]`, `[continuous.py:apply_corruption_stack]`: Continuous-strength
  (`0.0..max_strength`) single-corruption and multi-step stack execution with serializable provenance records.
- `[atmosphere.py:haze]`, `[atmosphere.py:estimate_airlight]`, `[atmosphere.py:plasma_field]`,
  `[atmosphere.py:HAZE_DESCRIPTOR]`: Continuous atmospheric haze (`'haze'` v`1.0.0`) with `'depth'`, `'horizon'`, and
  `'homogeneous'` modes.
- `[exposure.py:low_light]`, `[exposure.py:overexposure]`, `[exposure.py:white_balance_warm]`,
  `[exposure.py:white_balance_cool]`, `[exposure.py:shot_noise]`, `[exposure.py:BLACKBODY_RGB]`,
  `[exposure.py:LOW_LIGHT_DESCRIPTOR]`, `[exposure.py:OVEREXPOSURE_DESCRIPTOR]`,
  `[exposure.py:WHITE_BALANCE_WARM_DESCRIPTOR]`, `[exposure.py:WHITE_BALANCE_COOL_DESCRIPTOR]`,
  `[exposure.py:SHOT_NOISE_DESCRIPTOR]`: Continuous linear-light and display-sRGB exposure, blackbody color-temperature
  (`3000..15000 K`), and Poisson shot-noise corruptions and descriptors (all v`1.0.0`).
- `[lens.py:lens_droplets]`, `[lens.py:LENS_DROPLETS_DESCRIPTOR]`, `[motion.py:motion_blur]`,
  `[motion.py:motion_kernel]`, `[motion.py:MOTION_BLUR_DESCRIPTOR]`, `[precipitation.py:rain]`,
  `[precipitation.py:RAIN_DESCRIPTOR]`, `[water.py:sun_glitter]`, `[water.py:SUN_GLITTER_DESCRIPTOR]`: Continuous
  optical, motion, precipitation, and water-glitter corruptions (all v`1.0.0`).
- `[standard.py:gaussian_blur]`, `[standard.py:gaussian_noise]`, `[standard.py:jpeg_compression]`,
  `[standard.py:contrast_reduction]`, `[standard.py:brightness_reduction]`, `[standard.py:GAUSSIAN_BLUR_DESCRIPTOR]`,
  `[standard.py:GAUSSIAN_NOISE_DESCRIPTOR]`, `[standard.py:JPEG_COMPRESSION_DESCRIPTOR]`,
  `[standard.py:CONTRAST_REDUCTION_DESCRIPTOR]`, `[standard.py:BRIGHTNESS_REDUCTION_DESCRIPTOR]`,
  `[standard.py:GAUSSIAN_BLUR_SIGMA]`, `[standard.py:GAUSSIAN_NOISE_STD_UNIT]`, `[standard.py:JPEG_QUALITY]`,
  `[standard.py:CONTRAST_FACTOR]`, `[standard.py:BRIGHTNESS_FACTOR]`: Versioned `1..5` discrete sRGB `uint8` benchmark
  corruptions, severity tables, and descriptors (all v`1.0.0`).
- `[blur.py:defocus_blur]`, `[blur.py:BLUR_DESCRIPTOR]`, `[blur.py:DEFOCUS_RADIUS]`, `[blur.py:DEFOCUS_ALIAS]`,
  `[digital.py:pixelate]`, `[digital.py:DIGITAL_DESCRIPTOR]`, `[digital.py:PIXELATE_RATIOS]`,
  `[noise.py:gaussian_noise]`, `[noise.py:NOISE_DESCRIPTOR]`, `[noise.py:NOISE_STD]`, `[weather.py:snow]`,
  `[weather.py:WEATHER_DESCRIPTOR]`, `[weather.py:SNOW_PARAMS]`: Mini-C discrete `1..5` corruptions, parameter tables,
  and descriptors registered under `'blur'`, `'digital'`, `'noise'`, and `'weather'` (all v`1.0.0`).

## Invariants

- `[registry.py:CorruptionDescriptor.__post_init__]`, `[registry.py:CorruptionDescriptor._validate_strength_contract]`:
  Enforce semantic `MAJOR.MINOR.PATCH` versions, immutable plain parameter values, and strict separation between
  discrete-only descriptors (which omit continuous fields from `to_dict()` to preserve SHA-256 identity hashes) and
  continuous descriptors (`supports_strength == True`, `rounding_policy == "clip_then_round_half_to_even"`, matching
  `identity_parameters`/`interpolation` keys, and `bool(auxiliary_inputs) == bool(modes)` with a zero-auxiliary final
  fallback mode whenever `auxiliary_inputs` is non-empty).
- `[registry.py:register_corruption]`: Serializes `_CORRUPTION_REGISTRY` and `_CORRUPTION_DESCRIPTOR_REGISTRY` access
  under `_CORRUPTION_LOCK`, requires `descriptor.name == name`, and raises `ValueError` citing the prior
  `qualified_name` on duplicate `(name, version)` registration.
- `[registry.py:_validate_image]`, `[registry.py:_validate_seed]`, `[registry.py:_finalize_corruption]`,
  `[registry.py:_with_input_shape]`: Require rank-3 `[H, W, 3]` positive-size RGB tensors in `descriptor.input_dtypes`
  and `int32` shape-`[2]` stateless seeds, apply the descriptor's exact rounding policy
  (`clip_then_truncate_toward_zero`, `clip_then_round_half_to_even`, or `codec_native_uint8_decode`), and preserve input
  spatial shape.
- `[continuous.py:_validate_strength]`, `[continuous.py:_interpolate]`, `[continuous.py:apply_corruption_at]`,
  `[continuous.py:apply_corruption_stack]`: Reject `bool`, `tf.Tensor`, non-finite, or out-of-range
  `[0.0, max_strength]` strengths; bypass corruption rendering at `strength == 0` (passing `tf.cast(image, tf.float32)`
  directly through `_finalize_corruption`); execute multi-step stacks in `[0, 1]` float space before rounding once at
  stack completion; and derive per-step seeds from `stable_int64_hash(f"{i}:{descriptor.identity}") % 2**31`.
- `[continuous.py:_reject_undeclared]`, `[continuous.py:_validate_auxiliary]`, `[atmosphere.py:_flat_water_distance]`:
  Reject undeclared non-`None` auxiliary inputs (`ValueError`), validate `depth_m` (`float32` `[H, W]`, `>= 0`),
  `region` (`bool` `[H, W]`), and `horizon` (`Horizon`), and require non-`None` `Horizon.camera_height_m` and
  `Horizon.focal_length_px` in `'haze'` `'horizon'` mode.
- `[registry.py:render_unit_image]`, `[lens.py:lens_droplets]`, `[motion.py:motion_blur]`, `[precipitation.py:rain]`,
  `[water.py:sun_glitter]`: Derive missing `episode_seed` via `draw_seed(frame_seed, EPISODE_SEED_SALT)`, scale
  `spatial_parameters` by `width / reference_width` when `spatial_scale_policy == "relative_to_width"`, and keep
  episode-level layouts/angles/azimuths invariant across frames sharing an `episode_seed`.

## Side Effects

- `[registry.py:_CORRUPTION_REGISTRY]`, `[registry.py:_CORRUPTION_DESCRIPTOR_REGISTRY]`,
  `[registry.py:_CORRUPTION_LOCK]`: Maintain module-level mutable corruption and descriptor registries synchronized by
  `threading.Lock()`.
- `[__init__.py:__all__]`, `[atmosphere.py:haze]`, `[blur.py:defocus_blur]`, `[digital.py:pixelate]`,
  `[exposure.py:low_light]`, `[lens.py:lens_droplets]`, `[motion.py:motion_blur]`, `[noise.py:gaussian_noise]`,
  `[precipitation.py:rain]`, `[standard.py:gaussian_blur]`, `[water.py:sun_glitter]`, `[weather.py:snow]`: Importing
  `justdata.vision.corruptions` eagerly imports all corruption submodules and registers all 19 versioned (`1.0.0`)
  corruption implementations into the global corruption registry.

## Verification

- Run: `uv run pytest tests/test_pipelines.py`

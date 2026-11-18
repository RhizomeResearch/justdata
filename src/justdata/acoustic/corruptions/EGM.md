# `justdata.acoustic.corruptions`

## Overview

Deterministic, stateless TensorFlow audio corruption transforms (severities 1–5), severity lookup tables, corruption
registry helpers, and benchmark evaluation dataset builders (`Audio-C`-style waveform and spectrogram corruptions). All
modules depend only on `tensorflow` and internal `justdata` packages.

## Interface

- `[__init__.py:__all__]` Exports `apply_audio_corruption`, `codec`, `create_audio_corruption_datasets`, `device`,
  `dynamics`, `filters`, `get_audio_corruption`, `has_audio_corruption`, `list_audio_corruptions`, `noise`,
  `register_audio_corruption`, `reverb`, and `timefreq`.
- `[registry.py:SeverityTable]` Module type alias `Mapping[int, float | int | tuple[float, ...]]`.
- `[registry.py:apply_audio_corruption]`
  `(audio_or_features: tf.Tensor, corruption: str, severity: int, seed: tf.Tensor, config: dict | None = None) -> tf.Tensor`
  validating severity, normalizing `seed`, resolving `corruption`, and invoking it.
- `[registry.py:register_audio_corruption]`, `[registry.py:get_audio_corruption]`,
  `[registry.py:list_audio_corruptions]`, `[registry.py:has_audio_corruption]` Re-exported corruption registry decorator
  and query functions from `justdata.acoustic.registry`.
- `[datasets.py:create_audio_corruption_datasets]`
  `(corruption_types, severity, base_dataset, preset, *, split='validation', seed=0, corruption_domain='waveform', **load_kwargs) -> tuple[tf.data.Dataset | list[tf.data.Dataset], int]`
  constructing deterministic corrupted `tf.data.Dataset` pipelines (popping optional `corruption_configs` from
  `load_kwargs` for per-corruption `config` dicts).
- `[codec.py:CODEC_BITRATE_KBPS]`, `[codec.py:CODEC_MULAW_LEVELS]`, `[codec.py:RESAMPLING_INTERMEDIATE_HZ_32K]`,
  `[codec.py:codec_compression]`, `[codec.py:resampling_degradation]` Mu-law quantization roundtrip and resampling
  degradation corruptions and severity tables.
- `[device.py:DEVICE_IR_SET]`, `[device.py:device_ir]` Symmetric 5-tap FIR device impulse-response corruption and
  severity table.
- `[dynamics.py:CLIPPING_THRESHOLD]`, `[dynamics.py:DRC_THRESHOLD_DB]`, `[dynamics.py:DRC_RATIO]`,
  `[dynamics.py:clipping]`, `[dynamics.py:dynamic_range_compression]` Amplitude hard-clipping and static dB-domain
  dynamic range compression corruptions and severity tables.
- `[filters.py:LOWPASS_HZ_32K]`, `[filters.py:HIGHPASS_HZ_32K]`, `[filters.py:BANDPASS_HZ_32K]`,
  `[filters.py:EQUALIZATION_TILT_DB_PER_OCTAVE]`, `[filters.py:low_pass]`, `[filters.py:high_pass]`,
  `[filters.py:band_pass]`, `[filters.py:equalization_tilt]` RFFT brick-wall filter masks and spectral tilt corruptions
  and severity tables.
- `[noise.py:WHITE_NOISE_SNR_DB]`, `[noise.py:PINK_NOISE_SNR_DB]`, `[noise.py:BROWN_NOISE_SNR_DB]`,
  `[noise.py:BACKGROUND_NOISE_SNR_DB]`, `[noise.py:additive_white_noise]`, `[noise.py:additive_pink_noise]`,
  `[noise.py:additive_brown_noise]`, `[noise.py:background_noise]` Stateless unit-RMS colored and background/hum noise
  corruptions and SNR severity tables.
- `[reverb.py:REVERB_RT60_SECONDS]`, `[reverb.py:reverb_rir]` Delay-compensated L2-normalized custom or synthetic RT60
  room impulse response corruption and severity table.
- `[timefreq.py:TIME_STRETCH_FACTOR]`, `[timefreq.py:PITCH_SHIFT_SEMITONES]`,
  `[timefreq.py:PACKET_DROPOUT_PROBABILITY]`, `[timefreq.py:PACKET_DROPOUT_GAP_MS]`, `[timefreq.py:time_stretch]`,
  `[timefreq.py:pitch_shift]`, `[timefreq.py:packet_dropout_gaps]` Time-stretch, pitch-shift, and contiguous
  packet-dropout corruptions and severity tables.

## Invariants

- `[registry.py:_validate_severity]` and `[registry.py:_severity_value]` Truncate numeric `severity` via
  `int(...)`/`tf.cast(..., tf.int32)` and enforce inclusive `1..5` via Python, eager `.numpy()`, and
  `tf.debugging.assert_greater_equal` / `assert_less_equal` checks across `apply_audio_corruption` and severity-table
  lookups (direct calls to `[device.py:device_ir]` with `'ir' in config` or `[reverb.py:reverb_rir]` with
  `config.get('rir') is not None` bypass severity lookup).
- `[registry.py:_as_tc]` and `[registry.py:_restore_rank]` Require static input rank `1` (`[T]`) or `2` (`[T, C]`),
  raising `ValueError('Audio corruption input must have shape [T] or [T, C].')` otherwise, compute internally in
  `tf.float32`, and restore 1D rank when input rank was `1`.
- `[registry.py:_sample_rate]` Reads `config['sample_rate']` when provided and defaults to `32000` Hz (`tf.float32`).
- `[registry.py:_seed_from_index]` Derives deterministic per-sample `int32[2]` stateless seeds
  `[base[0], (base[1] + index + salt) mod (2**31 - 1)]` in `int64` arithmetic; all stochastic corruptions (`noise.py`,
  `reverb.py`, `timefreq.py`) use `tf.random.stateless_*` exclusively.
- `[registry.py:_metadata_with_corruption]` Copies `sample` and `sample['metadata']`, writing Tensor keys `'corruption'`
  (`tf.string`), `'severity'` (`tf.int32`), and `'corruption_domain'` (`tf.string`).
- `[datasets.py:create_audio_corruption_datasets]` Validates `corruption_domain in {'waveform', 'spectrogram'}` (raising
  `ValueError` otherwise), pops optional `corruption_configs` (`dict[str, dict]`) from `load_kwargs`, defaults
  `dataset_type` to `'train'` when `split == 'train'` else `'validation'`, forces `return_raw_ds=True`, defaults
  `deterministic=True` and `batch_size=32`, and returns `([], 0)` for an empty list, `(list[Dataset], n_batches)` for a
  list, or `(Dataset, n_batches)` for a `str`.
- `[datasets.py:_corrupt_sample]` Raises `ValueError` if `'waveform'` (plus `'sample_rate'`) or `'features'` is missing
  for the selected `corruption_domain`, injects `sample['sample_rate']` via `setdefault` for waveform corruptions, and
  salts per-corruption seeds with `index * 1_000_003`.
- `[codec.py:codec_compression]`, `[filters.py:equalization_tilt]`, `[reverb.py:reverb_rir]`,
  `[timefreq.py:packet_dropout_gaps]` Signal bounds: `codec_compression` clips inputs/outputs to `[-1.0, 1.0]` with
  `levels >= 2`; `equalization_tilt` floors frequencies at `20.0` Hz and clips linear gain to `[0.125, 4.0]`;
  `reverb_rir` clamps synthetic IR length to `[2, max_ir_samples]` (`4096`) and L2-normalizes with `1e-8` floor;
  `packet_dropout_gaps` guarantees at least one dropped gap segment when the initial Bernoulli draw is all-`False`.

## Side Effects

- `[__init__.py:__all__]` Importing `justdata.acoustic.corruptions` imports `codec`, `device`, `dynamics`, `filters`,
  `noise`, `reverb`, and `timefreq`, registering 17 built-in corruptions into
  `justdata.acoustic.registry._AUDIO_CORRUPTIONS` under `_REGISTRY_LOCK` (`codec_compression`, `resampling_degradation`,
  `device_ir`, `clipping`, `dynamic_range_compression`, `low_pass`, `high_pass`, `band_pass`, `equalization_tilt`,
  `additive_white_noise`, `additive_pink_noise`, `additive_brown_noise`, `background_noise`, `reverb_rir`,
  `time_stretch`, `pitch_shift`, `packet_dropout_gaps`).
- `[registry.py:register_audio_corruption]` Acquires `justdata.acoustic.registry._REGISTRY_LOCK` and mutates global
  `_AUDIO_CORRUPTIONS`, raising `ValueError` on duplicate names.
- `[datasets.py:create_audio_corruption_datasets]` Delegates dataset loading, optional caching, and pipeline resolution
  to `justdata.core.registry.get_pipeline` and `justdata.core.loader.load_ds`; no direct file/network I/O, logging, or
  stateful `tf.random.Generator` mutation occurs inside `justdata.acoustic.corruptions`.

## Verification

- Run: `uv run pytest tests/acoustic`

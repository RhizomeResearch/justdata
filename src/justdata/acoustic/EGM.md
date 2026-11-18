# `justdata.acoustic`

## Overview

TensorFlow-native acoustic modality subsystem implementing canonical audio sample adaptation, WAV and Hugging Face /
manifest / DCASE 2025 Task 1 source loading, resampling, channel reduction, waveform and feature normalization,
segmentation and evaluation views, spectral frontends, stateless augmentations and corruptions, serializable/hashable
presets, thread-safe registries, metadata encoding, JAX/NumPy conversion, and core `tf.data` task pipelines.

## Interface

- `[__init__.py:__all__]` Exports 176 public symbols across `justdata.acoustic`, eagerly importing all 31 submodules
  (`adapters`, `augment`, `batching`, `channel`, `compat`, `configs`, `corruptions`, `datasets`, `decoding`,
  `dcase2025`, `evals`, `eval_views`, `filters`, `frontends`, `jax`, `labels`, `layouts`, `metadata`, `normalization`,
  `pipelines`, `postprocessing`, `preprocessing`, `presets`, `registry`, `resampling`, `schema`, `segment`, `sources`,
  `stages`, `stats`, `tasks`).
- `[schema.py:AudioSample]` `TypedDict` defining canonical keys `[schema.py:WAVEFORM]` (`'waveform'`),
  `[schema.py:SAMPLE_RATE]` (`'sample_rate'`), and optional `[schema.py:LABEL]`, `[schema.py:FEATURES]`,
  `[schema.py:DURATION]`, `[schema.py:METADATA]`, alongside source/manifest key constants (`AUDIO`,
  `ORIGINAL_SAMPLE_RATE`, `PATH`, `FILENAME`, `DATASET`, `SPLIT`, `EXAMPLE_ID`, `CLIP_ID`, `SOURCE_ID`, `START_TIME`,
  `END_TIME`).
- `[configs.py:AudioPreset]`, `[configs.py:AudioPreprocessConfig]`, `[configs.py:SegmentStrategyConfig]`,
  `[configs.py:STFTConfig]`, `[configs.py:MelConfig]`, `[configs.py:LogCompressionConfig]`,
  `[configs.py:FeatureNormConfig]`, `[configs.py:FrontendConfig]`, `[configs.py:LabelTransformConfig]` Frozen
  `_SerializableConfig` dataclasses providing `validate()`, `to_dict()`, `to_json()`, `from_dict()`, `from_json()`, and
  `hash()`, plus `SegmentPadMode`, `DurationPolicy`, and `LabelTransformMode` type aliases.
- `[adapters.py:AcousticAdapter]`, `[adapters.py:adapt_acoustic_sample]`, `[adapters.py:ensure_audio_rank]`,
  `[adapters.py:to_float32_waveform]`, `[adapters.py:standardize_waveform_layout]`, `[adapters.py:infer_duration]`,
  `[adapters.py:make_audio_metadata]`, `[adapters.py:esc50_adapter]`, `[adapters.py:speech_commands_adapter]`,
  `[adapters.py:audioset_adapter]`, `[adapters.py:acoustic_source_adapter]`, `[adapters.py:dcase2025_task1_adapter]`
  Sample schema adapters and waveform normalization helpers.
- `[decoding.py:decode_wav_bytes]`, `[decoding.py:decode_wav_file]`, `[decoding.py:decode_audio_file]`,
  `[decoding.py:decode_wav_sample]` WAV byte/file decoders returning `(tf.float32 [T, C], tf.int32 sample_rate)` and
  registered `'wav'` decoder.
- `[sources.py:load_local_audio_manifest_splits]`, `[sources.py:load_dcase2025_splits]`,
  `[sources.py:load_huggingface_audio_splits]` Registered core source loaders for `'local_audio:'`, `'dcase2025:'`, and
  `'hf_audio:'` prefixes; `[datasets.py:AUDIO_CLASSIFICATION_SINGLE_LABEL]`,
  `[datasets.py:AUDIO_CLASSIFICATION_MULTI_LABEL]`, `[datasets.py:AUDIO_EVENT_DETECTION]`,
  `[datasets.py:AUDIO_RETRIEVAL]`, `[datasets.py:AUDIO_TEXT_PAIR]`, `[datasets.py:AUDIO_SELF_SUPERVISED]` task-kind
  constants.
- `[dcase2025.py:DCASE2025Task1Adapter]`, `[dcase2025.py:dcase2025_task1_adapter]`, `[dcase2025.py:SplitLeakageError]`,
  `[dcase2025.py:assert_stats_allowed]`, `[dcase2025.py:load_dcase2025_task1_splits]`,
  `[dcase2025.py:make_source_dataset]`, `[dcase2025.py:make_target_dataset]`, `[dcase2025.py:parse_scene_from_row]`,
  `[dcase2025.py:parse_device_from_row]`, `[dcase2025.py:parse_city_from_row]`,
  `[dcase2025.py:parse_source_recording_id]` DCASE 2025 Task 1 loaders, registered adapter, split-leakage guards,
  metadata parsers, device sets (`known_train_devices`, `real_devices`, `simulated_devices`), and split-policy map
  `allow_stats_on_split`, backed by `[_dcase.py:DCASE_CLASSES]`.
- `[channel.py:apply_channel_strategy]`, `[channel.py:mono_mean]`, `[channel.py:mono_left]`, `[channel.py:mono_right]`,
  `[channel.py:keep]` Registered `[T, C]` channel strategies.
- `[resampling.py:resample_waveform]`, `[resampling.py:resample_sample]`, `[resampling.py:resample_sample_soxr]`,
  `[resampling.py:resample_sample_kaiser_best]`, `[resampling.py:resample_sample_hf_audio]` Waveform and sample-dict
  resamplers (`'tensorflow'`, `'soxr'`, `'kaiser_best'`, `'hf_audio'`, `'identity'`).
- `[preprocessing.py:make_preprocessing]`, `[preprocessing.py:normalize_waveform]`,
  `[preprocessing.py:remove_dc_offset]` Deterministic pre-cache decoding, resampling, channel reduction, DC removal, and
  `'none'`/`'peak'`/`'rms'` normalization.
- `[segment.py:pad_waveform]`, `[segment.py:segment_waveform]`, `[stages.py:SeededSampleStage]`,
  `[stages.py:make_segment_stage]`, `[stages.py:make_preprocessing_stage]`, `[eval_views.py:generate_eval_views]`,
  `[eval_views.py:make_eval_views]` Waveform padding, seeded segmentation (`'random_crop'`, `'center_crop'`, `'full'`,
  `'sliding'`, `'pad_or_crop'`, `'multi_crop'`), and deterministic evaluation view builders.
- `[normalization.py:apply_feature_normalization]`, `[normalization.py:normalize_none]`,
  `[normalization.py:normalize_dataset_mean_std]`, `[normalization.py:normalize_checkpoint_mean_std]`,
  `[normalization.py:normalize_per_clip_mean_std]`, `[normalization.py:normalize_per_frequency_mean_std]`,
  `[normalization.py:normalize_kaldi_cmvn]`, `[normalization.py:normalize_affine]` Deterministic spectral feature
  normalizers.
- `[layouts.py:AudioLayout]`, `[layouts.py:OutputKind]`, `[layouts.py:convert_audio_layout]`,
  `[layouts.py:unbatched_shape_for_layout]` Unbatched layout converters (`'bt'`, `'btc'`, `'btf'`, `'bft'`, `'bcft'`,
  `'btfc'`) and static shape calculators.
- `[labels.py:transform_label]` Label encoder supporting `'index'`, `'one_hot'`, `'multi_hot'`, `'text'`, and
  `'event_frames'` modes.
- `[postprocessing.py:frontend_output_kind]`, `[postprocessing.py:default_output_key]`,
  `[postprocessing.py:expected_audio_static_shape]`, `[postprocessing.py:set_static_audio_shape]`,
  `[postprocessing.py:cast_audio_dtype]`, `[postprocessing.py:pad_or_crop_time]`,
  `[postprocessing.py:pad_time_to_multiple]`, `[postprocessing.py:pad_to_patch_multiple]`,
  `[postprocessing.py:make_model_input_stage]`, `[postprocessing.py:preset_info]` Deterministic model-input projection
  and static-shape utilities.
- `[tasks.py:identity]`, `[tasks.py:make_preprocessing]`, `[tasks.py:make_augmentations]`,
  `[tasks.py:make_late_augmentations]`, `[tasks.py:make_postprocessing]` and `[pipelines.py:default_pipeline]`,
  `[pipelines.py:classification_pipeline]`, `[pipelines.py:ast_classification_pipeline]`,
  `[pipelines.py:identity_pipeline]` Stage builders and registered core pipelines (`'acoustic/default'`,
  `'acoustic/classification'`, `'acoustic/ast_classification'`, `'acoustic/identity'`).
- `[presets.py:register_preset]`, `[presets.py:get_dataset_presets]`, `[presets.py:get_resolved_preset]`,
  `[presets.py:merge_with_presets]` Modality-scoped (`modality='acoustic'`) preset resolution and 26 built-in presets.
- `[registry.py:register_audio_decoder]`, `[registry.py:register_audio_resampler]`,
  `[registry.py:register_audio_channel_strategy]`, `[registry.py:register_audio_segment_strategy]`,
  `[registry.py:register_audio_frontend]`, `[registry.py:register_audio_waveform_augment]`,
  `[registry.py:register_audio_spectrogram_augment]`, `[registry.py:register_audio_batch_augment]`,
  `[registry.py:register_audio_normalization]`, `[registry.py:register_audio_eval_view_strategy]`,
  `[registry.py:register_audio_postprocessor]`, `[registry.py:register_audio_corruption]` (plus matching `get_audio_*`,
  `list_audio_*`, `has_audio_*`, and `*_metadata` queries).
- `[metadata.py:MetadataEncoder]`, `[metadata.py:MetadataSidecar]`, `[metadata.py:stable_int64_hash]`,
  `[filters.py:filter_by_metadata]`, `[filters.py:groupby_metadata]`, `[filters.py:filter]`, `[filters.py:groupby]`,
  `[filters.py:metadata_filter_predicate]`, `[batching.py:pad_nested]`, `[jax.py:to_numpy]`,
  `[jax.py:as_numpy_iterator]`, `[stats.py:make_stats_iterator]`, `[stats.py:compute_feature_stats]`,
  `[evals.py:macro_classwise_accuracy]`, `[evals.py:per_device_accuracy]`, `[evals.py:per_scene_accuracy]`,
  `[evals.py:confusion_matrix]` Metadata encoding/filtering, padding, NumPy/JAX iteration, split-checked feature
  statistics, and evaluation metrics.
- Subdirectory contracts:
  - [`augment`](./augment/EGM.md): Stateless waveform, spectrogram, PaSST patchout, and batch mixing augmentations,
    `*Config` dataclasses, and `tf.data` augmentation stage builders.
  - [`compat`](./compat/EGM.md): Frontend-golden presets, frontend configurations, and pipelines for AST, CED,
    EfficientAT/DyMN, PANNs, and PaSST.
  - [`corruptions`](./corruptions/EGM.md): Deterministic severity 1–5 audio corruptions, severity lookup tables, and
    `create_audio_corruption_datasets`.
  - [`frontends`](./frontends/EGM.md): Deterministic waveform, STFT, mel, log/dB/PCEN, MFCC, Kaldi, and AST Kaldi
    feature frontends.

## Invariants

- `[schema.py:AudioSample]` and `[adapters.py:adapt_acoustic_sample]` Canonical sample dicts always carry `waveform`
  (`tf.float32` `[T, C]`) and `sample_rate` (`tf.int32`), with PCM scaling (`int16` `/32768`, `int32` `/2147483648`,
  `uint8` `(x-128)/128` in `[adapters.py:to_float32_waveform]`) and rank-1/2 layout standardization
  (`[adapters.py:standardize_waveform_layout]`).
- `[configs.py:_SerializableConfig]` and `[configs.py:AudioPreset.validate]` All config dataclasses validate in
  `__post_init__`, serialize deterministically (`sort_keys=True`, `separators=(',', ':')`), and compute 16-hex-character
  SHA-256 hashes over canonical UTF-8 JSON; `AudioPreset` enforces
  `preprocess.target_sample_rate == target_sample_rate`, `dtype in {'float32', 'float16', 'bfloat16'}`, and
  `metadata_mode in {'full', 'numeric_only', 'none'}`.
- `[registry.py:_register]` and `[registry.py:_register_augment]` All 12 acoustic registries synchronize mutations under
  `_REGISTRY_LOCK`, fail with `ValueError` on duplicate registration or unknown lookup, and return sorted immutable
  tuples from `list_audio_*`.
- `[pipelines.py:default_pipeline]`, `[pipelines.py:_resolve_acoustic_config]`, `[tasks.py:make_augmentations]`,
  `[tasks.py:make_late_augmentations]`, `[tasks.py:make_postprocessing]` Enforce the
  `(preprocess_fn, augment_fn, late_augment_fn, postprocess_fn)` contract, reject unknown config keys via
  `reject_unknown_keys`, forbid `spectrogram_augmentations` in pre-frontend `make_augmentations`, require `layout`
  whenever `frontend` is set, and split stateless `int32[2]` seeds (`[_random.py:_seed_tensor]`) across sample and batch
  stages without sharing stateful `tf.random.Generator` instances inside `justdata.acoustic`.
- `[postprocessing.py:_fixed_eval_view_count]` Rejects `duration_policy == 'sliding_windows'` and
  `eval_mode == 'sliding'` during static-shape resolution (`ValueError`) because variable view counts cannot be assigned
  a fixed static shape; `[eval_views.py:generate_eval_views]` forces `is_training=False` and `seed=None`.
- `[dcase2025.py:assert_stats_allowed]` and `[stats.py:make_stats_iterator]` Allow split statistics only on
  `'dev_train_25'`, raising `SplitLeakageError` on `'dev_test'` or `'eval'` unless `allow_override=True` (which logs a
  warning and emits `RuntimeWarning`), and raising `ValueError` on unknown splits.
- `[sources.py:load_local_audio_manifest_splits]`, `[sources.py:load_huggingface_audio_splits]`, and
  `[sources.py:_import_datasets]` Require all 7 manifest columns
  (`{'path', 'label', 'split', 'clip_id', 'source_id', 'start_time', 'end_time'}`) for `'local_audio:'`, enforce
  `decode_mode in {'hf_native', 'justdata'}` and existing audio/label columns for `'hf_audio:'`, assert dataset
  cardinality, and guard optional `datasets` behind an explicit `ImportError` message.
- `[resampling.py:resample_waveform]`, `[resampling.py:_soxr_resample_numpy]`, and
  `[resampling.py:_kaiser_best_resample_numpy]` Enforce
  `method in {'identity', 'hf_audio', 'tensorflow', 'soxr', 'kaiser_best'}`, asserting
  `original_sample_rate == target_sample_rate` via `tf.debugging.assert_equal` for `'identity'` and `'hf_audio'`,
  short-circuiting via `tf.cond` when rates match, raising `ImportError` when `soxr` is absent, and falling back from
  `librosa` to `_soxr_resample_numpy` when `librosa` is absent in `'kaiser_best'`.
- `[metadata.py:MetadataEncoder.encode]` Encodes categorical fields (`device_id`, `scene_id`, `city_id`, `split_id`,
  `dataset_id`) as `tf.int32` (passing `int`/`bool` values through directly and mapping `None` or unseen strings to
  `-1`), `is_known_device` as `tf.bool`, `source_id_hash` as `tf.int64` via `stable_int64_hash`, and `example_id` as
  `tf.int64` (hashing `(dataset, split, clip_id)` or string `example_id` via `stable_int64_hash`, or passing an existing
  integer `example_id` through directly when those triple keys are not all present).

## Side Effects

- `[__init__.py:__all__]` Importing `justdata.acoustic` eagerly imports all 31 submodules (including `augment`,
  `compat`, `corruptions`, and `frontends`), mutating global registries in `justdata.core.registry`,
  `justdata.core.sources`, `justdata.core.adapters`, `justdata.core.presets`, and `justdata.acoustic.registry` under
  thread locks to register built-in datasets (`esc50`, `speech_commands`, `audioset`, `dcase2025_task1`), source loaders
  (`local_audio:`, `dcase2025:`, `hf_audio:`, `dcase2025_task1`), adapters, 4 core pipelines, 27 presets (`presets.py` +
  `dcase2025.py`), and all built-in decoders, resamplers, channel/segment/eval-view strategies, normalizers,
  postprocessors (`identity`), frontends, augmentations, and corruptions.
- `[sources.py:load_local_audio_manifest_splits]`, `[sources.py:load_huggingface_audio_splits]`,
  `[dcase2025.py:load_dcase2025_task1_splits]`, `[decoding.py:decode_wav_file]`, `[preprocessing.py:_extract_audio]`
  Read local manifests and audio files (`_read_manifest`, `tf.io.read_file`) and, for `'hf_audio:'`, create cache
  directories via `source_cache_dir` and invoke `datasets.load_dataset` (network/disk I/O).
- `[resampling.py:_soxr_resample_numpy]`, `[resampling.py:_kaiser_best_resample_numpy]`, `[sources.py:_import_datasets]`
  Lazily import optional `acoustic`-extra libraries (`soxr`, `librosa`, `datasets`) at call time.
- `[dcase2025.py:assert_stats_allowed]`, `[sources.py:_records_to_dataset]`, and
  `[sources.py:load_huggingface_audio_splits]` Emit `loguru.logger.warning` messages (and `RuntimeWarning` on
  `allow_override=True` split overrides).
- `[metadata.py:MetadataEncoder.fit]`, `[postprocessing.py:set_static_audio_shape]`, `[pipelines.py:default_pipeline]`
  Mutate `MetadataEncoder.vocab` on `fit()`, tensor `TensorShape` metadata via `set_shape()`, and non-empty
  caller-passed `*_kwargs` dicts via `.setdefault(...)`.

## Verification

- Run: `uv run pytest tests/acoustic`

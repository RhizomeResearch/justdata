# `justdata.core`

## Overview

`justdata.core` is the modality-neutral execution engine for `justdata`. It provides thread-safe dataset, source,
adapter, pipeline, and preset registries; the canonical `tf.data` stage orchestrator and batch finalizer; deterministic
JSON configuration snapshots; quota-enforced TFRecord stage caching; detached metadata sidecar transport and filtering;
SHA-256-verified offline inventory snapshots; cursor-resumable deterministic epoch replay; and grouped Welford feature
statistics without importing modality packages or assuming modality-specific sample keys.

## Interface

- `[__init__.py:__all__]` — Re-exports the 52 package-level public symbols across loading, finalization, registries,
  sources, adapters, presets, caching, execution, configuration, metadata, filtering, inventory, replay, and statistics.
- `[loader.py:fetch_ds]` —
  `fetch_ds(dataset_names, splits_info, data_dir=None, *, source_filter_fn=None, map_parallel_calls=None) -> Optional[tf.data.Dataset]`
  resolves source loaders and dataset adapters, applies optional pre-adapter `source_filter_fn`, and concatenates splits
  and datasets in requested order.
- `[loader.py:load_ds]` — `load_ds(dataset_names_arg, splits_arg, dataset_type, batch_size, seed, ...)` executes the
  canonical `tf.data` pipeline, returning `(dataset_or_iter, n_batches)` by default, `(raw_ds, tools_dict)` exposing
  `finalize_fn`, `finalize_epoch`, `postprocess_fn`, and `rng` when `return_raw_ds=True`, and appending `ExecutedConfig`
  when `return_config=True`.
- `[finalization.py:finalize_dataset]` — `finalize_dataset(ds, *, postprocess_fn, num_classes, batch_size, ...)` runs
  post-cache stages (`shuffle` before `postprocess` by default or after `postprocess` + model-input cache when
  `cache_model_inputs=True`, `batch`, `late_augment`, `pad` with `padding_mask`, `prefetch`, and optional `as_numpy`
  conversion).
- `[registry.py:DatasetInfo]`, `[registry.py:register_dataset]`, `[registry.py:get_dataset_info]`,
  `[registry.py:get_task_for_dataset]` — Frozen dataset metadata record and global dataset registration/lookup helpers
  with exact-then-longest-prefix matching.
- `[registry.py:DataPipeline]`, `[registry.py:PipelineFuncs]`, `[registry.py:register_pipeline]`,
  `[registry.py:get_pipeline]`, `[registry.py:get_pipeline_for_dataset]`, `[registry.py:has_pipeline]`,
  `[registry.py:list_pipelines]`, `[registry.py:qualified_name]` — Modality-aware
  `(preprocess, augment, batch_augment, postprocess)` pipeline container, decorator registration, config resolution
  (`resolve_config`), and callable stage builder (`build`).
- `[sources.py:SourceLoader]`, `[sources.py:register_source_loader]`, `[sources.py:register_default_source_loader]`,
  `[sources.py:get_source_loader]`, `[sources.py:load_tfds_splits]`, `[sources.py:source_cache_dir]` — Source loader
  protocol, prefix/default loader registration, TFDS default split loader, and `~/.cache/justdata` cache directory
  resolver.
- `[adapters.py:DatasetAdapter]`, `[adapters.py:register_adapter]`, `[adapters.py:get_adapter]` — Raw-to-canonical
  sample dictionary adapter protocol and registry with identity fallback (`_default_adapter`) when `required=False`.
- `[presets.py:ResolvedPreset]`, `[presets.py:register_preset]`, `[presets.py:get_dataset_presets]`,
  `[presets.py:get_resolved_preset]`, `[presets.py:merge_with_presets]`, `[presets.py:merge_explicit_overrides]`,
  `[presets.py:canonical_preset_json]`, `[presets.py:preset_hash]` — Modality-scoped preset registry, default-aware and
  explicit override merging, canonical JSON encoding, and 16-hex SHA-256 preset hashing.
- `[executed_config.py:ExecutedConfig]`, `[executed_config.py:canonical_config_json]` — Frozen canonical JSON snapshot
  (`from_dict`, `to_dict`, `to_json`, `to_bytes`) of resolved pipeline and execution parameters.
- `[config_resolution.py:is_positive_int]`, `[config_resolution.py:reject_unknown_keys]`,
  `[config_resolution.py:resolve_callable_config]` — Signature-inspected callable parameter validation and deep-copied
  default binding helpers.
- `[cache.py:CachePolicy]`, `[cache.py:CachePolicy.status]`, `[cache.py:CacheError]`, `[cache.py:inspect_cache]` —
  Quota-bounded (`max_bytes`, `max_examples`), fingerprint-verified TFRecord cache policy (`justdata.cache.v1`) with
  per-stage status lookup, structured `RuntimeError` (`code`, `path`, `message`), and cache directory state inspector
  (`missing`, `incomplete`, `complete`, `corrupt`).
- `[execution.py:configure_tensorflow_cpu]` — Configures CPU-only device visibility and TensorFlow intra-op/inter-op
  thread counts.
- `[augmentations.py:AugmentationDomain]`, `[augmentations.py:AugmentationMetadata]`,
  `[augmentations.py:attach_augmentation_metadata]` — Frozen augmentation descriptor (`waveform`, `spectrogram`,
  `image`, `batch`) and callable metadata attachment helper.
- `[label_mixing.py:LabelMixMode]`, `[label_mixing.py:apply_label_smoothing]`,
  `[label_mixing.py:prepare_labels_for_mixing]`, `[label_mixing.py:broadcast_mix_coefficient]`,
  `[label_mixing.py:blend_prepared_labels]`, `[label_mixing.py:mix_labels]` — Label smoothing and linear/BCE label
  blending across `single_label`, `multi_label`, and `event_frames` modes.
- `[metadata.py:MetadataSidecar]`, `[metadata.py:apply_metadata_mode]`, `[metadata.py:numeric_metadata]`,
  `[metadata.py:stable_int64_hash]`, `[metadata.py:python_value]`, `[metadata.py:json_value]`,
  `[metadata.py:get_metadata_value]`, `[metadata.py:attach_sidecar_writer]`, `[metadata.py:attach_sidecar_mapping]`,
  `[metadata.py:attach_sidecar_views]`, `[metadata.py:verify_sidecar_keys]` — Sample metadata filtering (`full`,
  `numeric_only`, `none`), process-stable 63-bit integer hashing, and detached source/view JSONL sidecar binding and
  post-cache fingerprint verification.
- `[filters.py:metadata_filter_predicate]`, `[filters.py:filter_by_metadata]`, `[filters.py:groupby_metadata]` — Exact
  and `<key>_in` membership `tf.data` sample filtering and metadata-keyed dataset grouping.
- `[inventory.py:InventoryAsset]`, `[inventory.py:InventoryRecord]`, `[inventory.py:InventorySource]`,
  `[inventory.py:InventoryFilter]`, `[inventory.py:InventoryLoadError]`, `[inventory.py:AdmittedInventory]`,
  `[inventory.py:admit_inventory]`, `[inventory.py:open_inventory]`, `[inventory.py:load_inventory]` — Offline SHA-256
  asset verification, TFRecord snapshot admission (`justdata.inventory.v1`), manifest/row integrity verification, and
  pipeline preparation.
- `[replay.py:ReplayState]`, `[replay.py:ReplayEpoch]`, `[replay.py:ReplayError]`, `[replay.py:count_real_examples]`,
  `[replay.py:load_replay_epoch]` — Deterministic, cursor-resumable epoch builder (`justdata.replay.v1`) and batch
  padding-mask real-example counter.
- `[stats.py:make_stats_iterator]`, `[stats.py:compute_feature_stats]` — Non-augmented NumPy batch iterator and float64
  Chan/Welford grouped feature mean/std accumulator.

## Invariants

- **Modality-Neutral Core**: `justdata.core` imports only Python stdlib, `numpy`, `tensorflow`, `tensorflow_datasets`,
  `loguru`, `justdata.__version__`, and `justdata.core.*`, never importing modality packages or inspecting
  modality-specific tensor keys (`[loader.py:__version__]`, `[loader.py:_prepare_ds]`, `[finalization.py:_pad_nested]`).
- **Canonical Stage Order**: Standard pipeline execution follows
  `('fetch_ds', 'adapter', 'preprocess', 'cache', 'augment', 'shuffle', 'postprocess', 'batch', 'late_augment', 'pad', 'prefetch')`
  (when `cache_model_inputs=True`, `finalize_dataset` moves `shuffle` after `postprocess` and the model-input cache);
  `return_raw_ds=True` stops after `cache` (and optional `filter_fn`), with `finalize_epoch` applying `augment_fn` and
  post-cache stages while `finalize_fn` runs only post-`augment` finalization (`not_applied_by_finalize_fn`)
  (`[loader.py:_STAGE_ORDER]`, `[loader.py:_RAW_PENDING_STAGES]`, `[loader.py:finalize_epoch]`,
  `[finalization.py:finalize_dataset]`).
- **RNG Isolation & Determinism**: Per-sample `seeded_augment` and batch-level `late_augment` never share a stateful
  `tf.random.Generator`; `_prepare_ds` allocates distinct generators (`from_seed(seed)` vs `.split(1)[0]`) and forces
  `num_parallel_calls=1` when `deterministic=True`, while `finalize_epoch` splits `seed` into stateless per-example and
  per-batch indexed seeds via `_seed_for_index` with `reshuffle_each_iteration=False` (`[loader.py:_prepare_ds]`,
  `[loader.py:finalize_epoch]`, `[finalization.py:_seed_for_index]`, `[finalization.py:finalize_dataset]`).
- **Batch Padding Contract**: Every finalized batch includes a boolean `padding_mask` of shape `(batch_size,)` (`True`
  for real examples, `False` for padded rows), padding partial batches along axis `0` with `""` for `tf.string` and `0`
  for numeric/boolean leaves (`[finalization.py:_pad_dataset]`, `[finalization.py:_pad_nested]`,
  `[replay.py:count_real_examples]`).
- **Thread-Safe Duplicate-Rejecting Registries**: Mutations to datasets (`[registry.py:register_dataset]`), pipelines
  (`[registry.py:register_pipeline]`), adapters (`[adapters.py:register_adapter]`), source loaders
  (`[sources.py:register_source_loader]`, `[sources.py:register_default_source_loader]`), and modality-scoped presets
  (`[presets.py:register_preset]`) are guarded by module locks and raise `ValueError` on duplicate registration; lookups
  match exact keys before longest registered prefix (`[registry.py:get_dataset_info]`, `[adapters.py:get_adapter]`,
  `[sources.py:get_source_loader]`, `[presets.py:_find_preset]`).
- **Pipeline & Preset Immutability**: `DatasetInfo`, `ResolvedPreset`, `ExecutedConfig`, `AugmentationMetadata`,
  `InventoryAsset`, `InventoryRecord`, `InventorySource`, `InventoryFilter`, `AdmittedInventory`, `ReplayState`, and
  `ReplayEpoch` are frozen dataclasses, and `DataPipeline`, preset helpers, and `ExecutedConfig` deep-copy configuration
  dicts across caller boundaries (`[registry.py:DataPipeline]`, `[presets.py:get_dataset_presets]`,
  `[executed_config.py:ExecutedConfig]`, `[config_resolution.py:resolve_callable_config]`).
- **Canonical JSON & Hashing**: `canonical_config_json` serializes UTF-8 sorted-key compact JSON with `allow_nan=False`
  (rejecting `NaN`/`Inf` and raising `TypeError` on non-string mapping keys); `_json_bytes` serializes UTF-8 sorted-key
  compact JSON with `allow_nan=False`; `MetadataSidecar` normalizes mapping keys via `str(key)` in `json_value` and
  writes sorted-key JSONL with `allow_nan=False`; `preset_hash` hashes sorted-key compact `canonical_preset_json` to 16
  hex chars; and `stable_int64_hash` produces process-stable integers in `[0, 2**63)`
  (`[executed_config.py:canonical_config_json]`, `[presets.py:preset_hash]`, `[metadata.py:json_value]`,
  `[metadata.py:MetadataSidecar.write_jsonl]`, `[metadata.py:stable_int64_hash]`, `[inventory.py:_json_bytes]`).
- **Cache Safety & Protected Cache Quotas**: `cache_model_inputs=True` with active augmentation requires
  `allow_train_model_input_cache=True` (and is forbidden under `CachePolicy`, `finalize_epoch`, and
  `load_replay_epoch`); `protected_cache` requires `callbacks_are_deterministic=True`, known-rank `tf.TensorSpec`
  leaves, matching `justdata.cache.v1` SHA-256 configuration fingerprints and `records.tfrecord` SHA-256 digests, and
  incremental plus final enforcement of `max_examples` and `max_bytes` (`[loader.py:_prepare_ds]`,
  `[cache.py:CachePolicy]`, `[cache.py:inspect_cache]`, `[cache.py:protected_cache]`).
- **Metadata Mode & Sidecar Integrity**: Sidecar usage (`sidecar_metadata_path` XOR `metadata_sidecar`) requires
  `metadata_mode == "numeric_only"`, rejects reserved transport keys (`row_id`, `row_fingerprint`, `view_id`,
  `view_fingerprint`), wraps sample tensors in `tf.control_dependencies` so `tf.data` cannot prune sidecar
  `tf.py_function` ops, enforces strictly positive `view_id` (`0` reserved for no string view), rejects orphan views,
  and validates 4x`int64` SHA-256 fingerprints (`[finalization.py:_validate_metadata_options]`,
  `[metadata.py:_TRANSPORT_FIELDS]`, `[metadata.py:_attach_metadata]`, `[metadata.py:MetadataSidecar]`,
  `[metadata.py:verify_sidecar_keys]`).
- **Inventory Snapshot Verification**: `admit_inventory` verifies every requested asset's 64-hex SHA-256 digest before
  invoking `reader`, `adapter`, `_tensorize`, and `validate`, enforces globally unique `record_id`s and
  `metadata.example_id` alignment, freezes serialized TFRecords prior to `selection.predicate`, and requires 1-to-1
  single-threaded readback reconciliation in `_verify_rows` before atomically publishing `manifest.json`;
  `open_inventory` re-verifies all artifact digests, schema, report partitions, and row order
  (`[inventory.py:_preflight]`, `[inventory.py:admit_inventory]`, `[inventory.py:_check_report]`,
  `[inventory.py:_verify_rows]`, `[inventory.py:open_inventory]`).
- **Replay Qualification & Cursor Monotonicity**: `load_replay_epoch` requires an `AdmittedInventory`, a `DataPipeline`
  with `ExecutedConfig` v1 resolution, `callbacks_are_stateless=True`, no `drop_remainder` outside `"train"`, and exact
  runtime/version/digest parity on resume (`ReplayState` `justdata.replay.v1`), asserting total batch cardinality before
  `.skip(cursor)` and enforcing monotonic `with_next_batch` cursor advancement (`[replay.py:ReplayState]`,
  `[replay.py:load_replay_epoch]`).
- **Feature Statistics Semantics**: `make_stats_iterator` disables augmentation even if `augment=True` (retaining
  `groupby` metadata keys under `numeric_only`), and `compute_feature_stats` accumulates population mean/std in
  `float64` via Chan's parallel merge before returning `tf.float32` tensors (`[stats.py:make_stats_iterator]`,
  `[stats.py:_apply_stats_metadata_mode]`, `[stats.py:compute_feature_stats]`).

## Side Effects

- **In-Memory Global Registry & Lock Mutations**: `register_dataset`, `register_pipeline`, `register_adapter`,
  `register_source_loader`, `register_default_source_loader` (invoked at import time by
  `[sources.py:load_tfds_splits]`), `register_preset`, and `_path_lock` acquire module-level `threading.Lock` instances
  and mutate module-global dictionaries (`[registry.py:_REGISTRY_LOCK]`, `[adapters.py:_ADAPTER_LOCK]`,
  `[sources.py:_SOURCE_LOCK]`, `[presets.py:_PRESET_LOCK]`, `[metadata.py:_path_lock]`).
- **Callable Attribute Mutation**: `attach_augmentation_metadata` mutates `augmentation_metadata`, `name`, `domain`,
  `is_training_only`, and `requires_labels` in place on the target function
  (`[augmentations.py:attach_augmentation_metadata]`).
- **Process-Global TensorFlow Runtime & Stateful RNG**: `configure_tensorflow_cpu` mutates global TensorFlow visible GPU
  devices and intra/inter-op thread pools and initializes logical devices (`[execution.py:configure_tensorflow_cpu]`);
  `_prepare_ds` allocates and advances stateful `tf.random.Generator` variables (`[loader.py:_prepare_ds]`), while
  `finalize_dataset` advances a caller-supplied `rng` via `rng.make_seeds(1)` when late augmentation is enabled
  (`[finalization.py:finalize_dataset]`).
- **Source Fetching, `CachePolicy` State & Cache Disk I/O**: `fetch_ds` and `load_tfds_splits` read or download external
  datasets under `~/.cache/justdata` (resolving user home via `Path.expanduser`/`Path.home`); `load_ds`,
  `finalize_dataset`, `protected_cache`, and `inspect_cache` read/write TFRecord and `tf.data` cache files, mutate
  caller `CachePolicy._paths`/`_failures` dicts in place, lock directories (`root.mkdir()`), write
  `schema.json`/`failure.json`, and atomically swap `manifest.pending` to `manifest.json` (`[loader.py:fetch_ds]`,
  `[sources.py:load_tfds_splits]`, `[sources.py:source_cache_dir]`, `[cache.py:protected_cache]`,
  `[cache.py:inspect_cache]`, `[_records.py:file_sha256]`, `[_records.py:read_examples]`).
- **Metadata Sidecar Disk I/O**: `MetadataSidecar.write_jsonl` (eagerly), `attach_sidecar_writer` (during graph
  construction and `tf.py_function` execution), and `attach_sidecar_views` (inside `tf.py_function`) create parent
  directories, write sibling temporary files (`NamedTemporaryFile`), `os.fsync` file and directory descriptors, and
  atomically `os.replace` JSONL sidecars, while `MetadataSidecar.read_jsonl` and `verify_sidecar_keys` read JSONL
  sidecars from disk (`[metadata.py:MetadataSidecar.write_jsonl]`, `[metadata.py:MetadataSidecar.read_jsonl]`,
  `[metadata.py:attach_sidecar_writer]`, `[metadata.py:attach_sidecar_views]`, `[metadata.py:verify_sidecar_keys]`).
- **Inventory & Replay Snapshot I/O**: `admit_inventory` creates `snapshot_dir`, reads source asset bytes, writes
  `records.tfrecord`, `schema.json`, and `report.json`, atomically replaces `manifest.pending` to `manifest.json`, and
  deletes `snapshot_dir` via `shutil.rmtree` on pre-publication failure; `open_inventory`, `load_inventory`, and
  `load_replay_epoch` read and hash snapshot artifacts and query Python/TF/`justdata` runtime versions
  (`[inventory.py:admit_inventory]`, `[inventory.py:open_inventory]`, `[inventory.py:load_inventory]`,
  `[replay.py:load_replay_epoch]`, `[replay.py:_runtime]`).
- **Eager Dataset Consumption & Logging**: `groupby_metadata`, `MetadataSidecar.from_inventory`, `protected_cache` (when
  `materialization="eager"`), `_verify_rows` (invoked by `admit_inventory`, `open_inventory`, and `load_replay_epoch`),
  and `compute_feature_stats` eagerly iterate `tf.data.Dataset` streams; `fetch_ds`, `_prepare_ds`, and
  `make_stats_iterator` emit diagnostics via `loguru.logger` (`[filters.py:groupby_metadata]`,
  `[metadata.py:MetadataSidecar.from_inventory]`, `[cache.py:protected_cache]`, `[inventory.py:_verify_rows]`,
  `[stats.py:compute_feature_stats]`, `[loader.py:fetch_ds]`, `[stats.py:make_stats_iterator]`).

## Verification

- Run: `uv run pytest tests/test_cross_modal_parity.py`

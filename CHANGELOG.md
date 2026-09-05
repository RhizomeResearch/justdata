# Changelog

## 1.1.0 — 2026-09-05

This release adds reusable seeded training epochs, more vision corruption and
source-metadata options, and performance improvements across vision, acoustic,
and shared data pipelines.

### Added

- `tools["finalize_epoch"]` for datasets prepared with
  `load_ds(..., return_raw_ds=True)`. Reuse loader and graph preparation while
  choosing an explicit seed for each complete epoch. With `deterministic=True`,
  repeating a seed reproduces augmentation and shuffle independently of earlier
  iterator consumption. See [reusable epochs](README.md#reusable-explicitly-seeded-epochs).
- Optional `augment_is_stateless=True` on `finalize_epoch` to parallelize
  per-sample augmentation while retaining output order and indexed seeds with
  `deterministic=True`. The callback must depend only on its sample and supplied seed, without
  stateful RNG, mutable state, or side effects. The default remains `False`.
- `load_ds` concurrency controls: `map_parallel_calls`,
  `private_threadpool_size`, and `max_intra_op_parallelism`. Omitted values retain
  TensorFlow defaults; deterministic standard augmentation remains serial unless
  explicitly enabled through the stateless epoch option.
- Five versioned TensorFlow vision corruptions: `gaussian_blur`,
  `gaussian_noise`, `jpeg_compression`, `contrast_reduction`, and
  `brightness_reduction`. The `apply_corruption` API takes an exact operator
  version, severity, and stateless seed. Immutable descriptors expose numerical
  and seed contracts. Mini-C supports these names and adds corruption identity,
  version, domain, and hash metadata. See [vision corruptions](docs/vision.md#10-versioned-image-corruptions).
- Classification `crop_kwargs` and the `random_resized_hvflip` strategy, combining
  a random resized crop with independently seeded horizontal and vertical flips.
  Existing presets retain their hashes and crop behavior.
- Optional WILDS FMoW `source_metadata=location_id,timestamp`, aligned through the
  public-to-raw index mapping and validated against authoritative source fields.
  Requested strings appear under `metadata["wilds_source"]`; omitting the option
  leaves the source schema unchanged. See [WILDS sources](docs/vision.md#5-wilds-source-loading).

### Performance

- Tensorize materialized local audio, DCASE, and Zenodo image records to remove
  Python generator overhead during iteration. Media reading and decoding remain
  lazy, and conversion failures preserve iteration-time validation errors.
- Avoid extra tensor loops for single-view vision evaluation and mono acoustic
  convolution, retaining the existing multi-view and multichannel paths.
- Reduce NumPy allocations in statistics accumulation and Hugging Face waveform
  conversion without changing arithmetic or input ownership.
- Apply spectrogram augmentation to feature tensors without copying unchanged
  waveform and metadata fields through per-sample maps. Preserve shape validation
  and debug metadata behavior.
- Use scatter-based patchout masks for large grids to avoid large pairwise
  comparison matrices; retain the original path for small grids.
- Run eligible Mixup and CutMix batches directly in NCHW when random erasing is
  disabled. Broadcast random-erasing masks across channels, and skip solarization
  pixel arithmetic when its probability gate is off.
- Fuse safe metadata filtering into postprocessing for `numeric_only` and `none`.
  Preserve separate stages where required for sidecar writers, hooks, or errors
  from computed fields that are subsequently discarded.
- Add standalone source/evaluation and training benchmarks with output hashes.
  All 29 benchmark cases matched their respective saved baseline outputs.

Representative CPU measurements from the September audit:

| Workload | Before | After |
| --- | ---: | ---: |
| 257 PNG records, including decoding | 176 ms | 89 ms |
| Spectrogram masking, 32 rows retaining 320,000-sample waveforms | 13.12 ms | 7.57 ms |
| Patchout, 12,800 positions and 1,000 dropped positions | 3.20 ms | 1.18 ms |
| NCHW CutMix, 32 × 224 × 224 RGB batch | 14.31 ms | 8.44 ms |
| Seeded 512-image epoch, serial to four augmentation workers | 1,430 ms | 549 ms |

These measure CPU input-pipeline work, not end-to-end training or GPU throughput.
Record tensorization trades additional construction work for faster iteration.
See [benchmark methodology and results](benchmarks/README.md) for environments,
baselines, memory measurements, and compatibility checks.

### Fixed

- Apply constant fill directly in geometric augmentation operations, including
  rotation, translation, and shear, without temporary alpha-channel wrapping.
- Honor `map_parallel_calls` when writing metadata sidecars.

### Maintenance and compatibility

- Consolidate shared acoustic manifest, signal, RNG, augmentation, and padding
  helpers, and simplify vision preset and augmentation internals.
- Retain the fixed 14-operation automatic-augmentation pool. The compatibility
  argument `cutout_const` remains accepted and has no effect.
- Preserve legacy Mini-C `noise`, `blur`, `weather`, and `digital` behavior.
  Corruption operator version `1.0.0` is independent of this package release.
- The performance changes preserve existing arithmetic, RNG assignment, pipeline
  stage order, and default concurrency. The new epoch parallelism is opt-in.
- Use CPU PyTorch wheels for uv development/golden dependencies to reduce CI
  disk usage; repair direnv activation and run GitLab lint in a compatible image
  using a dedicated lint dependency group.

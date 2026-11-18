# justdata.vision

## Overview

Computer vision modality package providing dataset registrations, remote/cached vision source loaders (`hf:`, `zenodo:`,
`wilds:`), modality-scoped recipe presets, registered task pipelines (`vision/classification`, `vision/segmentation`,
`vision/panoptic_segmentation`), recorded-geometry dense/panoptic view replay and score restoration, deterministic
evaluation views, Mini-C corruption dataset generation, colorspace/filtering/Poisson/compositing primitives, and child
subpackages for augmentations, corruptions, target encodings, and task stages.

## Interface

- `[__init__.py:__all__]`: Re-exports `augmentations`, `corruptions`, `datasets`, `pipelines`, `presets`, and `sources`
  (importing `justdata.vision` triggers built-in vision registrations).
- `[datasets.py:register_dataset]`: Registers built-in `modality="vision"` datasets (`cifar10`, `cifar100`, `imagenet`,
  `imagenette`, `stanford_dogs`, `zenodo:`, `kitti_road`, `wilds:camelyon17`, `wilds:fmow`, `wilds:iwildcam`,
  `wilds:rxrx1`).
- `[sources.py:load_huggingface_vision_splits]`, `[sources.py:load_zenodo_imagefolder_splits]`,
  `[sources.py:load_wilds_vision_splits]`: Registered source loaders for `hf:`, `zenodo:`, and `wilds:` prefixes.
- `[presets.py:register_preset]`, `[presets.py:get_dataset_presets]`, `[presets.py:get_resolved_preset]`,
  `[presets.py:merge_with_presets]`, `[presets.py:ResolvedPreset]`: Vision-scoped (`modality="vision"`) preset
  registration, lookup, resolution, and merging across 20 built-in presets (`_default`, `imagenet_resnet`,
  `imagenet_a1`..`a3`, `cifar`, `cifar100`, `dinov2`, `kitti_road`, `segmentation_a1`..`a3`, and standard/strong
  `wilds:*` presets).
- `[pipelines.py:default_classification_pipeline]`, `[pipelines.py:default_segmentation_pipeline]`,
  `[pipelines.py:default_panoptic_pipeline]`: Registered `PipelineFuncs` builders for `'vision/classification'`,
  `'vision/segmentation'` (legacy and recorded-geometry dense modes), and `'vision/panoptic_segmentation'`.
- `[stages.py:EvalViewConfig]`, `[stages.py:apply_eval_views]`, `[stages.py:normalize_image_format]`,
  `[stages.py:resize_and_normalize]`, `[stages.py:recorded_view_augmentations]`, `[stages.py:finish_recorded_view]`,
  `[stages.py:apply_crop_strategy]`, `[stages.py:co_transform]`: Shared stage helpers for RGB format normalization,
  deterministic multi-crop/flip evaluation views, crop dispatch, and recorded-geometry wrapping.
- `[geometry.py:DenseGeometryConfig]`, `[geometry.py:PanopticGeometryConfig]`, `[geometry.py:sample_dense_geometry]`,
  `[geometry.py:sample_panoptic_geometry]`, `[geometry.py:center_dense_crop]`, `[geometry.py:validate_dense_sample]`,
  `[geometry.py:replay_dense_geometry]`, `[geometry.py:replay_panoptic_geometry]`, `[geometry.py:decode_mask_png]`,
  `[geometry.py:dense_views_from_files]`, `[geometry.py:restore_dense_scores]`,
  `[geometry.py:restore_dense_predictions]`: Recorded-geometry sampling, deterministic replay, exact 8/16-bit PNG mask
  decoding, and evaluation score/logit restoration.
- `[transforms.py:nhwc_to_nchw]`, `[transforms.py:normalize]`, `[transforms.py:resize_short_side]`,
  `[transforms.py:center_crop]`, `[transforms.py:resize_image]`, `[transforms.py:pad_to_patch_multiple]`,
  `[utils.py:to_4d]`, `[utils.py:from_4d]`, `[utils.py:gaussian_filter2d]`: Deterministic spatial/layout transforms and
  low-level tensor utilities.
- `[minic.py:create_minic_datasets]`: Builds deterministic per-sample-seeded corrupted evaluation datasets and returns
  `(dataset_or_list, n_batches)`.
- `[colorspace.py:srgb_to_linear]`, `[colorspace.py:linear_to_srgb]`, `[colorspace.py:rgb_to_lab]`,
  `[colorspace.py:lab_to_rgb]`, `[colorspace.py:estimate_illuminant]`, `[filters.py:reflect_indices]`,
  `[filters.py:reflect_pad_hwc]`, `[filters.py:gaussian_kernel_1d]`, `[filters.py:depthwise_hwc]`,
  `[filters.py:gaussian_blur_hwc]`, `[filters.py:correlate_hwc]`, `[filters.py:splat_points]`, `[horizon.py:Horizon]`,
  `[horizon.py:horizon_coordinates]`, `[horizon.py:horizon_pixel_coordinates]`: Colorspace, Grey-Edge illuminant
  estimation, reflect-101/FFT filtering, bilinear splatting, and horizon geometry primitives.
- `[poisson.py:solve_poisson_dirichlet]`, `[poisson.py:clone_bounds]`, `[poisson.py:clone_region]`,
  `[poisson.py:seamless_clone]`, `[compositing.py:Instance]`, `[compositing.py:PlacementRule]`,
  `[compositing.py:BlendingRule]`, `[compositing.py:CompositeDescriptor]`, `[compositing.py:PasteRecord]`,
  `[compositing.py:CompositeResult]`, `[compositing.py:composite_instances]`, `[compositing.py:BLENDING_MODES]`,
  `[compositing.py:PLACEMENT_SCALES]`: 2D DST-I Poisson seamless cloning and eager Philox-seeded object/null-paste scene
  compositing (`COMPOSITE_VERSION = "2.1.0"`, `LABEL_ALPHA_THRESHOLD = 0.5`, `REFERENCE_WIDTH = 1024`).
- Subdirectory [`augmentations`](./augmentations/EGM.md): Stateless seeded per-sample/batch vision augmentations,
  multi-crop builders, and crop/augment strategy registries (`[__init__.py:augmentations]`).
- Subdirectory [`corruptions`](./corruptions/EGM.md): Versioned discrete and continuous corruption descriptors,
  registries, stacks, and 19 v`1.0.0` corruption implementations (`[__init__.py:corruptions]`).
- Subdirectory [`encodings`](./encodings/EGM.md): Deterministic bounding-box heatmap and semantic/panoptic dense target
  encoders (`[geometry.py:validate_dense_sample]`, `[pipelines.py:default_panoptic_pipeline]`).
- Subdirectory [`tasks`](./tasks/EGM.md): Stage-builder factories for classification, semantic/dense segmentation,
  panoptic segmentation, and depth (`[pipelines.py:default_classification_pipeline]`).

## Invariants

- `[__init__.py:__all__]`, `[presets.py:register_preset]`, `[pipelines.py:_VISION_TOP_LEVEL]`: Vision datasets, presets,
  and pipelines must be registered via `import justdata.vision` before resolution; preset helpers bind
  `modality="vision"` and deep-copy configs while deriving `model_input` contracts
  (`[presets.py:_with_model_input_contract]`); all pipeline resolvers reject unknown keys (`reject_unknown_keys`);
  recorded-geometry dense and panoptic resolvers (`[pipelines.py:_resolve_dense_segmentation_config]`,
  `[pipelines.py:_resolve_panoptic_config]`) reject mutually non-`None` `color_jitter_kwargs` and `photometric_kwargs`
  (`[pipelines.py:_resolve_color_options]`) and disallow `augment_eval=True` or non-empty
  `preproc_kwargs`/`aug_kwargs`/`laug_kwargs`, while legacy segmentation rejects `color_jitter_kwargs` /
  `photometric_kwargs` without `geometry_kwargs` (`[pipelines.py:_reject_recorded_geometry_options]`) and classification
  rejects both as unknown keys.
- `[sources.py:load_huggingface_vision_splits]`, `[sources.py:_validate_zenodo_download_url]`,
  `[sources.py:_extract_archive]`, `[sources.py:_validate_image_file]`, `[sources.py:_validate_wilds_splits]`,
  `[sources.py:_parse_wilds_source_metadata]`: `hf:` datasets must expose `'image'` and `'label'` columns; `zenodo:`
  enforces `https://zenodo.org` (port `None`/`443`, no credentials), `<= 20 GiB` archives with checksum verification,
  preflighted ZIP/TAR extraction (`<= 100,000` members, `<= 100 GiB`, no symlinks/hardlinks/traversal), and Pillow
  header checks (`<= 256 MiB`, `<= 32,768` px/side, `<= 64,000,000` pixels); `wilds:` supports only
  `{'camelyon17', 'fmow', 'iwildcam', 'rxrx1'}`, forbids mixing labeled and unlabeled splits, and restricts
  `source_metadata` to `'fmow'` `('location_id', 'timestamp')` with explicit UTC/timezone offsets.
- `[stages.py:normalize_image_format]`, `[transforms.py:normalize]`, `[stages.py:_per_image_channel_standardize]`,
  `[transforms.py:nhwc_to_nchw]`, `[stages.py:EvalViewConfig.__post_init__]`: `normalize_image_format` outputs 3-channel
  HWC RGB without rescaling values; both `'mean_std'` and `'per_image'` normalization divide `float32`-cast inputs by
  `255.0` (expecting `[0, 255]` inputs); `nhwc_to_nchw` requires known static rank 3 or 4 (`ValueError` when rank is
  unknown, `NotImplementedError` for other static ranks); `EvalViewConfig` requires positive `image_size`/`num_crops`
  and `mode` in `{'center_crop', 'resize_crop', 'multi_crop'}`.
- `[geometry.py:DenseGeometryConfig.__post_init__]`, `[geometry.py:sample_dense_geometry]`,
  `[geometry.py:replay_dense_geometry]`, `[geometry.py:decode_mask_png]`, `[geometry.py:restore_dense_scores]`: Require
  exactly one of `train_resize_range` or `train_scale_range`, bottom-right-only patch-multiple padding, version 1 or 2
  numeric geometry records (`[geometry.py:_check_record]`), exact 8/16-bit grayscale PNG headers in `decode_mask_png`,
  and uncropped/unflipped evaluation records (`is_training == False`) in `restore_dense_scores`.
- `[minic.py:_resolve_corruption_version]`, `[minic.py:create_minic_datasets]`: Default legacy Mini-C names
  `{'blur', 'digital', 'noise', 'weather'}` to `'1.0.0'`, require explicit `corruption_versions` when multiple versions
  exist, validate `severity` against `descriptor.severity_values`, and derive deterministic `int32[2]` seeds per
  `(sample_index, corruption_index)`.
- `[compositing.py:composite_instances]`, `[compositing.py:_validated_scene]`, `[poisson.py:solve_poisson_dirichlet]`,
  `[poisson.py:seamless_clone]`, `[horizon.py:Horizon.__post_init__]`, `[colorspace.py:estimate_illuminant]`:
  `composite_instances` requires eager execution (`RuntimeError` inside `tf.function`), `uint8` HWC RGB scenes, `int32`
  shape-`(2,)` seeds, and non-`None` `Horizon` when `placement.scale == "horizon"`; `Horizon` requires `camera_height_m`
  and `focal_length_px` to be both `None` or both `> 0`; `estimate_illuminant` requires static rank-3 width;
  `solve_poisson_dirichlet` solves interior pixels in `tf.float64` preserving the 1-pixel Dirichlet boundary, and
  `seamless_clone` raises `ValueError` if the offset ROI exits `destination`.

## Side Effects

- `[__init__.py:__all__]`, `[datasets.py:register_dataset]`, `[pipelines.py:default_classification_pipeline]`,
  `[pipelines.py:default_segmentation_pipeline]`, `[pipelines.py:default_panoptic_pipeline]`,
  `[presets.py:register_preset]`, `[sources.py:load_zenodo_imagefolder_splits]`: Importing `justdata.vision` mutates the
  global core dataset, pipeline, preset (`modality="vision"`), source-prefix (`zenodo:`, `hf:`, `wilds:`), augmentation,
  and corruption registries; `default_classification_pipeline` also mutates non-empty caller `postproc_kwargs` via
  `setdefault` when MixUp/CutMix is enabled.
- `[sources.py:_fetch_zenodo_record]`, `[sources.py:_download_file]`, `[sources.py:_prepare_zenodo_archive]`,
  `[sources.py:load_huggingface_vision_splits]`, `[sources.py:load_wilds_vision_splits]`,
  `[sources.py:_wilds_fmow_datetime_compat]`: Perform HTTPS requests and third-party dataset downloads, acquire
  `filelock.FileLock` locks, stage and atomically `os.replace` cached archives/extractions, emit `loguru` info/warning
  logs, and temporarily wrap `pandas.to_datetime` around `'fmow'` loading; `[geometry.py:dense_views_from_files]` and
  `[sources.py:_records_to_vision_dataset]` perform lazy `tf.io.read_file` disk reads during dataset iteration.
- `[stages.py:normalize_image_format]`, `[stages.py:finish_recorded_view]`, `[transforms.py:resize_image]`,
  `[poisson.py:seamless_clone]`: Call `Tensor.set_shape` in-place on sample tensors, populate `@tf.function` trace
  caches, and allocate fresh host NumPy buffers in `seamless_clone` without mutating caller mask/destination arrays.

## Verification

- Run: `uv run pytest tests/test_pipelines.py`

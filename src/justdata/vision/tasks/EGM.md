# justdata.vision.tasks

## Overview

Task-specific stage builders assembling `(preprocess, augment, late_augment, postprocess)` callables for image
classification (supervised and self-supervised multi-crop), semantic and recorded-geometry dense segmentation, panoptic
segmentation, and depth map preparation.

## Interface

- `[classification.py:make_preprocessing]`, `[classification.py:make_augmentations]`,
  `[classification.py:make_late_augmentations]`, `[classification.py:make_postprocessing]`,
  `[classification.py:mixup_cutmix]`: Stage factories for supervised (`mode="sl"`) and self-supervised (`mode="ssl"`)
  classification pipelines plus re-exported `mixup_cutmix`.
- `[depth.py:prepare_depth]`: `@tf.function` resizing depth maps to `[image_size, image_size]`, clipping to
  `[min_depth, max_depth]`, and optionally min-max normalizing to `[0, 1]` in `tf.float32`.
- `[panoptic.py:make_panoptic_preprocessing]`, `[panoptic.py:make_panoptic_augmentations]`,
  `[panoptic.py:make_panoptic_postprocessing]`: Recorded-geometry panoptic pipeline stage factories (exported in
  `[panoptic.py:__all__]`).
- `[segmentation.py:make_preprocessing]`, `[segmentation.py:make_augmentations]`,
  `[segmentation.py:make_late_augmentations]`, `[segmentation.py:make_postprocessing]`: Legacy/standard semantic
  segmentation stage factories supporting joint image-mask spatial transforms and ViT `patch_align` evaluation padding.
- `[segmentation.py:make_dense_preprocessing]`, `[segmentation.py:make_dense_augmentations]`,
  `[segmentation.py:make_dense_postprocessing]`: Recorded-geometry dense semantic segmentation stage factories with
  optional `with_semantic_targets` emission.

## Invariants

- `[classification.py:make_augmentations]`: Returns an identity `no_op` closure immediately when `enable=False`; when
  `enable=True`, rejects reserved keys `{'interpolation', 'pad_mode', 'padding', 'seed', 'size'}` in `crop_kwargs`
  (`ValueError`), eagerly resolves `crop_type` and `augment_type`, splits `seed` into two stateless sub-seeds, validates
  `mode in ('sl', 'ssl')` at call time, and in `'ssl'` mode replaces `'image'` with `'global_crops'` (and
  `'local_crops'` when `local_crops=True`).
- `[classification.py:make_late_augmentations]`: Acts as a no-op when `not enable`, `mode == "ssl"`, or
  `'image'`/`'label'` is missing; raises `ValueError` if `(mixup_alpha > 0 or cutmix_alpha > 0)` with
  `num_classes is None`; normalizes NCHW/NHWC layout transitions across `random_erasing` and `_mixup_cutmix`.
- `[classification.py:make_postprocessing]`, `[classification.py:_add_hard_label_metadata]`,
  `[classification.py:_is_dense_label_vector]`: Resolves `val_resize_size="auto"` to `int(image_size / 0.875)` during
  evaluation, routes evaluation through `apply_eval_views` when `eval_view_config` is set and `'image'` is in `sample`,
  records `metadata['hard_label']` (and optional `metadata['class_name']`) when `keep_hard_label_in_metadata`,
  `not is_training`, and `label.shape.rank in (0, None)`, and raises `ValueError` at call time if `one_hot_labels=True`,
  a key in `label_keys` is present in `sample`, and `num_classes` is `None`.
- `[segmentation.py:make_preprocessing]`, `[segmentation.py:make_augmentations]`,
  `[segmentation.py:make_postprocessing]`: `make_augmentations` indexes `sample['mask']` directly (`mask_key` in
  `make_preprocessing` is unused), shares crop sub-seed `seeds[0]` between `image` and `mask` (forcing `'nearest'` mask
  interpolation), forwards `segmentation_mask` with `segmentation_fill_value=mask_fill_value` for
  `rand_augment`/`trivial_augment`/`trivial_augment_wide` (and `**ta_kwargs` for other `augment_type` strategies), and
  requires non-`None` `normalization_params` during normalized `patch_align` evaluation.
- `[segmentation.py:make_dense_preprocessing]`, `[segmentation.py:make_dense_augmentations]`,
  `[segmentation.py:make_dense_postprocessing]`, `[panoptic.py:make_panoptic_preprocessing]`,
  `[panoptic.py:make_panoptic_augmentations]`, `[panoptic.py:make_panoptic_postprocessing]`: Validate
  categorical/panoptic annotations during preprocessing (retaining `original_*` fields when requested), sample seeded
  geometry (`is_training=True`) in `recorded_view_augmentations`, apply deterministic geometry (`is_training=False`) in
  evaluation postprocessing prior to `finish_recorded_view`, and optionally attach encoded `'targets'`.

## Side Effects

- `[classification.py:make_augmentations]`: When `enable=True`, mutates caller-supplied non-empty `gc_kwargs` and
  `lc_kwargs` dictionaries in place via `.setdefault('size', ...)` and `.setdefault('scale', ...)` when those keys are
  omitted.
- `[classification.py:make_postprocessing]`, `[panoptic.py:make_panoptic_preprocessing]`,
  `[segmentation.py:make_dense_preprocessing]`: Stage callables return shallow-copied sample/metadata dicts without
  mutating input `sample` dictionaries in place; `[depth.py:prepare_depth]` caches traced `@tf.function` graphs in
  memory with no disk/network I/O or global registry mutations.

## Verification

- Run: `uv run pytest tests/test_pipelines.py`

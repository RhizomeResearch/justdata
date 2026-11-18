# justdata.vision.augmentations

## Overview

Stateless seeded per-sample and batch-level vision augmentations, self-supervised multi-crop builders, and thread-safe
registries for image crop and policy augmentation strategies.

## Interface

- `[__init__.py:__all__]`: Re-exports submodules `auto`, `color`, `composed`, `geometric`, `mixing` and registry
  functions `register_crop_strategy`, `register_augment_strategy`, `get_crop_strategy`, `get_augment_strategy`,
  `list_crop_strategies`, `list_augment_strategies`, `get_crop_strategy_metadata`, `get_augment_strategy_metadata`,
  `list_crop_strategy_metadata`, `list_augment_strategy_metadata`.
- `[registry.py:CropStrategyFn]`, `[registry.py:AugmentStrategyFn]`: Callable protocols
  `(image: tf.Tensor, *, seed: tf.Tensor, **kwargs) -> tf.Tensor` for registered crop and augmentation strategies.
- `[registry.py:register_crop_strategy]`, `[registry.py:register_augment_strategy]`: Decorators registering named crop
  or policy strategies with `AugmentationMetadata(domain="image", is_training_only=..., requires_labels=...)`.
- `[registry.py:get_crop_strategy]`, `[registry.py:get_augment_strategy]`, `[registry.py:get_crop_strategy_metadata]`,
  `[registry.py:get_augment_strategy_metadata]`, `[registry.py:list_crop_strategies]`,
  `[registry.py:list_augment_strategies]`, `[registry.py:list_crop_strategy_metadata]`,
  `[registry.py:list_augment_strategy_metadata]`: Strategy and `AugmentationMetadata` lookup/listing functions.
- `[auto.py:RAND_AUGMENT_OPS]`, `[auto.py:RAND_AUGMENT_SPATIAL_OPS]`: Canonical 14-operation RandAugment tuple and
  5-operation spatial subset (`Rotate`, `ShearX`, `ShearY`, `TranslateX`, `TranslateY`).
- `[auto.py:rand_augment]`, `[auto.py:trivial_augment]`, `[auto.py:trivial_augment_wide]`: `@tf.function` automated
  augmentation policies supporting optional `bboxes` or `segmentation_mask` co-transforms; registered as augment
  strategies `'rand_augment'` (`[auto.py:_aug_rand_augment]`), `'trivial_augment'` (`[auto.py:_aug_trivial_augment]`),
  `'trivial_augment_wide'` (`[auto.py:_aug_trivial_augment_wide]`), plus identity `'none'` (`[auto.py:_aug_none]`).
- `[color.py:photometric_distortion]`, `[color.py:color_jitter]`, `[color.py:gaussian_blur]`, `[color.py:solarize]`:
  `@tf.function` stateless photometric transforms; `'color_jitter'` is registered via `[color.py:_aug_color_jitter]`.
- `[composed.py:create_global_crops]`, `[composed.py:create_local_crops]`: `@tf.function` multi-crop builders producing
  stacked `[crops_number, ...]` views with optional spatially aligned `image_label` or Gaussian bounding-box heatmaps.
- `[geometric.py:random_resized_crop]`, `[geometric.py:random_horizontal_flip]`, `[geometric.py:random_vertical_flip]`,
  `[geometric.py:random_crop_with_pad]`, `[geometric.py:resize_to_square]`, `[geometric.py:random_rot90]`:
  `@tf.function` geometric transforms; registered crop strategies are `'random_resized'`
  (`[geometric.py:_crop_random_resized]`), `'random_resized_hvflip'` (`[geometric.py:_crop_random_resized_hvflip]`),
  `'random_pad'` (`[geometric.py:_crop_random_pad]`), `'random_hflip'` (`[geometric.py:_crop_random_hflip]`),
  `'resize_random_hflip'` (`[geometric.py:_crop_resize_random_hflip]`), `'random_rot90_hflip'`
  (`[geometric.py:_crop_random_rot90_hflip]`), and `'none'` (`[geometric.py:_crop_none]`).
- `[mixing.py:mixup_cutmix]`, `[mixing.py:random_erasing]`: `@tf.function` 4D NHWC batch-level MixUp/CutMix
  image-and-label blending and batch-mapped (`tf.map_fn`) stateless rectangular erasing.

## Invariants

- `[registry.py:register_crop_strategy]`, `[registry.py:register_augment_strategy]`: Synchronize mutations with
  `_REGISTRY_LOCK`, enforce `domain="image"`, and raise `ValueError` citing the existing `qualified_name` on duplicate
  strategy names; unknown lookups in `[registry.py:get_crop_strategy]`, `[registry.py:get_augment_strategy]`,
  `[registry.py:get_crop_strategy_metadata]`, and `[registry.py:get_augment_strategy_metadata]` raise `ValueError`
  listing registered keys.
- `[auto.py:rand_augment]`, `[geometric.py:random_resized_crop]`, `[color.py:photometric_distortion]`,
  `[mixing.py:_mixup_cutmix]`, `[mixing.py:random_erasing]`: All stochastic operations are stateless and
  deterministically derived from explicit `seed` tensors (`tf.random.split`, `stateless_split`, `stateless_uniform`,
  `stateless_normal`, `stateless_gamma`).
- `[auto.py:rand_augment]`: Raises `ValueError` if both `bboxes` and `segmentation_mask` are passed or if `exclude_ops`
  removes all candidate operations; clips non-`uint8` images to `[0.0, 255.0]` before `uint8` execution and restores
  `image.dtype` on return; `[auto.py:_apply_segmentation_geometric_op]` applies `'BILINEAR'` to images with fill `128`
  and `'NEAREST'` to masks with `segmentation_fill_value` (default `255`).
- `[color.py:_validate_photometric_params]`, `[color.py:photometric_distortion]`: Reject boolean, non-real, non-finite,
  or negative distortion magnitudes (`hue <= 0.5`, `probability` in `[0, 1]`); `photometric_distortion` enforces shape
  `[None, None, 3]`, asserts input values in `[0.0, 255.0]`, and clips `float32` output to `[0.0, 255.0]`.
- `[composed.py:create_global_crops]`: Raises `ValueError` when `bboxes` is provided without both `labels` and
  `num_classes`; shares crop and flip seeds between `image` and `image_label` to preserve spatial alignment.
- `[geometric.py:_validate_flip_probability]`, `[geometric.py:_transform_bboxes_for_crop]`: Flip probabilities outside
  `[0.0, 1.0]` raise `ValueError`; bounding boxes use normalized `[ymin, xmin, ymax, xmax]` coordinates clipped to
  `[0.0, 1.0]` with positive intersection area filtering.
- `[mixing.py:_mixup_cutmix]`: Blends images in `tf.float32` and restores `images.dtype`, prepares labels via
  `prepare_labels_for_mixing` even when mixing is skipped, applies `label_smoothing` only when
  `label_mode == "single_label"`, and recomputes CutMix `lam_adjusted` from clipped integer box area.

## Side Effects

- `[registry.py:_CROP_STRATEGIES]`, `[registry.py:_AUGMENT_STRATEGIES]`, `[registry.py:_CROP_STRATEGY_METADATA]`,
  `[registry.py:_AUGMENT_STRATEGY_METADATA]`, `[registry.py:_REGISTRY_LOCK]`: Maintain module-level mutable strategy and
  metadata dictionaries (synchronizing registration writes with `threading.Lock()`) and attach `AugmentationMetadata`
  onto callables during registration and in `get_crop_strategy` / `get_augment_strategy` lookups.
- `[__init__.py:__all__]`, `[auto.py:_aug_rand_augment]`, `[color.py:_aug_color_jitter]`,
  `[geometric.py:_crop_random_resized]`: Importing `justdata.vision.augmentations` eagerly imports `auto`, `color`,
  `composed`, `geometric`, and `mixing`, registering built-in crop strategies (`random_resized`,
  `random_resized_hvflip`, `random_pad`, `random_hflip`, `resize_random_hflip`, `random_rot90_hflip`, `none`) and
  augment strategies (`rand_augment`, `trivial_augment`, `trivial_augment_wide`, `color_jitter`, `none`).

## Verification

- Run: `uv run pytest tests/test_pipelines.py`

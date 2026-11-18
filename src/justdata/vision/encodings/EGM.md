# justdata.vision.encodings

## Overview

Deterministic TensorFlow target encoders converting bounding boxes into spatial class heatmaps and converting
categorical semantic or panoptic segmentation maps into validated fixed-capacity dense target dictionaries.

## Interface

- `[__init__.py:__all__]`: Re-exports `semantic_map_to_targets`, `with_semantic_targets`, `decode_panoptic_rgb`,
  `panoptic_map_to_semantic`, `panoptic_map_to_targets`, `validate_panoptic_sample`, and `with_panoptic_targets`.
- `[heatmaps.py:bboxes_to_heatmaps]`: `@tf.function` converting normalized `[ymin, xmin, ymax, xmax]` boxes `(N, 4)` and
  integer class labels `(N,)` into clipped `[0.0, 1.0]` binary `tf.float32` heatmaps `(H, W, num_classes)`.
- `[heatmaps.py:bboxes_to_gaussian_heatmaps]`: `@tf.function` converting normalized `[ymin, xmin, ymax, xmax]` boxes
  `(N, 4)` and labels `(N,)` into box-bounded 2D Gaussian `tf.float32` heatmaps `(H, W, num_classes)`.
- `[panoptic_targets.py:decode_panoptic_rgb]`: Decodes `(H, W, 3)` `tf.uint8` COCO-format RGB panoptic masks into
  `(H, W)` `tf.int64` segment IDs (`R + 256*G + 65536*B`).
- `[panoptic_targets.py:validate_panoptic_sample]`: Validates `image`, `panoptic_mask`, `segments`, and optional
  `annotation_valid_mask`, returning a sample dict with `int64` `panoptic_mask` and ID-sorted segment table padded to
  `max_segments`.
- `[panoptic_targets.py:panoptic_map_to_semantic]`: Projects panoptic segment IDs and `segments` table to `tf.int32`
  semantic class labels with `ignore_value` fill for void, invalid, and (optionally) crowd pixels.
- `[panoptic_targets.py:panoptic_map_to_targets]`, `[panoptic_targets.py:with_panoptic_targets]`: Convert panoptic maps
  and segment tables into fixed-capacity (`max_segments`) target dicts (`class_ids`, `class_indices`, `segment_ids`,
  `is_thing`, `masks`, `target_valid_mask`, `pixel_valid_mask`, `num_targets`, `num_overflow_things`,
  `supervision_valid`) and attach `'targets'` onto a sample dict.
- `[semantic_targets.py:semantic_map_to_targets]`, `[semantic_targets.py:with_semantic_targets]`: Convert `HW`/`HW1`
  categorical masks (or `None` unlabeled masks) into fixed-capacity (`len(class_values)`) class-mask target dicts and
  attach `'targets'` onto a sample dict.

## Invariants

- `[heatmaps.py:bboxes_to_heatmaps]`, `[heatmaps.py:bboxes_to_gaussian_heatmaps]`: Expect normalized
  `[ymin, xmin, ymax, xmax]` boxes scaled by `(H, W)`; `bboxes_to_heatmaps` floors `(y1, x1)` and ceils `(y2, x2)`
  before skipping boxes with `y1 >= y2` or `x1 >= x2`, while `bboxes_to_gaussian_heatmaps` clamps box dimensions to
  `>= 1.0` (`sigma = size / 4.0`) and masks with inclusive `y1 <= y_grid <= y2` and `x1 <= x_grid <= x2`; both preserve
  output shape `[None, None, num_classes]` in `tf.float32`.
- `[panoptic_targets.py:decode_panoptic_rgb]`: Requires `rgb.dtype == tf.uint8` (`TypeError` otherwise) and shape
  `[None, None, 3]`, computing exact `tf.int64` IDs without floating-point intermediates.
- `[panoptic_targets.py:_values]`, `[panoptic_targets.py:_validate_map_segments]`,
  `[panoptic_targets.py:validate_panoptic_sample]`, `[panoptic_targets.py:panoptic_map_to_semantic]`: Require non-empty
  unique Python `int` `class_values` in `[0, 2**31)`, subset `thing_class_values`, `void_value` in `[0, 2**63)`,
  `max_segments > 0` (enforcing valid segment count `<= max_segments` when `max_segments` is supplied), `segments` keys
  restricted to `{"segment_ids", "category_ids", "is_crowd", "valid_mask"}` (optionally plus both `{"area", "bbox"}`),
  `len(semantic_values) == len(class_values)` in `panoptic_map_to_semantic`, integer `panoptic_mask` matching `image`
  `(H, W)`, and exact equality between non-void mask IDs and valid `segments['segment_ids']`.
- `[panoptic_targets.py:panoptic_map_to_targets]`, `[panoptic_targets.py:_things_with_own_target]`: Require
  `overflow_policy` in `('error', 'exclude_supervision', 'merge_by_category')`, rank visible non-crowd things by valid
  pixel area descending against remaining capacity `max_segments - num_present_stuff`, mask void/crowd/`!example_valid`
  pixels out of `pixel_valid_mask`, and assign `segment_ids = void_value` and `is_thing = False` to stuff and merged
  targets.
- `[semantic_targets.py:_class_contract]`, `[semantic_targets.py:_categorical_mask]`,
  `[semantic_targets.py:_assert_declared_labels]`, `[semantic_targets.py:semantic_map_to_targets]`: Require unique
  signed-`int32` Python `int` `class_values` disjoint from `ignore_value`,
  `mask.dtype.min <= ignore_value <= mask.dtype.max`, integer `HW`/`HW1` masks whose pixels all lie in
  `class_values | {ignore_value}`, and all-`False` `pixel_valid_mask` whenever `mask is None`.
- `[panoptic_targets.py:with_panoptic_targets]`, `[semantic_targets.py:with_semantic_targets]`: Raise `ValueError` if
  `'targets'` already exists in `sample` or if `'pixel_valid_mask'` is absent.

## Side Effects

- None (`[heatmaps.py:bboxes_to_heatmaps]`, `[panoptic_targets.py:with_panoptic_targets]`,
  `[semantic_targets.py:with_semantic_targets]` are pure tensor/dict transformations with no global state mutations,
  disk I/O, or network access).

## Verification

- Run: `uv run pytest tests/test_pipelines.py`

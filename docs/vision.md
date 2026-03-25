# Vision Pipelines

`justdata.vision` is the image modality package for TensorFlow-native loading,
schema normalization, augmentation, batching, metadata, presets, and corruption
benchmarks.

## 1. Why justdata owns image transforms

Vision checkpoints and training recipes assume exact preprocessing semantics:
image decoding shape, channel order, resize and crop policy, interpolation,
pixel range, normalization, label transform, tensor layout, and evaluation
views. If those steps live outside the data pipeline, training and evaluation
can silently drift from the model contract.

`justdata` owns image transforms so a preset can describe the full input
contract in one hashable object and so TensorFlow, JAX/NumPy consumers, preset
contract tests, and corruption benchmarks all use the same preprocessing path.

## 2. Vision canonical schema

Raw or adapted vision samples use these keys:

| Key | Meaning |
| :-- | :-- |
| `image` | Image tensor in HWC layout before postprocessing. |
| `label` | Optional class index, dense vector, or task target. |
| `mask` | Optional segmentation mask. |
| `depth` | Optional depth map. |
| `bboxes` | Optional object detection boxes. |
| `metadata` | Optional nested metadata such as dataset, split, filename, example id, or class name. |

Classification pipelines always preserve unknown keys unless a stage explicitly
owns them. Metadata handling is shared with acoustic through `justdata.core`.

## 3. Four vision pipeline stages

Vision pipelines use the same four callables as acoustic:

```python
(preprocess_fn, augment_fn, late_augment_fn, postprocess_fn)
```

The loader executes them as:

```text
fetch_ds -> adapter -> preprocess -> cache -> augment -> shuffle -> postprocess -> batch -> late_augment -> pad -> prefetch
```

For vision classification the stages are:

| Stage | Vision responsibility |
| :-- | :-- |
| `preprocess` | Normalize image rank and channels before cache. |
| `augment` | Training-only per-sample crop, flip, RandAugment, TrivialAugment, ColorJitter, or SSL multi-crop. |
| `postprocess` | Deterministic resize/crop for eval, tensor normalization, layout conversion, one-hot labels, and hard-label metadata. |
| `late_augment` | Training-only batch transforms such as Mixup, CutMix, and random erasing. |

## 4. How to choose a preset

Choose the preset that matches the dataset scale and model recipe:

| Use case | Prefer |
| :-- | :-- |
| CIFAR-10 or CIFAR-like 32 px classification | `cifar` through `dataset="cifar10"`. |
| CIFAR-100 32 px classification | `cifar100`. |
| Standard ImageNet-style ViT or ConvNeXt recipe | `_default` through `dataset="imagenet"`. |
| Legacy ImageNet ResNet recipe | `imagenet_resnet`. |
| ResNet Strikes Back recipes | `imagenet_a1`, `imagenet_a2`, or `imagenet_a3`. |
| DINOv2-style self-supervised multi-crop | `dinov2`. |

Use `justdata.vision.presets.get_resolved_preset(name).hash()` in experiment
metadata. A changed hash means the image preprocessing or augmentation contract
changed.

## 5. CIFAR, ImageNet, and DINOv2 preset contracts

CIFAR presets use 32 px inputs, random padded crop, horizontal flip,
TrivialAugmentWide, CIFAR-specific normalization, `bchw` model layout, and
classification labels.

ImageNet default presets use 224 px inputs, random resized crop during training,
deterministic resize and center crop during evaluation, ImageNet mean/std
normalization, `bchw` model layout, RandAugment or ColorJitter depending on the
recipe, and optional batch mixing.

ResNet Strikes Back presets (`imagenet_a1`, `imagenet_a2`, `imagenet_a3`) encode
the recipe-specific augmentation strength, Mixup/CutMix settings, validation
resize behavior, and train/eval resolution policy.

DINOv2 uses asymmetric multi-crop training with global and local crops while
keeping the downstream evaluation path deterministic.

## 6. Evaluation split safety

Vision pipelines keep training-only stochastic transforms out of validation and
test views. `load_ds(..., dataset_type="validation")` builds the pipeline with
`is_training=False`, so evaluation uses deterministic resize/crop behavior and
no Mixup, CutMix, random erasing, or training crop randomness.

When computing validation metrics or corruption benchmarks, use
`dataset_type="validation"` and `deterministic=True` if the input order must be
stable.

## 7. Metadata modes for JAX

`load_ds(..., metadata_mode=...)` is shared by vision and acoustic:

| Mode | Behavior |
| :-- | :-- |
| `full` | Keep all metadata, including strings. |
| `numeric_only` | Keep numeric metadata and remove strings from batches. |
| `none` | Drop metadata from output batches. |

Use `metadata_mode="numeric_only"` with `as_numpy=True` for JAX-friendly arrays.
If string metadata is needed for later joins, pass
`sidecar_metadata_path="metadata.jsonl"` and keep batches numeric.

## 8. Golden compatibility tests

Vision preset contracts live in `tests/test_vision_preset_contracts.py`. They
pin preset hashes, model input fields, normalization, static shapes, and
deterministic validation postprocessing.

Optional external-reference golden tests should be marked `golden` and placed
under a vision-specific test file when a reference implementation is available,
for example torchvision/timm preprocessing parity for a named ImageNet recipe.

## 9. Mini-C corruption benchmark

`create_minic_datasets` forks a raw preprocessed dataset, applies deterministic
severity 1-5 image corruptions, runs postprocessing, batches, and preserves
`padding_mask`.

```python
import justdata.vision
from justdata.core.registry import get_pipeline
from justdata.vision.minic import create_minic_datasets

pipeline = get_pipeline(dataset="cifar10")

datasets, n = create_minic_datasets(
    corruption_types=["noise", "blur"],
    severity=3,
    dataset_names_arg=["cifar10"],
    splits_arg={"cifar10": ["test"]},
    dataset_type="validation",
    batch_size=128,
    seed=0,
    pipeline=pipeline,
    num_classes=10,
    metadata_mode="numeric_only",
)
```

## 10. Vision/acoustic parity guarantees

The parity harness in `tests/test_cross_modal_parity.py` checks that the final
release surface does not drift between modalities.

| Capability                      |   Vision | Acoustic |
| ------------------------------- | -------: | -------: |
| Hashable presets                | required | required |
| Metadata propagation            | required | required |
| Deterministic eval views        | required | required |
| Stateless stochastic transforms | required | required |
| Corruption datasets             | required | required |
| JAX-friendly numeric metadata   | required | required |
| Golden preprocessing tests      | required | required |

Shared guarantees are implemented in `justdata.core` where possible:
`metadata_mode`, `as_numpy`, padding masks, seeded execution, and preset hashing.
Modality packages own schema-specific stages, registries, corruptions, and
frontend or transform contracts.

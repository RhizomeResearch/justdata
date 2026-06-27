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

`cache_dataset/cache_path` controls the pre-augment cache. For ViT fine-tuning
on large datasets, `cache_model_inputs/model_input_cache_path` can additionally
cache resized and normalized model inputs after postprocess. Training use
requires `allow_train_model_input_cache=True`; use it only for deterministic
training views because stochastic augmentation is cached on first fill.

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
| WILDS image classification | Benchmark defaults: `wilds:camelyon17`, `wilds:fmow`, `wilds:iwildcam`, or `wilds:rxrx1`. Strong opt-ins add `_strong`. |

Use `justdata.vision.presets.get_resolved_preset(name).hash()` in experiment
metadata. A changed hash means the image preprocessing or augmentation contract
changed.

## 5. WILDS source loading

Install `justdata[wilds]` and import `justdata.vision` before resolving WILDS
pipelines. WILDS datasets use the same `load_ds` syntax as TFDS and Hugging
Face sources:

```python
local_ssd = "/local_ssd/justdata"
wilds_ds = "wilds:fmow?split_scheme=time_after_2016"
pipeline = get_pipeline(dataset="wilds:fmow")

ds, n = load_ds(
    dataset_names_arg=[wilds_ds],
    splits_arg={wilds_ds: ["train"]},
    dataset_type="train",
    batch_size=32,
    seed=0,
    pipeline=pipeline,
    num_classes=62,
    data_dir=f"{local_ssd}/sources",
    cache_dataset=True,
    cache_path=f"{local_ssd}/decoded/fmow-train",
    cache_model_inputs=True,
    model_input_cache_path=f"{local_ssd}/model-inputs/fmow-train-224",
    allow_train_model_input_cache=True,
    metadata_mode="numeric_only",
)
```

For remote/downloaded sources, `data_dir` is a cache root. justdata namespaces
source-owned caches under it, for example `hf/vision/`, `wilds/`, and `zenodo/`.
If omitted, the root is `~/.cache/justdata`.

The base `wilds:fmow` preset is deterministic during training, so the explicit
train model-input cache above is suitable for local SSD benchmarking. Use
`cache_dataset/cache_path` alone for stochastic presets such as
`wilds:fmow_strong`, unless freezing the first sampled augmentations is desired.

Supported WILDS datasets are image classification only: Camelyon17, FMoW,
iWildCam, and RxRx1. FMoW temporal drift testing uses WILDS split schemes such
as `?split_scheme=time_after_2016`. Labeled and unlabeled splits must be loaded
in separate `load_ds` calls.

WILDS downloads are disabled by default (`download=false`). To let WILDS
download the dataset under `data_dir/wilds`, add `download=true` to the dataset
string. Use `&` between query options:

```python
wilds_ds = "wilds:fmow?split_scheme=time_after_2016&download=true"
```

For datasets without other options, use:

```python
wilds_ds = "wilds:camelyon17?download=true"
```

## 6. CIFAR, ImageNet, DINOv2, and WILDS preset contracts

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

WILDS benchmark presets keep the reference input policy and avoid broad
ImageNet-style training recipes by default: Camelyon17 uses 96 px ImageNet
normalization, FMoW uses 224 px ImageNet normalization, iWildCam uses 448 px
ImageNet normalization, and RxRx1 uses 256 px inputs with per-image per-channel
standardization plus train-only 90-degree rotations and horizontal flips.

Opt-in WILDS `_strong` presets keep the same input sizes but add stronger
training augmentation. Camelyon17 uses mild color jitter with rotation/flip,
FMoW and iWildCam use resize/flip plus RandAugment, and RxRx1 adds random
erasing while keeping per-image normalization. Mixup and CutMix stay disabled
for WILDS presets unless the caller explicitly overrides them.

## 7. Evaluation split safety

Vision pipelines keep training-only stochastic transforms out of validation and
test views. `load_ds(..., dataset_type="validation")` builds the pipeline with
`is_training=False`, so evaluation uses deterministic resize/crop behavior and
no Mixup, CutMix, random erasing, or training crop randomness.

When computing validation metrics or corruption benchmarks, use
`dataset_type="validation"` and `deterministic=True` if the input order must be
stable.

## 8. Metadata modes for JAX

`load_ds(..., metadata_mode=...)` is shared by vision and acoustic:

| Mode | Behavior |
| :-- | :-- |
| `full` | Keep all metadata, including strings. |
| `numeric_only` | Keep numeric metadata and remove strings from batches. |
| `none` | Drop metadata from output batches. |

Use `metadata_mode="numeric_only"` with `as_numpy=True` for JAX-friendly arrays.
If string metadata is needed for later joins, pass
`sidecar_metadata_path="metadata.jsonl"` and keep batches numeric.

## 9. Golden compatibility tests

Vision preset contracts live in `tests/test_vision_preset_contracts.py`. They
pin preset hashes, model input fields, normalization, static shapes, and
deterministic validation postprocessing.

Optional external-reference golden tests should be marked `golden` and placed
under a vision-specific test file when a reference implementation is available,
for example torchvision/timm preprocessing parity for a named ImageNet recipe.

## 10. Mini-C corruption benchmark

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

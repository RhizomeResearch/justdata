# justdata

A TensorFlow-native data pipeline library providing optimized preprocessing, augmentation, and evaluation recipes for modern supervised and self-supervised computer vision architectures. `justdata` wraps [TensorFlow Datasets](https://www.tensorflow.org/datasets) (TFDS) and [Hugging Face `datasets`](https://huggingface.co/docs/datasets) behind a unified interface, encoding best-practice recipes for architectures such as ConvNeXt, ViT, ResNet, and DINOv2 as first-class, versioned presets.

______________________________________________________________________

## Table of Contents

1. [Installation](#installation)
1. [Architecture](#architecture)
1. [Data Pipeline Stages](#data-pipeline-stages)
1. [Supervised Learning — Training Pipelines](#supervised-learning--training-pipelines)
1. [Supervised Learning — Validation Pipelines](#supervised-learning--validation-pipelines)
1. [The `timm` ImageNet Recipes (ResNet Strikes Back: A1/A2/A3)](#the-timm-imagenet-recipes-resnet-strikes-back-a1a2a3)
1. [Self-Supervised Learning Pipeline (DINOv2)](#self-supervised-learning-pipeline-dinov2)
1. [Registry System](#registry-system)
1. [Presets and Smart Merging](#presets-and-smart-merging)
1. [Dataset Adapters](#dataset-adapters)
1. [Mini-C Corruption Benchmark](#mini-c-corruption-benchmark)
1. [Usage](#usage)
1. [Development](#development)

______________________________________________________________________

## Installation

```bash
pip install justdata
```

**Requirements:** Python ≥ 3.11, TensorFlow ≥ 2.18.1, TensorFlow Datasets ≥ 4.9.9, Hugging Face `datasets[vision]` ≥ 4.5.0.

______________________________________________________________________

## Architecture

`justdata` is structured around a fixed, four-stage pipeline abstraction. Each task (classification, segmentation, depth estimation) exposes exactly four composable functions:

```
(preprocess_fn, augment_fn, late_augment_fn, postprocess_fn)
```

These are assembled by `loader.load_ds` into the following execution graph:

```
fetch_ds -> adapter -> preprocess -> cache -> augment -> shuffle -> postprocess -> batch -> late_augment -> pad -> prefetch
```

The strict ordering reflects the execution domain requirements articulated throughout this document: format normalization occurs before caching; spatial and photometric distortions precede tensor conversion and normalization; and batch-level operations (Mixup, CutMix, random erasing) occur after batching on the GPU.

### Core Public API

| Function                                                             | Description                                                                                                    |
| :------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------- |
| `loader.fetch_ds(dataset_names, splits_info, data_dir)`              | Raw dataset loading from TFDS or Hugging Face. Returns a `tf.data.Dataset` in canonical schema.                |
| `loader.load_ds(...)`                                                | Full pipeline for training or evaluation. Handles caching, augmentation, shuffling, batching, and prefetching. |
| `loader.create_minic_datasets(corruption_types, severity, **kwargs)` | Constructs Mini-C corruption benchmark datasets from a preprocessed, cached base dataset.                      |

Hugging Face datasets are referenced with the `hf:` prefix (e.g., `hf:cifar10`). They are expected to expose `image` and `label` columns and are converted to `tf.data.Dataset` via a generator.

______________________________________________________________________

## Data Pipeline Stages

### Stage 1: Preprocessing (before cache)

Format normalization is performed before the dataset is cached to disk. Operations include rank fixing (2D -> 3D), CHW -> HWC transposition, grayscale -> RGB expansion, and RGBA -> RGB projection. The output shape is forced to `[None, None, 3]`.

### Stage 2: Augmentation (per-sample, training only)

Per-sample spatial and photometric augmentations are applied after the cache, ensuring that each training epoch receives independently sampled augmentations. All augmentation functions use `tf.random.split` for stateless randomization, enabling full reproducibility.

### Stage 3: Postprocessing (every sample)

Resize, channel-wise normalization to zero mean and unit standard deviation, and optional HWC -> CHW transposition. This stage runs unconditionally for both training and evaluation.

### Stage 4: Late Augmentation (per-batch, training only)

Batch-level operations—Mixup, CutMix, and random erasing—are applied after batching. These operations are strictly training-only and require a formed batch to operate across the sample dimension.

______________________________________________________________________

## Supervised Learning — Training Pipelines

To ensure mathematical correctness, the training pipeline is strictly segregated by execution domain. Spatial and photometric distortions are applied to `[0, 255]` image tensors (PIL-equivalent domain) before zero-mean tensor normalization.

### 1. Image Domain Augmentations

#### Geometric Augmentations

- **CIFAR-10/100:** `RandomCrop(32, padding=4, padding_mode='zeros')` -> `RandomHorizontalFlip(p=0.5)`.
- **ImageNet-1K:** `RandomResizedCrop(size=224)` (or **256**). Scale: **(0.08, 1.0)**, Aspect ratio: **(0.75, 1.33)**, Interpolation: Bicubic. -> `RandomHorizontalFlip(p=0.5)`.

#### Photometric Augmentations (Mutually Exclusive Branches)

The photometric augmentation strategy is architecture-dependent and the two branches are mutually exclusive.

- **Modern Branch (ViT / ConvNeXt):** Apply `RandAugment(num_ops=2, magnitude=9)` or `TrivialAugmentWide`. Color Jitter is explicitly disabled to prevent redundant and destructive color space distortion.
- **Legacy Branch (ResNet):** Apply `ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)`. RandAugment is disabled.

### 2. Tensor Domain

- **Conversion & Scaling:** `ToImage()` -> `ToDtype(float32, scale=True)`. Maps **[0, 255]** -> **[0.0, 1.0]**.
- **Normalization:**
  - **CIFAR-10:** `mean=(0.4914, 0.4822, 0.4465)`, `std=(0.2023, 0.1994, 0.2010)`
  - **CIFAR-100:** `mean=(0.5071, 0.4867, 0.4408)`, `std=(0.2675, 0.2565, 0.2761)`
  - **ImageNet-1K:** `mean=(0.485, 0.456, 0.406)`, `std=(0.229, 0.224, 0.225)`
- **Random Erasing (Cutout):** Applied to the normalized tensor. `p=0.25`, `scale=(0.02, 0.33)`, `ratio=(0.3, 3.3)`.

### 3. Mini-Batch Domain

Applied across the batch dimension during training.

- **Repeated Augmentation (RA):** Enabled for ViTs. Typically **3** repetitions per sample per mini-batch.
- **Mixup & CutMix:** Controlled by `mixup_prob=1.0` (probability of batch mixing) and `switch_prob=0.5` (probability of selecting CutMix over Mixup).
  - **Mixup:** $\tilde{x} = \lambda x_i + (1 - \lambda) x_j$, where $\lambda \sim \mathrm{Beta}(0.8,, 0.8)$.
  - **CutMix:** Replaces a rectangular bounding box region; $\lambda \sim \mathrm{Beta}(1.0,, 1.0)$.
  - *Note: The $\alpha$ values (0.8 and 1.0) reflect the DeiT baseline. Mixup/CutMix parameters are strictly recipe-dependent; see the A1/A2/A3 section for ResNet-specific variations.*
- **Label Smoothing:** Cross-entropy loss modification with $\varepsilon = 0.1$.

______________________________________________________________________

## Supervised Learning — Validation Pipelines

Validation pipelines are strictly deterministic. The objective shifts from regularization to feature preservation and scale alignment.

### CIFAR-10 / CIFAR-100 Standard Validation

Because CIFAR images are inherently **32×32** and contain minimal background, spatial cropping destroys the primary subject.

**Pipeline:** `ToImage()` -> `ToDtype(float32, scale=True)` -> `Normalize` (training-set statistics as above).

### ImageNet-1K Standard Baseline (The 0.875 Rule)

For standard supervised models and baseline linear probing, the canonical **0.875** crop ratio discards peripheral background.

**Pipeline:** `Resize(256, interpolation=Bicubic)` -> `CenterCrop(224)` -> `ToImage()` -> `ToDtype(float32, scale=True)` -> `Normalize`.

### ImageNet-1K Modern Recipes (FixRes Strategy)

Modern recipes correct train-test resolution discrepancies by manipulating the validation crop percentage (`crop_pct`).

- **A3 (Light) Validation:** Train at **160×160**, validate at **224×224**. Resize shorter edge to **≈236**, then `CenterCrop(224)`.
- **A1 / A2 (Heavy/Moderate) Validation:** Default test at **224×224** (`crop_pct=1.0`, so `Resize(224)` -> `CenterCrop(224)`). Accuracy improves further via FixRes evaluation at **288×288** (`Resize(288)` -> `CenterCrop(288)`) as demonstrated in the RSB paper.

______________________________________________________________________

## The `timm` ImageNet Recipes (ResNet Strikes Back: A1/A2/A3)

The "ResNet Strikes Back" (RSB) recipes dynamically scale augmentation intensity and training schedules to match model capacity.

> **Crucial context:** These recipes were designed specifically for **ResNet-family architectures**. ViTs typically use distinct recipes (e.g., DeiT, BEiT) with different Mixup $\alpha$, optimizers (AdamW), and loss functions. While the principle of scaling augmentation with model capacity generalizes, the specific hyperparameters below do not transfer directly to ViTs.

| Parameter               | A1 (Heavy)                                           | A2 (Moderate)           | A3 (Light)                                    |
| :---------------------- | :--------------------------------------------------- | :---------------------- | :-------------------------------------------- |
| **Target Architecture** | Large ResNets (e.g., ResNet-152/200) or high compute | ResNet-50 (standard)    | ResNet-50 (fast) or smaller (e.g., ResNet-18) |
| **Training Resolution** | 224                                                  | 224                     | **160** (FixRes strategy)                     |
| **Test Resolution**     | 224 (scales to 288 via FixRes)                       | 224                     | **224**                                       |
| **Epochs**              | **600**                                              | **300**                 | **100**                                       |
| **Optimizer**           | LAMB                                                 | LAMB                    | LAMB                                          |
| **LR Schedule**         | Cosine with warmup                                   | Cosine with warmup      | Cosine with warmup                            |
| **Loss Function**       | **BCE** (per-class binary)                           | **BCE**                 | **CE** (standard)                             |
| **RandAugment**         | $m \approx 7$, $n = 2$                              | $m \approx 6$, $n = 2$ | $m \approx 6$, $n = 2$                       |
| **Random Erasing**      | `p=0.35`                                             | `p=0.25`                | **Disabled** (`p=0.0`)                        |
| **Repeated Aug (RA)**   | Enabled (3×)                                         | Enabled (3×)            | **Disabled**                                  |
| **Mixup** $\alpha$     | 0.2                                                  | 0.2                     | 0.1                                           |
| **CutMix** $\alpha$    | 1.0                                                  | 1.0                     | 1.0                                           |
| **Stochastic Depth**    | Capacity-dependent (e.g., 0.05+)                     | 0.0                     | 0.0                                           |
| **EMA**                 | Yes                                                  | Yes                     | No (or lighter)                               |
| **Weight Decay**        | 0.02                                                 | 0.02                    | 0.02                                          |

*Note: RandAugment parameter $n$ defaults to 2 per the original specification, though exact magnitude strings fluctuate across `timm` versions.*

______________________________________________________________________

## Self-Supervised Learning Pipeline (DINOv2)

DINOv2 employs a Teacher-Student knowledge distillation framework operating on multi-crop asymmetry to force the learning of semantic invariance over low-level frequency matching.

### Multi-Crop Geometric Strategy

- **Global Crops (Context):** 2 crops at **224×224**. Scale: **(0.32, 1.0)**. Passed to both Teacher and Student.
- **Local Crops (Detail):** 8 crops at **96×96** (default). *Use **98×98** ($14 \times 7$) to avoid positional embedding interpolation when using ViT-14 backbones.* Scale: **(0.05, 0.32)**. Passed to Student only.

### Asymmetric Pipeline Implementation

The per-crop asymmetry is structurally enforced by three distinct `Compose` pipelines rather than conditional branching within a single block. Each source image passes through all three pipelines to produce the full 10-crop suite.

- **`global_transform_1`**: Global Crop 1. Enforces strict blurring (`p=1.0`), disables solarization (`p=0.0`).
- **`global_transform_2`**: Global Crop 2. Minimizes blurring (`p=0.1`), enables solarization (`p=0.2`).
- **`local_transform`**: The 8 Local Crops. Moderate blurring (`p=0.5`), disables solarization (`p=0.0`).

### Execution Order and Domain Separation (Per Crop)

| Step | Operation              | Parameters / Per-Crop Asymmetry                                                                                     | Domain             |
| :--- | :--------------------- | :------------------------------------------------------------------------------------------------------------------ | :----------------- |
| 1    | `RandomResizedCrop`    | Aspect ratio **(0.75, 1.33)**, Bicubic interpolation.                                                               | Image              |
| 2    | `RandomHorizontalFlip` | `p=0.5` (all crops)                                                                                                 | Image              |
| 3    | `ColorJitter`          | `p=0.8` (all crops). `b=0.4, c=0.4, s=0.2, h=0.1`                                                                   | Image              |
| 4    | `RandomGrayscale`      | `p=0.2` (all crops)                                                                                                 | Image              |
| 5    | `GaussianBlur`         | $\sigma \sim \mathrm{Uniform}(0.1, 2.0)$, dynamic kernel. Global 1: `p=1.0` · Global 2: `p=0.1` · Local: `p=0.5` | Image              |
| 6    | `Solarization`         | Invert pixels $> 128$. Global 1: `p=0.0` · Global 2: `p=0.2` · Local: `p=0.0`                                       | Image              |
| 7    | Convert & Scale        | `ToImage()` -> `ToDtype(float32, scale=True)`                                                                       | Image -> Tensor    |
| 8    | `Normalize`            | ImageNet `mean` and `std`.                                                                                          | Tensor (zero-mean) |

### SSL Downstream Evaluation

**Image-Level Classification (Linear Probing / $k$-NN):**
Standard supervised inference transforms: `Resize(256, Bicubic)` -> `CenterCrop(224)` -> `ToImage()` -> `ToDtype(float32)` -> `Normalize`. Features are extracted from the `[CLS]` token or a concatenation of `[CLS]` and average-pooled patch tokens.

**Dense Tasks (Segmentation / Depth Validation — Patch Alignment):**
To avoid dropping boundary pixels or forcing complex interpolation during dense evaluation, images are resized to the target scale and padded (reflection or zero) on the bottom and right edges such that both height and width are exact multiples of the ViT patch size (e.g., 14). This is implemented in `transforms.pad_to_patch_multiple`.

______________________________________________________________________

## Registry System

All extensible components in `justdata` use a decorator-based registry pattern with thread-safe lookups.

| Registry              | Decorator                           | Lookup                        |
| :-------------------- | :---------------------------------- | :---------------------------- |
| Crop strategies       | `@register_crop_strategy(name)`     | `get_crop_strategy(name)`     |
| Augment strategies    | `@register_augment_strategy(name)`  | `get_augment_strategy(name)`  |
| Corruptions           | `@register_corruption(name)`        | `apply_minic_corruption(...)` |
| Dataset adapters      | `@register_adapter(dataset_name)`   | `get_adapter(dataset_name)`   |
| Pipelines             | `@register_pipeline(name)`          | `get_pipeline(name)`          |
| Dataset->task mapping | `register_dataset(name, task_type)` | `get_task_for_dataset(name)`  |

`get_pipeline_for_dataset` is the high-level resolver: it infers the task type from the dataset name, merges preset defaults with user-supplied kwargs (via smart merge; see below), and invokes the appropriate pipeline factory.

**Built-in crop strategies:** `random_resized`, `random_pad`.

**Built-in augment strategies:** `rand_augment`, `trivial_augment`, `color_jitter`.

______________________________________________________________________

## Presets and Smart Merging

`presets.py` stores dataset-specific default kwargs for all four pipeline stages. The `_default` preset (ImageNet statistics, 224px, RandAugment) serves as the fallback.

**Available presets:**

| Preset            | Description                                               |
| :---------------- | :-------------------------------------------------------- |
| `_default`        | ImageNet-1K statistics, 224px, RandAugment, modern branch |
| `cifar`           | 32px, TrivialAugment, CIFAR-10 normalization, no resizing |
| `cifar100`        | 32px, TrivialAugment, CIFAR-100 normalization             |
| `imagenet_resnet` | ImageNet statistics, legacy branch with ColorJitter       |
| `imagenet_a1`     | RSB A1: heavy augmentation, BCE loss, 600 epochs          |
| `imagenet_a2`     | RSB A2: moderate augmentation, BCE loss, 300 epochs       |
| `imagenet_a3`     | RSB A3: light augmentation, CE loss, 160px training       |
| `dinov2`          | Asymmetric multi-crop SSL pipeline                        |

`merge_with_presets(dataset, user_kwargs)` implements a **smart merge**: user-supplied kwargs that are identical to the `_default` preset values are treated as "not explicitly overridden," allowing dataset-specific preset values to take precedence. Only kwargs that genuinely differ from the defaults are considered intentional user overrides.

______________________________________________________________________

## Dataset Adapters

Adapters map raw dataset schemas to the canonical schema expected by all pipeline stages (`image`, `label`, and optionally `mask`, `depth`). The identity adapter is applied by default when no specific adapter is registered for a dataset.

```python
from justdata.adapters import register_adapter

@register_adapter("my_dataset")
def my_adapter(sample):
    return {"image": sample["img"], "label": sample["class_id"]}
```

______________________________________________________________________

## Mini-C Corruption Benchmark

`create_minic_datasets` constructs Mini-C benchmark evaluation datasets by forking from a preprocessed, cached base dataset and applying severity-parameterized corruptions in place of standard augmentation. This avoids re-reading raw data for each corruption type.

**Corruption categories and implementations:**

| Category | Corruptions                   |
| :------- | :---------------------------- |
| Blur     | Defocus blur (disk kernel)    |
| Noise    | Gaussian noise, impulse noise |
| Digital  | JPEG compression, pixelation  |
| Weather  | Fog, frost, rain, snow        |

Each corruption is parameterized by severity levels 1–5 via pre-defined lookup tables.

```python
from justdata.loader import create_minic_datasets

datasets = create_minic_datasets(
    corruption_types=["gaussian_noise", "defocus_blur", "fog"],
    severity=3,
    dataset_names_arg=["imagenet2012"],
    splits_arg={"imagenet2012": ["validation"]},
    batch_size=256,
)
```

______________________________________________________________________

## Usage

### Basic Classification Pipeline

```python
from justdata.loader import load_ds
from justdata.registry import get_pipeline_for_dataset

preprocess_fn, augment_fn, late_augment_fn, postprocess_fn = get_pipeline_for_dataset(
    dataset="cifar10",
    is_training=True,
)

train_ds = load_ds(
    dataset_names_arg=["cifar10"],
    splits_arg={"cifar10": ["train"]},
    dataset_type="classification",
    batch_size=128,
    seed=42,
    preprocess_fn=preprocess_fn,
    augment_fn=augment_fn,
    late_augment_fn=late_augment_fn,
    postprocess_fn=postprocess_fn,
    num_classes=10,
)
```

### Loading a Hugging Face Dataset

```python
train_ds = load_ds(
    dataset_names_arg=["hf:cifar10"],
    splits_arg={"hf:cifar10": ["train"]},
    ...
)
```

### Overriding Preset Parameters

```python
preprocess_fn, augment_fn, late_augment_fn, postprocess_fn = get_pipeline_for_dataset(
    dataset="imagenet2012",
    is_training=True,
    aug_kwargs={"augment_type": "trivial_augment"},  # overrides preset default
)
```

### DINOv2 SSL Pipeline

```python
preprocess_fn, augment_fn, late_augment_fn, postprocess_fn = get_pipeline_for_dataset(
    dataset="imagenet2012",
    pipeline_name="classification",
    is_training=True,
    apply_presets=True,
    aug_kwargs={"mode": "ssl"},
)
```

______________________________________________________________________

## Development

This project uses [devenv](https://devenv.sh/) (Nix-based) with `uv` for Python dependency management.

```bash
devenv shell        # enter the development environment
uv sync             # install/update dependencies
uv run pytest       # run all tests
uv run pytest tests/path/to/test_file.py::test_name  # run a single test
```

The project targets Python 3.12 (see `.python-version`). `LD_LIBRARY_PATH` is configured by devenv for native libraries.

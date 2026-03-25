# justdata

A TensorFlow-native data pipeline library with a modality-neutral core and first-class computer vision and acoustic recipes. `justdata.core` owns loading, adapter, preset, metadata, padding, and seeded execution machinery; `justdata.vision` owns image schemas, transforms, augmentations, corruptions, tasks, and vision presets. `justdata.acoustic` owns audio schemas, decoding, resampling, segmentation, frontends, augmentations, corruptions, DCASE helpers, stats, metadata/JAX helpers, and acoustic presets.

______________________________________________________________________

## Table of Contents

1. [Installation](#installation)
1. [Architecture](#architecture)
1. [Data Pipeline Stages](#data-pipeline-stages)
1. [Supervised Learning — Training Pipelines](#supervised-learning--training-pipelines)
1. [Automatic Augmentation Policies: RandAugment, TrivialAugment, and TrivialAugmentWide](#automatic-augmentation-policies-randaugment-trivialaugment-and-trivialaugmentwide)
1. [Supervised Learning — Validation Pipelines](#supervised-learning--validation-pipelines)
1. [The `timm` ImageNet Recipes (ResNet Strikes Back: A1/A2/A3)](#the-timm-imagenet-recipes-resnet-strikes-back-a1a2a3)
1. [Self-Supervised Learning Pipeline (DINOv2)](#self-supervised-learning-pipeline-dinov2)
1. [Registry System](#registry-system)
1. [Presets and Smart Merging](#presets-and-smart-merging)
1. [Acoustic Pipelines](#acoustic-pipelines)
1. [Dataset Adapters](#dataset-adapters)
1. [Mini-C Corruption Benchmark](#mini-c-corruption-benchmark)
1. [Audio Corruption Benchmark](#audio-corruption-benchmark)
1. [Usage](#usage)
1. [Development](#development)

______________________________________________________________________

## Installation

```bash
pip install justdata
```

**Requirements:** Python ≥ 3.11, TensorFlow ≥ 2.18.1, TensorFlow Datasets ≥ 4.9.9. Install optional modality extras with `justdata[vision]` for Hugging Face vision datasets or `justdata[acoustic]` for audio dataset/source dependencies.

______________________________________________________________________

## Architecture

`justdata` is structured around a fixed, four-stage pipeline abstraction. Each task (classification, segmentation, depth estimation) exposes exactly four composable functions:

```
(preprocess_fn, augment_fn, late_augment_fn, postprocess_fn)
```

These are assembled by `justdata.core.load_ds` into the following execution graph:

```
fetch_ds -> adapter -> preprocess -> cache -> augment -> shuffle -> postprocess -> batch -> late_augment -> pad -> prefetch
```

The strict ordering reflects the execution domain requirements articulated throughout this document: format normalization occurs before caching; spatial and photometric distortions precede tensor conversion and normalization; and batch-level operations (Mixup, CutMix, random erasing) occur after batching on the GPU.

### Core Public API

| Function                                                             | Description                                                                                                    |
| :------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------- |
| `justdata.core.fetch_ds(dataset_names, splits_info, data_dir)`        | Raw dataset loading through registered source loaders. Returns a `tf.data.Dataset` in canonical schema.        |
| `justdata.core.load_ds(...)`                                         | Full pipeline for training or evaluation. Handles caching, augmentation, shuffling, batching, and prefetching. |
| `justdata.vision.minic.create_minic_datasets(...)`                   | Constructs Mini-C corruption benchmark datasets from a preprocessed, cached base dataset.                      |
| `justdata.acoustic.corruptions.create_audio_corruption_datasets(...)` | Constructs acoustic corruption benchmark datasets from a preprocessed, cached base dataset.                    |
| `justdata.acoustic.dcase2025.make_source_dataset(...)`               | Builds DCASE Task 1 source-domain datasets with split-safety checks.                                           |

Import `justdata.vision` before resolving built-in vision datasets or pipelines. Hugging Face vision datasets are referenced with the `hf:` prefix (e.g., `hf:cifar10`) and are registered by the vision package.

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

- **Modern Branch (ViT / ConvNeXt):** Apply `RandAugment(num_ops=2, magnitude=9)`, `TrivialAugment`, or `TrivialAugmentWide` (see the [Automatic Augmentation Policies](#automatic-augmentation-policies-randaugment-trivialaugment-and-trivialaugmentwide) section for full specifications). Color Jitter is explicitly disabled to prevent redundant and destructive color space distortion.
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

## Automatic Augmentation Policies: RandAugment, TrivialAugment, and TrivialAugmentWide

`justdata` provides native TensorFlow implementations of three closely related automatic augmentation policies: **RandAugment** (Cubuk et al., 2020), **TrivialAugment** (Müller & Hutter, 2021), and **TrivialAugmentWide** (Müller & Hutter, 2021). All three are registered under the `augment_strategy` registry and share a common operation pool and magnitude discretization framework.

### 1. Pipeline Position and Domain Constraints

To guarantee deterministic reproduction of published results, all three policies are applied exclusively in the **image domain** (integer `uint8` tensors, pixel values $\in [0, 255]$), before tensor conversion to floating-point and channel-wise normalization. The canonical execution order within the per-sample augmentation stage is:

1. Random Resized Crop (or Random Pad Crop for CIFAR)
2. **RandAugment / TrivialAugment / TrivialAugmentWide**
3. Random Horizontal Flip *(integrated into the crop strategy)*
4. `ToTensor` — scales to float $[0.0, 1.0]$ *(postprocessing stage)*
5. `Normalize` — subtracts dataset mean, divides by standard deviation *(postprocessing stage)*

Applying photometric and geometric distortions prior to floating-point conversion ensures that operations such as `Posterize` and `Solarize`, which are defined on integer pixel arithmetic, remain numerically well-founded, and that fill values for geometric operations are expressed in the same integer domain as the source image.

### 2. The RA Operation Space ($K = 14$)

All three algorithms draw from the **RA augmentation space**, a fixed pool of $K = 14$ operations. The pool is partitioned into two subsets based on magnitude dependency.

#### Magnitude-Independent Operations (3 ops)

These operations ignore the sampled magnitude $m$ entirely.

| Operation | Description |
| :--- | :--- |
| `Identity` | Returns the image unmodified. |
| `AutoContrast` | Linearly scales the pixel intensity histogram so that the darkest pixel maps to 0 and the brightest to 255. |
| `Equalize` | Equalizes the image histogram per channel using a cumulative distribution function. |

#### Magnitude-Dependent Operations (11 ops)

These operations scale their physical intensity as a function of the sampled magnitude index $m$.

**Sign randomization.** For all geometric operations (`Rotate`, `ShearX`, `ShearY`, `TranslateX`, `TranslateY`) and all color enhancement operations (`Brightness`, `Color`, `Contrast`, `Sharpness`), the direction of the applied transformation is randomized bidirectionally. Given a raw physical magnitude $v$ computed from $m$, the applied value is:

$$v' = v \cdot s, \quad s \sim \mathcal{U}\{-1, +1\}$$

This sign randomization is sampled independently per operation per sample, using a stateless seed derived from the layer's random state.

**Geometric fill.** When pixels are shifted outside the image boundary by `Rotate`, `ShearX`, `ShearY`, `TranslateX`, or `TranslateY`, the vacated regions are filled with a constant value (default: 128). The interpolation mode defaults to nearest-neighbor for exact integer arithmetic; this may be overridden to bilinear to match the `torchvision` ImageNet presets.

**Solarize semantics.** `Solarize` inverts all pixel values that are greater than or equal to the computed threshold $\tau$. Formally, for each pixel $p$:

$$p' = \begin{cases} 255 - p & \text{if } p \geq \tau \\ p & \text{otherwise} \end{cases}$$

where $\tau = 255 \cdot (1 - m / B)$. At $m = 0$, the threshold is 255 (no pixels inverted); at $m = B$, the threshold is 0 (all pixels inverted).

### 3. Magnitude Discretization and Physical Mappings

The magnitude scale is a **31-bin discrete framework** with bin indices $m \in \{0, 1, \ldots, 30\}$ and $B = 30$ (the maximum bin index). This parameterization is the native scale of the `torchvision` (v0.13+) implementation of both RandAugment and TrivialAugmentWide.

The table below specifies the physical mapping formula and the operation-specific bound for each of the two spaces. $W$ and $H$ denote the input image width and height at the moment the augmentation is applied.

| Operation | Sign Rand. | Physical Mapping | Standard Bound (RA / TA) | Wide Bound (TA-Wide) |
| :--- | :---: | :--- | :--- | :--- |
| **Rotate** | Yes | $(m / B) \cdot \text{MaxDeg} \cdot s$ | 30.0° | 135.0° |
| **TranslateX** | Yes | $(m / B) \cdot \text{MaxPx}_X \cdot s$ | $(150 / 331) \times W$ | 32.0 px |
| **TranslateY** | Yes | $(m / B) \cdot \text{MaxPx}_Y \cdot s$ | $(150 / 331) \times H$ | 32.0 px |
| **ShearX** | Yes | $(m / B) \cdot \text{MaxShear} \cdot s$ | 0.3 | 0.99 |
| **ShearY** | Yes | $(m / B) \cdot \text{MaxShear} \cdot s$ | 0.3 | 0.99 |
| **Brightness** | Yes | $1.0 + (m / B) \cdot \text{MaxDelta} \cdot s$ | 0.9 | 0.99 |
| **Color** | Yes | $1.0 + (m / B) \cdot \text{MaxDelta} \cdot s$ | 0.9 | 0.99 |
| **Contrast** | Yes | $1.0 + (m / B) \cdot \text{MaxDelta} \cdot s$ | 0.9 | 0.99 |
| **Sharpness** | Yes | $1.0 + (m / B) \cdot \text{MaxDelta} \cdot s$ | 0.9 | 0.99 |
| **Posterize** | No | $8 - \mathrm{round}\!\left((m / B) \cdot \text{MaxBits}\right)$ | MaxBits = 4 (min 4 bits retained) | MaxBits = 6 (min 2 bits retained) |
| **Solarize** | No | $255.0 \cdot (1 - m / B)$ | threshold down to 0 | threshold down to 0 |
| **Identity** | — | — | — | — |
| **AutoContrast** | — | — | — | — |
| **Equalize** | — | — | — | — |

> **Note on TranslateX / TranslateY:** Translation is a deliberate exception to the "wider bounds" pattern. The TA-Wide space specifies a fixed ceiling of 32 px for both axes. For any image dimension exceeding approximately 71 px, the Standard RA space—whose ceiling scales proportionally with the image dimension—admits a **larger** maximum translation than the Wide space.

### 4. Algorithm Specifications

#### 4.1 RandAugment (RA)

RandAugment (Cubuk et al., 2020) reduces the search space of AutoAugment from $\mathcal{O}(K^N)$ policies to two scalar hyperparameters: the number of sequential operations $N$ and the global magnitude $M$.

**Procedure.**  Given a training image $x$:

1. Independently sample $N$ operation indices $k_1, \ldots, k_N$ uniformly at random **with replacement** from the pool of $K$ operations.
2. For each selected operation $T_{k_i}$, map the fixed global magnitude $M$ to a physical parameter via the operation-specific formula in the table above, applying sign randomization where applicable.
3. Apply the operations sequentially: $x \leftarrow T_{k_N}(\cdots T_{k_1}(x))$.

The magnitude $M$ is expressed on the **31-bin scale** ($M \in \{0, \ldots, 30\}$) used throughout `justdata`. An optional magnitude perturbation $\sigma > 0$ samples the effective level per layer from $\mathcal{N}(M, \sigma^2)$, clipped to $[0, B]$.

**API parameters** (`augment_strategy = "rand_augment"` or direct call to `rand_augment`):

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `num_layers` | `int` | 2 | Number of operations $N$ applied per image. |
| `magnitude` | `float` | 9.0 | Global magnitude $M$ on the 31-bin scale. |
| `magnitude_std` | `float` | 0.0 | Per-layer Gaussian magnitude noise $\sigma$. Set to 0.5 to replicate `timm` stochastic magnitude. |
| `prob_to_apply` | `float \| None` | `None` | If set, each layer is skipped with probability $1 - p$. |
| `rotate_max` | `float` | 30.0 | Maximum rotation angle in degrees. |
| `shear_max` | `float` | 0.3 | Maximum shear coefficient (Standard RA space). |
| `enhance_max` | `float` | 0.9 | Maximum delta for brightness, color, contrast, sharpness. |
| `posterize_max_bits` | `int` | 4 | Bits removed at maximum magnitude; min retained = $8 - \text{MaxBits}$. |
| `translate_const` | `float` | 100.0 | Absolute translate ceiling in pixels. For Standard RA, set to $(150/331) \times \text{image\_width}$. |
| `exclude_ops` | `list[str] \| None` | `None` | Operations to exclude from the sampling pool. |

**Published optimal hyperparameters and 31-bin scale conversion.** The original paper (Cubuk et al., 2020) reports hyperparameters on an **11-level scale** ($M_{\text{paper}} \in \{0, \ldots, 10\}$). To replicate those results using the 31-bin scale employed by `justdata`, apply the conversion $m = M_{\text{paper}} \times 3$.

| Dataset | Architecture | $N$ | $M_{\text{paper}}$ | $m$ (31-bin) |
| :--- | :--- | :---: | :---: | :---: |
| CIFAR-10 | WideResNet-28-2 | 3 | 4 | 12 |
| CIFAR-10 | WideResNet-28-10 | 3 | 5 | 15 |
| CIFAR-10 | PyramidNet + ShakeDrop | 3 | 7 | 21 |
| CIFAR-10 | Shake-Shake | 3 | 9 | 27 |
| ImageNet-1K | ResNet-50 | 2 | 9 | 27 |
| ImageNet-1K | EfficientNet-B7 | 2 | — | 28† |

> †The EfficientNet-B7 value was reported on an extended scale beyond 10; cross-verify against the target codebase before use. The `torchvision` default of `magnitude=9` on the 31-bin scale corresponds to moderate augmentation and is **not** equivalent to the paper's $M_{\text{paper}} = 9$.

#### 4.2 TrivialAugment (TA)

TrivialAugment (Müller & Hutter, 2021) eliminates hyperparameter search entirely by selecting a single operation and sampling its magnitude uniformly at random from the full discrete range on each forward pass.

**Procedure.** Given a training image $x$:

1. Sample one operation index $k \sim \mathcal{U}\{1, \ldots, K\}$ uniformly from the 14-op RA pool.
2. Sample a magnitude $m \sim \mathcal{U}\{0, 1, \ldots, 30\}$ uniformly from the full 31-bin range.
3. Map $m$ to a physical parameter via the **Standard** bounds in the table above, applying sign randomization where applicable.
4. Apply the single operation: $x \leftarrow T_k(x)$.

The proportional translate ceiling is computed dynamically as $(150 / 331) \times W$ (image width), matching the Standard RA space definition.

**API:** `augment_strategy = "trivial_augment"`. Accepts `translate_const` (float, optional; computed from image width if not supplied) and `exclude_ops` (list of strings, optional).

#### 4.3 TrivialAugmentWide (TA-Wide)

TrivialAugmentWide (Müller & Hutter, 2021) is the native `torchvision` variant of TrivialAugment. It uses the same zero-hyperparameter, single-operation protocol as baseline TA, but operates over the **Wide** magnitude bounds, substantially expanding the geometric and photometric search range.

**Procedure.** Identical to TrivialAugment, with two differences:

1. The **Wide** physical bounds from the table above are applied in place of the Standard bounds.
2. The translate ceiling is a fixed **32 px** (not image-proportional).

Wide bounds summary:

| Transformation | Standard RA / TA | TA-Wide |
| :--- | :--- | :--- |
| Rotation range | ±30° | **±135°** |
| Shear range (X and Y) | ±0.3 | **±0.99** |
| Enhancement delta (Brightness, Color, Contrast, Sharpness) | ±0.9 | **±0.99** |
| Posterize (minimum bits retained) | 4 | **2** |
| Translation (fixed ceiling) | $(150/331) \times \text{dim}$ | **32 px** |

**API:** `augment_strategy = "trivial_augment_wide"`. Accepts `exclude_ops` (list of strings, optional). The wide bounds are fixed by design and cannot be overridden via kwargs; to use custom bounds, call `rand_augment` directly with `num_layers=1` and the desired parameters.

### 5. Algorithm Comparison

| Property | RandAugment | TrivialAugment | TrivialAugmentWide |
| :--- | :--- | :--- | :--- |
| **Operations applied per image** | $N$ (tunable, with replacement) | 1 (fixed) | 1 (fixed) |
| **Magnitude $m$** | Fixed global $M$, constant across ops | Sampled: $m \sim \mathcal{U}\{0, \ldots, 30\}$ | Sampled: $m \sim \mathcal{U}\{0, \ldots, 30\}$ |
| **Hyperparameter search required** | Grid search over $(N, M)$ | None | None |
| **Transformation bounds** | Standard RA space | Standard RA space | Wide space |
| **Translate ceiling** | Configurable (default 100 px) | $(150/331) \times W$ (image-proportional) | 32 px (fixed) |
| **Reference implementation** | `torchvision` | `automl/trivialaugment` | `torchvision` |
| **`justdata` registry key** | `rand_augment` | `trivial_augment` | `trivial_augment_wide` |

### 6. Non-RA Operations

The following operations exist in `NAME_TO_FUNC` and can be invoked via direct calls to `rand_augment` but are **excluded from the default 14-op RA pool** and therefore not reachable through any of the three automatic augmentation strategies:

| Operation | Notes |
| :--- | :--- |
| `Invert` | Inverts all pixel values: $p' = 255 - p$. |
| `Cutout` | Erases a random square patch (fill with constant). Superseded by the `random_erasing` late-augmentation stage. |
| `SolarizeAdd` | Additive solarization variant. |
| `Grayscale` | Converts to single-channel luminance and broadcasts to RGB. |

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
To avoid dropping boundary pixels or forcing complex interpolation during dense evaluation, images are resized to the target scale and padded (reflection or zero) on the bottom and right edges such that both height and width are exact multiples of the ViT patch size (e.g., 14). This is implemented in `justdata.vision.transforms.pad_to_patch_multiple`.

______________________________________________________________________

## Registry System

All extensible components in `justdata` use a decorator-based registry pattern with thread-safe lookups.

| Registry              | Decorator                           | Lookup                        |
| :-------------------- | :---------------------------------- | :---------------------------- |
| Crop strategies       | `@register_crop_strategy(name)`     | `justdata.vision.augmentations.get_crop_strategy(name)` |
| Augment strategies    | `@register_augment_strategy(name)`  | `justdata.vision.augmentations.get_augment_strategy(name)` |
| Corruptions           | `@register_corruption(name)`        | `justdata.vision.corruptions.apply_minic_corruption(...)` |
| Dataset adapters      | `@register_adapter(dataset_name)`   | `justdata.core.get_adapter(dataset_name)` |
| Source loaders        | `@register_source_loader(prefix)`   | `justdata.core.get_source_loader(dataset_name)` |
| Pipelines             | `@register_pipeline("modality/task")` | `justdata.core.get_pipeline(...)` |
| Dataset metadata      | `register_dataset(name, task_type, modality=...)` | `justdata.core.get_dataset_info(name)` |
| Audio frontends       | `@register_audio_frontend(name)`    | `justdata.acoustic.get_audio_frontend(name)` |
| Audio components      | `@register_audio_*` decorators      | `justdata.acoustic.get_audio_*`, `list_audio_*`, `has_audio_*` |

`get_pipeline` is the high-level resolver: it infers the task type from the dataset name, merges preset defaults with user-supplied kwargs (via smart merge; see below), and invokes the appropriate pipeline factory.

**Built-in crop strategies:** `random_resized`, `random_pad`.

**Built-in augment strategies:** `rand_augment`, `trivial_augment`, `trivial_augment_wide`, `color_jitter`, `none`.

The acoustic package registers audio decoders, resamplers, channel strategies, segment strategies, frontends, augmentations, normalizations, eval views, postprocessors, and corruptions. Use `list_audio_*` helpers to inspect registered acoustic components.

______________________________________________________________________

## Presets and Smart Merging

`justdata.vision.presets` stores dataset-specific default kwargs for all four vision pipeline stages. The vision `_default` preset (ImageNet statistics, 224px, RandAugment) serves as the fallback for vision only.

**Available presets:**

| Preset            | Description                                               |
| :---------------- | :-------------------------------------------------------- |
| `_default`        | ImageNet-1K statistics, 224px, RandAugment, modern branch |
| `cifar`           | 32px, TrivialAugmentWide, CIFAR-10 normalization, no resizing |
| `cifar100`        | 32px, TrivialAugmentWide, CIFAR-100 normalization             |
| `imagenet_resnet` | ImageNet statistics, legacy branch with ColorJitter       |
| `imagenet_a1`     | RSB A1: heavy augmentation, BCE loss, 600 epochs          |
| `imagenet_a2`     | RSB A2: moderate augmentation, BCE loss, 300 epochs       |
| `imagenet_a3`     | RSB A3: light augmentation, CE loss, 160px training       |
| `dinov2`          | Asymmetric multi-crop SSL pipeline                        |

`merge_with_presets(dataset, user_kwargs)` implements a **smart merge**: user-supplied kwargs that are identical to the `_default` preset values are treated as "not explicitly overridden," allowing dataset-specific preset values to take precedence. Only kwargs that genuinely differ from the defaults are considered intentional user overrides.

Resolved presets are serializable and hashable across modalities. Use `get_resolved_preset(name)` from `justdata.vision.presets`, `justdata.acoustic.presets`, or `justdata.core.presets` to obtain an object with `.to_json()` and `.hash()`. Hashes use canonical JSON with sorted keys and a 16-character SHA-256 prefix.

Acoustic presets are typed with `AudioPreset` and nested frozen config dataclasses for preprocessing, segmentation, frontends, labels, normalization, train augment settings, eval views, layout, and metadata policy. The `justdata.audio` namespace is a compatibility alias for `justdata.acoustic`.

See [docs/presets.md](docs/presets.md) for acoustic preset contracts, including EfficientAT/DyMN, PaSST, and CED.

### Automatic preset resolution

When a vision dataset has a registered preset (e.g., `cifar10` -> `cifar`, `cifar100` -> `cifar100`), `get_pipeline` automatically applies it. Vision datasets without a dedicated preset fall back to the vision `_default`.

```python
import justdata.vision
from justdata.core.registry import get_pipeline

# cifar10 automatically gets the 'cifar' preset (32px, TrivialAugmentWide, CIFAR-10 stats)
pipeline = get_pipeline(dataset="cifar10")
```

### Applying a named preset to any dataset

The `dataset` argument in `get_pipeline` drives preset lookup, not just task inference. To apply a specific named preset to a dataset that does not have its own preset (e.g., using `imagenet_a3` for `imagenette`), pass it as `preset`:

```python
import justdata.vision
from justdata.core.loader import load_ds
from justdata.core.registry import get_pipeline

# Resolve the A3 (RSB light) pipeline for imagenette
pipeline = get_pipeline(
    dataset="imagenette",
    preset="imagenet_a3",       # drives preset lookup: 160px train, 224px val, RandAugment m=6
)

train_ds, N = load_ds(
    dataset_names_arg=["imagenette"],  # actual dataset to load
    splits_arg={"imagenette": ["train"]},
    dataset_type="train",
    batch_size=128,
    seed=42,
    pipeline=pipeline,
    num_classes=10,
)
```

The validation pipeline uses the same preset name; the A3 preset's FixRes strategy (`train_image_size=160`, `val_resize_size=236`) is applied automatically:

```python
# The same pipeline can be used for validation; `load_ds` automatically sets `is_training=False` based on `dataset_type`
# and applies the deterministic center-crop resize (FixRes 236 → CenterCrop 224)
val_ds, N = load_ds(
    dataset_names_arg=["imagenette"],
    splits_arg={"imagenette": ["validation"]},
    dataset_type="validation",
    batch_size=256,
    seed=0,
    pipeline=pipeline,
    num_classes=10,
)
```

______________________________________________________________________

## Acoustic Pipelines

Import `justdata.acoustic` before resolving built-in acoustic datasets or pipelines. Acoustic samples use the canonical keys `waveform`, `sample_rate`, optional `label`, `features`, `duration`, and `metadata`.

```python
import justdata.acoustic
from justdata.core.loader import load_ds
from justdata.core.registry import get_pipeline

pipeline = get_pipeline(
    dataset="dcase2025_task1",
    preset="dcase2025_task1_efficientat_32k_1s",
)

ds, n = load_ds(
    dataset_names_arg=["dcase2025:task1"],
    splits_arg={"dcase2025:task1": ["dev_train_25"]},
    dataset_type="train",
    batch_size=64,
    seed=0,
    pipeline=pipeline,
    num_classes=10,
    metadata_mode="numeric_only",
    as_numpy=True,
)
```

Modality documentation:

- [docs/vision.md](docs/vision.md): vision schema, four stages, preset contracts, metadata modes, Mini-C, and parity matrix.
- [docs/acoustic.md](docs/acoustic.md): canonical schema, four stages, metadata modes, golden tests, corruption benchmark, and parity matrix.
- [docs/dcase2025.md](docs/dcase2025.md): DCASE Task 1 source/target helpers and split safety.
- [docs/presets.md](docs/presets.md): vision and acoustic preset selection and hashable contracts.
- [docs/golden_tests.md](docs/golden_tests.md): optional golden compatibility test workflow.

Examples are split by modality under `examples/vision/` and
`examples/acoustic/`. Each directory includes loading, Hugging Face source,
statistics, and corruption benchmark examples.

______________________________________________________________________

## Dataset Adapters

Adapters map raw dataset schemas to the canonical schema expected by all pipeline stages (`image`, `label`, and optionally `mask`, `depth`). The identity adapter is applied by default when no specific adapter is registered for a dataset.

```python
from justdata.core.adapters import register_adapter

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
import justdata.vision
from justdata.vision.minic import create_minic_datasets
from justdata.core.registry import get_pipeline

pipeline = get_pipeline(dataset="imagenet")

datasets = create_minic_datasets(
    corruption_types=["gaussian_noise", "defocus_blur", "fog"],
    severity=3,
    dataset_names_arg=["imagenet2012"],
    splits_arg={"imagenet2012": ["validation"]},
    dataset_type="validation",
    batch_size=256,
    seed=0,
    pipeline=pipeline,
    num_classes=1000,
)
```

______________________________________________________________________

## Audio Corruption Benchmark

`create_audio_corruption_datasets` mirrors Mini-C for acoustic evaluation. It applies deterministic severity 1-5 corruptions in the waveform or spectrogram domain, attaches corruption metadata, runs the normal postprocessing stage, and preserves `padding_mask`.

```python
import justdata.acoustic
from justdata.acoustic.corruptions.datasets import create_audio_corruption_datasets
from justdata.core.registry import get_pipeline

pipeline = get_pipeline(
    dataset="dcase2025_task1",
    preset="dcase2025_task1_efficientat_32k_1s",
)

datasets, n = create_audio_corruption_datasets(
    corruption_types=["additive_white_noise", "clipping"],
    severity=3,
    base_dataset="dcase2025:task1",
    preset="dcase2025_task1_efficientat_32k_1s",
    split="dev_test",
    pipeline=pipeline,
    batch_size=64,
    seed=0,
    num_classes=10,
    metadata_mode="numeric_only",
)
```

______________________________________________________________________

## Usage

### Basic Classification Pipeline

```python
import justdata.vision
from justdata.core.loader import load_ds
from justdata.core.registry import get_pipeline

pipeline = get_pipeline(dataset="cifar10")

train_ds, N = load_ds(
    dataset_names_arg=["cifar10"],
    splits_arg={"cifar10": ["train"]},
    dataset_type="train",
    batch_size=128,
    seed=42,
    pipeline=pipeline,
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

### Applying a Named Preset

To use a specific preset for any dataset, pass the preset name as `preset`:

```python
# Load imagenette with the RSB A3 (light) recipe: 160px training, RandAugment m=6, Mixup α=0.1
pipeline = get_pipeline(
    dataset="imagenette",
    preset="imagenet_a3",  # apply the A3 preset to the imagenette task
)

train_ds, N = load_ds(
    dataset_names_arg=["imagenette"],
    splits_arg={"imagenette": ["train"]},
    dataset_type="train",
    batch_size=128,
    seed=42,
    pipeline=pipeline,
    num_classes=10,
)
```

### Overriding Preset Parameters

```python
pipeline = get_pipeline(
    dataset="imagenet",
    aug_kwargs={"augment_type": "trivial_augment"},  # overrides preset default
)
```

### DINOv2 SSL Pipeline

```python
pipeline = get_pipeline(
    preset="dinov2",
    pipeline_name="vision/classification",
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

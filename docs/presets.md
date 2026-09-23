# Presets

Presets are modality-scoped, serializable contracts. They are not just
convenience defaults: they define model input shape, layout, frontend semantics,
label transform, metadata policy, and augmentation policy.

## How to choose a preset

Start with the checkpoint or model family:

| Model family | Presets |
| :-- | :-- |
| EfficientAT/DyMN | `efficientat_32k_10s_logmel128`, `dymn_32k_10s_logmel128`, `dcase2025_task1_efficientat_32k_1s`, `dcase2025_task1_dymn_32k_1s` |
| PaSST | `passt_32k_10s_logmel128`, `dcase2025_task1_passt_32k_1s` and 10 s DCASE padding variants |
| CED | `ced_tiny_16k_logmel64`, `ced_mini_16k_logmel64`, `ced_small_16k_logmel64`, `ced_base_16k_logmel64`, `dcase2025_task1_ced_16k_1s` |
| Generic acoustic | `audio_default_16k_waveform`, `audio_default_32k_logmel64`, `audio_default_32k_logmel128`, `dcase2025_task1_native_44k_1s` |
| Vision classification | `cifar`, `cifar100`, `imagenet_resnet`, `imagenet_a1`, `imagenet_a2`, `imagenet_a3`, `dinov2`, WILDS `wilds:*` presets |
| Vision semantic or panoptic segmentation | `segmentation_a3` (low), `segmentation_a2` (medium), `segmentation_a1` (high). |

Then choose the dataset duration policy. For DCASE Task 1, 1 s direct-view
presets preserve the benchmark source duration. The 10 s padding variants adapt
1 s clips to checkpoints that require longer input.

## Hashable contracts

Resolved presets expose stable JSON and a short SHA-256 hash:

```python
import justdata.acoustic  # noqa: F401
import justdata.vision  # noqa: F401
from justdata.acoustic.presets import get_resolved_preset as get_audio_preset
from justdata.vision.presets import get_resolved_preset as get_vision_preset

vision_preset = get_vision_preset("cifar10")
audio_preset = get_audio_preset("dcase2025_task1_efficientat_32k_1s")

print(vision_preset.hash())
print(audio_preset.hash())
```

Store the hash in experiment metadata. If the hash changes, rerun compatibility
checks because the preprocessing contract changed.

The preset hash identifies the registered preset only. It does not identify
loader settings or the complete executed pipeline.

## Explicit overrides

`get_pipeline(..., overrides={...})` applies an authoritative recursive overlay
after legacy smart merging. Values remain explicit when they equal the modality
default, are `False`, are zero, or are `None` for a field that permits it.
Mappings preserve siblings; lists, tuples, scalars, and `None` replace the prior
value.

```python
pipeline = get_pipeline(
    dataset="cifar10",
    overrides={
        "aug_kwargs": {"enable": False, "image_size": 224},
        "laug_kwargs": {"enable": False},
        "postproc_kwargs": {
            "image_size": 224,
            "val_resize_size": None,
        },
    },
)
```

Passing `overrides={}` enables strict validation while retaining the selected
preset unchanged. Built-in pipeline resolvers validate top-level and nested
keys, required fields, and incompatible settings before source loading. The
legacy keyword route remains available with its existing smart-merge behavior.

`is_training` is resolved from the loader's `dataset_type`; it cannot be set in
a stage override. The `model_input` contract is derived from resolved stages and
cannot be overridden directly.

## Executed configuration

Use `return_config=True` with `load_ds` or `load_inventory` to append an
immutable `ExecutedConfig` to the return tuple. The snapshot includes all four
pipeline stages, the derived model-input layout, dtype, shape and normalization,
and loader execution settings such as randomness, shuffle, batching, caches,
metadata, concurrency limits, prefetching, and NumPy conversion.

`ExecutedConfig.to_json()` uses schema version 1, sorted keys, compact
separators, preserved Unicode, finite JSON numbers, and JSON arrays for Python
sequences. `to_bytes()` returns the exact UTF-8 encoding. Consumers that require
a full identity should hash and store those bytes:

```python
import hashlib

dataset, n_batches, config = load_ds(..., pipeline=pipeline, return_config=True)
encoded = config.to_bytes()
digest = hashlib.sha256(encoded).hexdigest()
```

`return_raw_ds=True` returns `(prepared, tools, config)` and records that later
stages are pending. `finalize_fn(..., return_config=True)` and
`finalize_epoch(..., return_config=True)` append snapshots with their actual
finalization and epoch settings. Opaque callback replacements cannot be
exported.

The snapshot describes JustData's configured execution. Consumers separately
bind source inventories, transformations outside JustData, exact library
revisions, checkpoints, and other run artifacts.

## Vision contracts

CIFAR presets use 32 px inputs, random padded crop, horizontal flip,
TrivialAugmentWide, CIFAR-specific normalization, `bchw` layout, and
classification labels. `cifar10` resolves through the `cifar` prefix while
`cifar100` has its own normalization contract.

ImageNet presets use ImageNet mean/std normalization, deterministic validation
views, and recipe-specific train augmentation:

| Preset | Contract |
| :-- | :-- |
| `_default` | 224 px ImageNet-style ViT/ConvNeXt recipe with RandAugment. |
| `imagenet_resnet` | 224 px legacy ResNet recipe with ColorJitter. |
| `imagenet_a1` | Heavy RSB recipe for larger ResNets. |
| `imagenet_a2` | Moderate RSB recipe for standard ResNet-50 training. |
| `imagenet_a3` | Light RSB recipe with 160 px train size and FixRes-style eval. |
| `dinov2` | SSL multi-crop contract with global and local crops. |

WILDS image-classification presets are benchmark-faithful by default:

| Preset | Contract |
| :-- | :-- |
| `wilds:camelyon17` | 96 px, ImageNet normalization, no broad training augmentation or batch mixing. |
| `wilds:fmow` | 224 px, ImageNet normalization, no broad training augmentation or batch mixing. |
| `wilds:iwildcam` | 448 px, ImageNet normalization, no broad training augmentation or batch mixing. |
| `wilds:rxrx1` | 256 px, per-image per-channel standardization, train-only 90-degree rotations and horizontal flips. |

The matching `_strong` presets are opt-in recipes for exploratory training:
Camelyon17 adds mild color jitter, FMoW and iWildCam add resize/flip plus
RandAugment, and RxRx1 adds random erasing. Mixup and CutMix stay disabled for
WILDS unless explicitly overridden.

Use `tests/test_vision_preset_contracts.py` as the source of truth for pinned
vision preset hashes, static shapes, normalization, and deterministic eval
behavior.

The segmentation presets share a 512 × 512 training crop and deterministic
rectangular evaluation view. A3 samples a fit-scale factor in `[0.8, 1.25]`
without photometric distortion; A2 uses `[0.5, 2.0]` with distortion; A1 uses
`[0.1, 2.0]` with distortion. A2 is the recommended general starting point.
Callers must supply their own semantic class/ignore IDs or panoptic category,
thing, void and capacity settings. See [`vision.md`](vision.md#15-generic-segmentation-augmentation-presets)
for the exact geometry and transform order.

## EfficientAT/DyMN contract

EfficientAT and DyMN presets use:

- `target_sample_rate=32000`
- mono channel folding with `mono_mean`
- 1024-point STFT, 800-sample window, 320-sample hop
- 128 HTK mel bins up to 15 kHz
- Kaldi-compatible filterbank implementation
- log compression
- `bcft` layout

DCASE variants use 10 scene classes and a `keep_1s`, zero-pad-to-10s, or
repeat-pad-to-10s duration policy.

The generic `dcase2025_task1_native_44k_1s` preset preserves 44.1 kHz audio and
uses its declared native log-mel contract. It is hash- and shape-pinned, but it
does not claim external-reference golden certification.

## PaSST contract

PaSST presets use:

- `target_sample_rate=32000`
- 1024-point STFT, 800-sample window, 320-sample hop
- 128 mel bins with Slaney normalization
- `bcft` layout
- patchout augmentation metadata for training presets

DCASE variants use the same DCASE 10-class scene label contract as EfficientAT.

## CED contract

CED presets use:

- `target_sample_rate=16000`
- Kaldi-style fbank frontend
- 400-point STFT, 400-sample Povey window, 160-sample hop
- 64 HTK mel bins up to 8 kHz
- `btf` layout

Use CED presets when matching CED checkpoints or feature extractors.

## Metadata policy

Acoustic presets default to `metadata_mode="numeric_only"` so batches can be
converted to NumPy/JAX arrays without string leaves. Vision uses the same loader
argument and can opt into the same behavior.

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
| Generic acoustic | `audio_default_16k_waveform`, `audio_default_32k_logmel64`, `audio_default_32k_logmel128` |
| Vision classification | `cifar`, `cifar100`, `imagenet_resnet`, `imagenet_a1`, `imagenet_a2`, `imagenet_a3`, `dinov2` |

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

Use `tests/test_vision_preset_contracts.py` as the source of truth for pinned
vision preset hashes, static shapes, normalization, and deterministic eval
behavior.

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

# Acoustic Pipelines

`justdata.acoustic` is the audio modality package for TensorFlow-native loading,
preprocessing, augmentation, batching, metadata, DCASE helpers, and corruption
benchmarks.

## 1. Why justdata owns audio frontends

Audio checkpoints usually assume exact frontend semantics: resampling, channel
folding, segmentation, STFT windowing, mel scale, log compression, feature
normalization, tensor layout, and static shape. If those steps live outside the
data pipeline, training and evaluation can silently drift from the model contract.

`justdata` owns audio frontends so a preset can describe the full input contract
in one hashable object and so TensorFlow, JAX/NumPy consumers, golden tests, and
corruption benchmarks all use the same preprocessing path.

## 2. Acoustic canonical schema

Raw or adapted acoustic samples use these keys:

| Key | Meaning |
| :-- | :-- |
| `waveform` | Float waveform in time-major layout, normally `[time, channels]`. |
| `sample_rate` | Integer sample rate for `waveform`. |
| `label` | Optional class index, dense vector, string label, or event target. |
| `features` | Optional model frontend output when the frontend is not raw waveform. |
| `duration` | Optional duration in seconds. |
| `metadata` | Optional nested metadata such as dataset, split, clip, device, source, city, and view fields. |

DCASE-compatible adapters also preserve `dataset`, `split`, `example_id`,
`clip_id`, `source_id`, `filename`, `path`, `start_time`, and `end_time` inside
metadata when available.

## 3. Four acoustic pipeline stages

Acoustic pipelines use the same four callables as vision:

```python
(preprocess_fn, augment_fn, late_augment_fn, postprocess_fn)
```

The loader executes them as:

```text
fetch_ds -> adapter -> preprocess -> cache -> augment -> shuffle -> postprocess -> batch -> late_augment -> pad -> prefetch
```

For acoustic data the stages are:

| Stage | Acoustic responsibility |
| :-- | :-- |
| `preprocess` | Decode or accept waveform, cast to float32, standardize layout, resample, fold channels, normalize waveform, and attach base metadata. |
| `augment` | Training-only per-sample segmentation plus waveform or spectrogram augmentations. Evaluation uses deterministic segmentation unless `augment_eval=True`. |
| `postprocess` | Evaluation segmentation, frontend computation, layout conversion, dtype cast, static shape assignment, and label transform. |
| `late_augment` | Training-only batch transforms such as Mixup, CutMixSpec, WavMix, and MixStyle. |

## 4. How to choose a preset

Choose the preset that matches the model frontend contract first, then the
dataset duration policy:

| Use case | Prefer |
| :-- | :-- |
| EfficientAT or DyMN checkpoints trained on 32 kHz log-mel inputs | `efficientat_32k_10s_logmel128`, `dymn_32k_10s_logmel128`, or the DCASE 1 s variants. |
| PaSST checkpoints with 128-bin 32 kHz log-mel input and patchout | `passt_32k_10s_logmel128` or the DCASE 1 s variants. |
| CED checkpoints using 16 kHz Kaldi-style fbank features | `ced_tiny_16k_logmel64`, `ced_mini_16k_logmel64`, `ced_small_16k_logmel64`, `ced_base_16k_logmel64`, or `dcase2025_task1_ced_16k_1s`. |
| Generic waveform experiments | `audio_default_16k_waveform`. |
| Generic log-mel experiments | `audio_default_32k_logmel64` or `audio_default_32k_logmel128`. |

Use `justdata.acoustic.presets.get_resolved_preset(name).hash()` in experiment
metadata. A changed hash means the frontend or pipeline contract changed.

## 5. EfficientAT/DyMN, PaSST, CED preset contracts

EfficientAT and DyMN DCASE presets use 32 kHz mono audio, 128 mel bins, HTK mel
scale, Kaldi-compatible filterbanks, log compression, `bcft` layout, 10 DCASE
Task 1 scene labels, and deterministic center evaluation. DCASE variants either
keep the 1 second source view or pad/repeat to a 10 second model duration.

PaSST presets use 32 kHz mono audio, 128 mel bins, Slaney mel normalization,
`bcft` layout, and patchout train augmentation metadata. The DCASE variants
share the DCASE 10-class label contract.

CED presets use 16 kHz mono audio, Kaldi-style fbank features with 64 mel bins,
Povey windowing, `btf` layout, and DCASE or AudioSet label contracts depending
on the preset.

## 6. DCASE split safety

DCASE 2025 Task 1 helpers distinguish source statistics from target evaluation.
Statistics are allowed on `dev_train_25` and blocked on `dev_test` and `eval`
unless `allow_override=True` is passed explicitly.

```python
from justdata.acoustic.dcase2025 import make_source_dataset, make_target_dataset

source_ds, source_n = make_source_dataset(split="dev_train_25")
target_ds, target_n = make_target_dataset(split="dev_test")
```

Use source-domain filters such as `source_domain={"device": "A"}` only on
source splits. Treat target splits as evaluation-only unless a benchmark protocol
explicitly allows otherwise.

## 7. Metadata modes for JAX

`load_ds(..., metadata_mode=...)` is shared by vision and acoustic:

| Mode | Behavior |
| :-- | :-- |
| `full` | Keep all metadata, including strings. |
| `numeric_only` | Keep numeric metadata and remove strings from batches. This is the default acoustic preset policy for JAX-friendly arrays. |
| `none` | Drop metadata from output batches. |

When string metadata is needed for later joins, pass
`sidecar_metadata_path="metadata.jsonl"` with `metadata_mode="numeric_only"`.
The loader writes string leaves keyed by stable example id while batches remain
NumPy/JAX friendly.

## 8. Golden compatibility tests

Golden tests live under `tests/acoustic/test_golden_*`. They are marked
`golden` and excluded from default CI. Run them only when reference packages,
fixtures, or converted checkpoints are available:

```bash
uv sync --extra golden
uv run pytest -m golden
```

Golden tests should compare exact frontend tensors or checkpoint logits against
external reference implementations. Non-golden tests should keep using local
synthetic data and must not depend on external checkpoints.

## 9. Audio corruption benchmark

`create_audio_corruption_datasets` mirrors the vision Mini-C helper. It forks a
raw preprocessed dataset, applies deterministic severity 1-5 waveform or
spectrogram corruptions, runs postprocessing, batches, and preserves
`padding_mask`.

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
frontend contracts.

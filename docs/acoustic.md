# Acoustic Pipelines

`justdata.acoustic` is the audio modality package for TensorFlow-native loading,
preprocessing, augmentation, batching, metadata, DCASE helpers, and corruption
benchmarks.

The `hf_audio:` source uses Hugging Face Datasets and defaults to its
TorchCodec decoder. The acoustic dependency set pins the compatible Torch 2.10
and TorchCodec 0.10 pair. TorchCodec also requires FFmpeg shared libraries at
runtime; the devenv shell supplies them. Set `decode_mode="justdata"` to ask
Datasets for undecoded path/bytes records and use justdata's TensorFlow WAV
decoder instead.

The vision counterpart is [docs/vision.md](vision.md). Keep the two documents
aligned when changing shared loader behavior, preset contracts, metadata modes,
or corruption dataset APIs.

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

`cache_dataset/cache_path` controls the pre-augment cache. It is disabled by
default: set `cache_dataset=True` with an empty path for an intentional memory
cache, or provide a nonempty filesystem path. Large datasets should use an
explicit disk path or remain uncached. This choice affects performance, not
output values, and is separate from source-owned download caches under
`data_dir`. `cache_model_inputs` can additionally cache deterministic
postprocess outputs before batching; train use requires
`allow_train_model_input_cache=True` because stochastic training views are
materialized on first fill.

For acoustic data the stages are:

| Stage | Acoustic responsibility |
| :-- | :-- |
| `preprocess` | Decode or accept waveform, cast to float32, standardize layout, resample, fold channels, normalize waveform, and attach base metadata. |
| `augment` | Training-only per-sample segmentation plus waveform or spectrogram augmentations. Evaluation uses deterministic segmentation unless `augment_eval=True`. |
| `postprocess` | Evaluation segmentation, frontend computation, layout conversion, dtype cast, static shape assignment, and label transform. |
| `late_augment` | Training-only batch transforms such as Mixup, CutMixSpec, WavMix, and MixStyle. |

### Segmentation contract

`SegmentStrategyConfig.pad_position` controls where short waveforms are padded:
`right` preserves the existing behavior, `center` splits the deficit with an
extra sample on the right, and `random` chooses the split statelessly from the
sample seed. The position applies consistently to zero, repeat, and reflect
padding.

Evaluation `multi_crop` produces a fixed leading view axis `[V, ...]` through
raw-waveform and feature frontends. Batching therefore produces `[B, V, ...]`,
and `duration` describes one emitted view. View start/end times remain vectors
in metadata. `sliding` has a data-dependent view count and is supported only by
the direct segmentation/evaluation-view helpers; model-input static shaping and
batching reject it with a clear error.

`drop_short=True` and non-`None` `min_duration` are rejected during config
validation. They require dataset-level filtering, including an explicit
cardinality and label contract, which is not implemented.

## 4. How to choose a preset

Choose the preset that matches the model frontend contract first, then the
dataset duration policy:

| Use case | Prefer |
| :-- | :-- |
| EfficientAT or DyMN checkpoints trained on 32 kHz log-mel inputs | `efficientat_32k_10s_logmel128`, `dymn_32k_10s_logmel128`, or the DCASE 1 s variants. |
| PaSST checkpoints with 128-bin 32 kHz log-mel input and patchout | `passt_32k_10s_logmel128` or the DCASE 1 s variants. |
| AST checkpoints from YuanGongND/ast | `ast_audioset_16k_10s_fbank128`, `ast_esc50_16k_5s_fbank128`, or `ast_speechcommands_16k_1s_fbank128` with `pipeline_name="acoustic/ast_classification"`. |
| CED checkpoints using 16 kHz Kaldi-style fbank features | `ced_tiny_16k_logmel64`, `ced_mini_16k_logmel64`, `ced_small_16k_logmel64`, `ced_base_16k_logmel64`, or `dcase2025_task1_ced_16k_1s`. |
| Generic waveform experiments | `audio_default_16k_waveform`. |
| Generic log-mel experiments | `audio_default_32k_logmel64` or `audio_default_32k_logmel128`. |

Use `justdata.acoustic.presets.get_resolved_preset(name).hash()` in experiment
metadata. A changed hash means the frontend or pipeline contract changed.

### Usage Examples

Inspect a preset before wiring it into an experiment:

```python
import justdata.acoustic  # registers acoustic presets and pipelines
from justdata.acoustic.presets import get_resolved_preset

preset = get_resolved_preset("efficientat_32k_10s_logmel128")

print(preset.hash())
print(preset["target_sample_rate"])
print(preset["frontend"]["name"])
print(preset["layout"])
```

Use a model preset with `load_ds` by resolving the matching acoustic pipeline
and passing the resulting `DataPipeline` to the loader:

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
    cache_dataset=False,
    metadata_mode="numeric_only",
    as_numpy=True,
)

batch = next(iter(ds))
inputs = batch["features"]
labels = batch["label"]
padding_mask = batch["padding_mask"]
```

AST presets use a dedicated pipeline because the original recipe applies fbank
padding/cropping, SpecAugment, normalization, and optional waveform mixup in a
specific order:

```python
import tensorflow as tf

import justdata.acoustic
from justdata.core.registry import get_pipeline

pipeline = get_pipeline(
    dataset="audioset",
    preset="ast_audioset_16k_10s_fbank128",
    pipeline_name="acoustic/ast_classification",
)

preprocess, augment, late_augment, postprocess = pipeline.build(is_training=False)
sample = {
    "waveform": tf.zeros([160000, 1], dtype=tf.float32),
    "sample_rate": tf.constant(16000, dtype=tf.int32),
    "label": tf.constant([0, 137], dtype=tf.int64),
}

model_sample = postprocess(preprocess(sample), num_classes=527)

assert model_sample["features"].shape == (1024, 128)
assert model_sample["label"].shape == (527,)
```

## 5. EfficientAT/DyMN, PaSST, CED, AST preset contracts

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

AST presets use 16 kHz mono audio, the official AST Torchaudio/Kaldi fbank
recipe with 128 mel bins, `btf` layout, target-frame right padding/front
cropping, and AST normalization `(x - mean) / (std * 2)`. Use the dedicated
`acoustic/ast_classification` pipeline so training SpecAugment runs after fbank
padding/cropping and before normalization. Operation-exact waveform mixup is
available when samples include explicit `ast_mix_waveform`, `ast_mix_label`,
and `ast_mix_lambda` fields; the pipeline does not reproduce AST's Python,
NumPy, and Torch RNG stream.

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
Sidecar identity must come from an integer/string `example_id` or the composite
`dataset`, `split`, and `clip_id` fields. Repeated dataset iterations do not add
duplicate records; a reused ID with different string metadata raises an error.
The file is populated as samples are consumed, so partial dataset consumption
can produce a partial sidecar.

## 8. Golden compatibility tests

Golden tests live under `tests/acoustic/test_golden_*`. They are marked
`golden` and excluded from default CI. AST golden tests run when
`tests/acoustic/golden/ast/*.npz` fixtures have been generated with the
Python 3.12-compatible Torch/Torchaudio stack in the `golden` extra. Run them
only when reference packages, fixtures, or converted checkpoints are available:

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

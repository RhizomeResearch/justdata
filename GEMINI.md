# GEMINI.md

Project-specific guidance for Gemini coding agents when working in this repository.

## Development Environment

This project uses [devenv](https://devenv.sh/) with `uv` for Python dependency management. A `.envrc` file enables automatic environment activation via `direnv`.

```bash
devenv shell        # enter the dev environment
uv sync             # install/update dependencies
uv lock             # update uv.lock after dependency changes
uv run pytest       # run all tests
uv run pytest tests/path/to/test_file.py::test_name  # run a single test
devenv test         # run tests as CI does (same as uv run pytest)
```

The project targets Python 3.12 (see `.python-version`). `LD_LIBRARY_PATH` is configured by devenv for native libraries. No linter or formatter is configured.

## Architecture

`justdata` is a TensorFlow-native data pipeline library split by modality:

- `justdata.core`: modality-neutral loading, adapters, source loaders, presets, dataset metadata, pipeline resolution, batching, padding, caching, and seeded execution.
- `justdata.vision`: computer vision schemas, transforms, stages, augmentations, corruptions, encodings, task pipelines, presets, Mini-C helpers, and Hugging Face vision source loading.
- `justdata.acoustic`: audio schemas, decoding, resampling, channel handling, segmentation, frontends, augmentations, corruptions, DCASE helpers, stats, presets, metadata/JAX helpers, and task pipelines.

Top-level `justdata` intentionally does not re-export old flat-module APIs. Use explicit namespace imports.

## Import Rules

Import `justdata.vision` before resolving built-in vision datasets or pipelines. The import registers vision datasets, presets, source loaders, augmentation strategies, corruptions, and pipelines.

```python
import justdata.vision
from justdata.core import get_pipeline, load_ds

pipeline = get_pipeline(dataset="cifar10")
ds, n = load_ds(..., pipeline=pipeline)
```

Do not reintroduce compatibility wrappers for removed paths such as:

- `justdata.loader`
- `justdata.registry`
- `justdata.presets`
- `justdata.tasks`
- `justdata.augmentations`
- `justdata.transforms`

Vision-specific imports belong under `justdata.vision.*`; generic imports belong under `justdata.core.*`.

## Data Flow

Core execution order:

```text
fetch_ds -> adapter -> preprocess -> cache -> augment -> shuffle -> postprocess -> batch -> late_augment -> pad -> prefetch
```

The four pipeline callables are:

```python
(preprocess_fn, augment_fn, late_augment_fn, postprocess_fn)
```

- `preprocess`: format/schema normalization before caching.
- `augment`: per-sample augmentation after cache, training only.
- `postprocess`: deterministic sample processing before batching.
- `late_augment`: batch-level augmentation after batching, training only.

Keep core code schema-neutral. `justdata.core` must not assume keys such as `image`, `mask`, `label`, `bboxes`, `global_crops`, or `local_crops`, and must not import `justdata.vision`.

## Registries

Core registries:

| Registry | Decorator/API | Lookup |
|---|---|---|
| Dataset metadata | `register_dataset(name, task_type, modality=...)` | `get_dataset_info(name)`, `get_task_for_dataset(name)` |
| Pipelines | `@register_pipeline("modality/task")` | `get_pipeline(...)` |
| Presets | `register_preset(dataset, config, modality=...)` | `get_dataset_presets(dataset, modality=...)` |
| Adapters | `@register_adapter(dataset_name)` | `get_adapter(dataset_name)` |
| Source loaders | `@register_source_loader(prefix)` | `get_source_loader(dataset_name)` |

Vision registries:

| Registry | Decorator/API | Lookup |
|---|---|---|
| Crop strategies | `@register_crop_strategy(name)` | `get_crop_strategy(name)` |
| Augment strategies | `@register_augment_strategy(name)` | `get_augment_strategy(name)` |
| Corruptions | `@register_corruption(name)` | `apply_minic_corruption(...)` |

Acoustic registries:

| Registry | Decorator/API | Lookup |
|---|---|---|
| Decoders/resamplers/channels/segments/frontends | `@register_audio_*(name)` | `get_audio_*`, `list_audio_*`, `has_audio_*` |
| Waveform/spectrogram/batch augmentations | `@register_audio_*_augment(name)` | `get_audio_*_augment(...)` |
| Corruptions | `@register_audio_corruption(name)` | `apply_audio_corruption(...)` |

Registries use locks and should raise descriptive errors on duplicate registration.

## Presets And Dependencies

Presets are modality-scoped in `justdata.core.presets`. Vision convenience wrappers live in `justdata.vision.presets` and register under `modality="vision"`. Acoustic wrappers live in `justdata.acoustic.presets` and register typed `AudioPreset` contracts under `modality="acoustic"`.

The base dependency set should stay as modality-neutral as practical. Vision-only dependencies, such as `datasets[vision]`, belong in the `vision` optional extra and dev dependencies. Audio source dependencies, such as `datasets[audio]`, `soundfile`, `soxr`, and `librosa`, belong in the `acoustic` optional extra and dev dependencies. Golden compatibility dependencies belong in the `golden` optional extra.

## Vision Package

Current vision behavior should remain functionally equivalent unless a task explicitly asks to change behavior.

Important locations:

- `justdata.vision.pipelines`: registers `vision/classification`, `vision/segmentation`, `vision/object_detection`, and `vision/depth_estimation`.
- `justdata.vision.sources`: registers the `hf:` Hugging Face vision source loader.
- `justdata.vision.minic`: owns `create_minic_datasets`.
- `justdata.vision.tasks`: task-specific pipeline factories.
- `justdata.vision.augmentations`, `justdata.vision.corruptions`, `justdata.vision.transforms`, `justdata.vision.stages`: image-specific implementation details.

Hugging Face vision datasets are referenced with the `hf:` prefix and must expose `image` and `label` columns.

## Acoustic Package

Current acoustic behavior should remain functionally equivalent unless a task explicitly asks to change behavior.

Important locations:

- `justdata.acoustic.pipelines`: registers `acoustic/default`, `acoustic/classification`, and `acoustic/identity`.
- `justdata.acoustic.presets`: owns hashable acoustic preset registration and model-family contracts.
- `justdata.acoustic.frontends`: owns waveform, STFT, mel, log-mel, Kaldi fbank, MFCC, and PCEN frontend implementations.
- `justdata.acoustic.dcase2025`: owns DCASE Task 1 parsing, split safety, source/target builders, and metrics helpers.
- `justdata.acoustic.corruptions`: owns audio corruption registration and corruption dataset helpers.
- `justdata.acoustic.metadata` and `justdata.acoustic.jax`: own numeric metadata and NumPy/JAX-friendly output helpers.

Preserve the canonical acoustic schema (`waveform`, `sample_rate`, optional `label`, `features`, `duration`, and `metadata`). Keep `justdata.core` schema-neutral.

## Cross-Modal Parity

Vision and acoustic must keep parity for hashable presets, metadata propagation, deterministic eval views, stateless stochastic transforms, corruption datasets, numeric metadata for JAX, golden preprocessing tests, `padding_mask`, and `as_numpy`.

When changing shared loader behavior, run `tests/test_cross_modal_parity.py`. When changing a modality-specific implementation, update the corresponding parity docs if the contract changes.

## Tests

Run tests with:

```bash
uv run pytest
```

Targeted tests:

- `tests/test_modalities.py`: core/vision import boundary and acoustic registration.
- `tests/test_cross_modal_parity.py`: final parity checks across vision and acoustic.
- `tests/test_registry.py`: modality-aware dataset and pipeline resolution.
- `tests/test_loader.py`: core loader plus vision Mini-C integration.
- `tests/test_pipelines.py`: vision recipes and augmentation behavior.
- `tests/acoustic`: acoustic tests, automatically marked `acoustic`.
- `tests/acoustic/test_golden_*`: optional golden compatibility tests, marked `golden`.

When changing dependencies, run `uv lock` and then `uv run pytest`.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
- `justdata.acoustic`: registration skeleton for future acoustic support. Do not add real audio decoding/features unless explicitly requested.

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

Registries use locks and should raise descriptive errors on duplicate registration.

## Presets And Dependencies

Presets are modality-scoped in `justdata.core.presets`. Vision convenience wrappers live in `justdata.vision.presets` and register under `modality="vision"`.

The base dependency set should stay as modality-neutral as practical. Vision-only dependencies, such as `datasets[vision]`, belong in the `vision` optional extra and dev dependencies.

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

`justdata.acoustic` is a skeleton only. It currently registers `acoustic/identity` to prove the core can resolve non-vision pipelines independently. Future acoustic work should add dataset adapters, decoding, resampling, channel handling, clip/window selection, waveform/spectrogram augmentations, feature normalization, batching, metadata propagation, deterministic evaluation views, and DCASE-style split/device handling under `justdata.acoustic`.

## Tests

Run tests with:

```bash
uv run pytest
```

Targeted tests:

- `tests/test_modalities.py`: core/vision import boundary and acoustic skeleton.
- `tests/test_registry.py`: modality-aware dataset and pipeline resolution.
- `tests/test_loader.py`: core loader plus vision Mini-C integration.
- `tests/test_pipelines.py`: vision recipes and augmentation behavior.

When changing dependencies, run `uv lock` and then `uv run pytest`.

# AGENTS.md

Repository-specific guidance for agents working on `justdata`. This file applies
to the entire repository. Add a nested `AGENTS.md` only when a subtree needs
rules that do not apply elsewhere.

## Start Here

- Read the relevant implementation and tests before changing behavior.
- Use `README.md` for the public API and the files under `docs/` for modality
  contracts. Treat code and tests as the source of truth when documentation has
  drifted.
- Keep changes within the requested scope. Preserve current behavior unless the
  task explicitly changes a contract.

## Repository Map

`justdata` is a TensorFlow-native data pipeline library using a `src/` layout:

- `src/justdata/core/`: modality-neutral loading, registries, presets, metadata,
  batching, padding, caching, and seeded execution.
- `src/justdata/vision/`: vision schemas, transforms, augmentations,
  corruptions, sources, presets, and task pipelines.
- `src/justdata/acoustic/`: acoustic schemas, decoding, resampling, frontends,
  augmentations, corruptions, DCASE helpers, presets, and task pipelines.
- `src/justdata/audio/`: compatibility alias for `justdata.acoustic`; preserve
  its module identity and public behavior.
- `tests/`: core and vision tests; `tests/acoustic/` contains acoustic and
  optional golden-compatibility tests.
- `examples/vision/` and `examples/acoustic/`: runnable, modality-specific
  examples.

Documentation routes:

- `docs/vision.md`: vision stages, schemas, presets, metadata, and parity.
- `docs/acoustic.md`: acoustic schema, stages, frontends, metadata, and parity.
- `docs/dcase2025.md`: DCASE Task 1 sources, targets, and split safety.
- `docs/presets.md`: preset selection, serialization, and hash contracts.
- `docs/golden_tests.md`: optional external-reference compatibility workflow.

## Environment And Commands

The development environment pins Python 3.12 through devenv; the published
package supports Python 3.11-3.13. The tracked `.envrc` activates devenv after a
one-time approval.

```bash
direnv allow                         # approve automatic activation once
devenv shell                         # enter the environment manually
uv sync                              # install/update development dependencies
uv run pytest                        # default suite (excludes golden tests)
uv run pytest tests/path.py::test    # focused test
uv run ruff check .                  # lint
uv run ruff format --check .         # formatting check
devenv test                          # CI-equivalent default pytest entry point
```

Run commands from the repository root. If the shell is not already activated,
use `devenv shell -- <command>`.

## Architectural Invariants

### Imports And Registration

- Import `justdata.vision` before resolving built-in vision datasets or
  pipelines; the import performs required registrations.
- Use explicit namespaces. Do not restore removed flat modules such as
  `justdata.loader`, `justdata.registry`, `justdata.presets`, `justdata.tasks`,
  `justdata.augmentations`, or `justdata.transforms`.
- Generic APIs belong under `justdata.core.*`; modality-specific APIs belong
  under `justdata.vision.*` or `justdata.acoustic.*`.
- Hugging Face vision sources use the `hf:` prefix and must expose `image` and
  `label` columns.
- Registry mutation must remain thread-safe and duplicate registration must
  fail with a descriptive error.

### Pipeline Contract

Core execution order is:

```text
fetch_ds -> adapter -> preprocess -> cache -> augment -> shuffle -> postprocess -> batch -> late_augment -> pad -> prefetch
```

The pipeline tuple is `(preprocess_fn, augment_fn, late_augment_fn,
postprocess_fn)`: preprocess normalizes before caching, augment is per-sample and
training-only, postprocess is deterministic before batching, and late augment is
batch-level and training-only.

- Keep `justdata.core` schema-neutral. It must not import `justdata.vision` or
  assume modality keys such as `image`, `mask`, `label`, `bboxes`, `waveform`,
  or `features`.
- Preserve the canonical acoustic schema: `waveform`, `sample_rate`, and
  optional `label`, `features`, `duration`, and `metadata`.
- Preserve deterministic evaluation views and stateless seeded stochastic
  transforms.

### Presets, Dependencies, And Compatibility

- Presets are modality-scoped, serializable, and hashable. Contract changes
  require corresponding preset-hash and parity test updates.
- Keep the base dependency set modality-neutral. Put vision-only dependencies
  in the `vision` extra, audio source dependencies in `acoustic`, WILDS support
  in `wilds`, and external-reference dependencies in `golden`.
- Change dependencies in `pyproject.toml`, then regenerate `uv.lock` with
  `uv lock`; do not hand-edit the lockfile.
- Maintain vision/acoustic parity for metadata propagation, evaluation views,
  seeded transforms, corruption datasets, numeric JAX metadata, `padding_mask`,
  and `as_numpy`.
- Do not claim checkpoint or logits compatibility from frontend golden tests;
  follow the certification levels in `docs/golden_tests.md`.
- Treat remote dataset metadata and archives as untrusted. Do not weaken source
  URL validation, checksums, download limits, extraction budgets, path/link
  checks, or atomic cache updates.

## Change And Verification Rules

- Bug fixes must include a regression test that fails without the fix.
- New or changed public behavior requires tests and updates to the relevant
  README, modality document, or runnable example.
- Prefer focused tests while iterating. Before handoff, run the smallest set
  that fully covers the changed contract and report any checks not run.
- For Python changes, run `uv run ruff check .` and
  `uv run ruff format --check .`.
- For shared loader, registry, preset, metadata, batching, or padding changes,
  run `uv run pytest tests/test_cross_modal_parity.py` in addition to focused
  tests.
- For broad or cross-cutting changes, run the full `uv run pytest` suite.
- For dependency, environment, or CI changes, run `uv lock`, the relevant
  `tests/doctor/` checks, and the full default suite.
- Golden tests are opt-in. Run the workflow in `docs/golden_tests.md` only when
  changing certified golden behavior, fixtures, or golden dependencies.
- Documentation-only changes do not require the Python suite unless they alter
  documented commands or executable contracts; verify links, paths, and
  examples against the repository instead.

Useful focused suites:

- `tests/test_modalities.py`: core/vision boundary and acoustic registration.
- `tests/test_registry.py`: modality-aware dataset and pipeline resolution.
- `tests/test_loader.py`: shared loader and vision Mini-C integration.
- `tests/test_pipelines.py`: vision recipes and augmentation behavior.
- `tests/test_cross_modal_parity.py`: shared vision/acoustic contracts.
- `tests/acoustic/`: acoustic implementation and contract tests.

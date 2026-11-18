# `justdata.audio`

## Overview

Compatibility alias package for `justdata.acoustic`. Re-exports the public `justdata.acoustic` namespace and registers
`justdata.audio.*` module aliases in `sys.modules` so imports under `justdata.audio` resolve to the canonical
`justdata.acoustic` symbols and `ModuleType` instances.

## Interface

- `[__init__.py:__all__]`: Re-exports `_acoustic.__all__` (`justdata.acoustic.__all__`) via
  `from justdata.acoustic import *`, exposing all public `justdata.acoustic` symbols and submodules (`adapters`,
  `augment`, `batching`, `channel`, `compat`, `configs`, `corruptions`, `datasets`, `decoding`, `dcase2025`, `evals`,
  `eval_views`, `filters`, `frontends`, `jax`, `labels`, `layouts`, `metadata`, `normalization`, `pipelines`,
  `postprocessing`, `preprocessing`, `presets`, `registry`, `resampling`, `schema`, `segment`, `sources`, `stages`,
  `stats`, `tasks`) under `justdata.audio`.
- `[__init__.py:_alias_name]`: Exposes `justdata.audio.<subpath>` module aliases in `sys.modules` for every `ModuleType`
  in `_acoustic.__all__` and every loaded `justdata.acoustic.<subpath>` descendant module.

## Invariants

- `[__init__.py:__all__]`: `__all__` is bound directly to `_acoustic.__all__`, keeping `justdata.audio` public exports
  identical to `justdata.acoustic`.
- `[__init__.py:_value]`: Every `ModuleType` exported in `_acoustic.__all__` is registered at `f"{__name__}.{_name}"`
  via `sys.modules.setdefault`, preserving `ModuleType` identity with `justdata.acoustic` without overwriting existing
  `sys.modules` entries.
- `[__init__.py:_acoustic_prefix]`: Every loaded `sys.modules` entry prefixed with `f"{_acoustic.__name__}."` is aliased
  to `f"{__name__}.{_canonical_name.removeprefix(_acoustic_prefix)}"` via `sys.modules.setdefault`, preserving canonical
  `ModuleType` identity across nested submodules.

## Side Effects

- `[__init__.py:_acoustic]`: Imports `justdata.acoustic` (`from justdata import acoustic as _acoustic` and
  `from justdata.acoustic import *`) at import time, executing `justdata.acoustic` module initialization and
  registrations.
- `[__init__.py:_value]`: Mutates `sys.modules` at import time via
  `sys.modules.setdefault(f"{__name__}.{_name}", _value)` for each `ModuleType` in `_acoustic.__all__`.
- `[__init__.py:_alias_name]`: Snapshots `tuple(sys.modules.items())` and mutates `sys.modules` at import time via
  `sys.modules.setdefault(_alias_name, _module)` for every loaded `justdata.acoustic.*` module.

## Verification

- Run: `uv run pytest tests/test_modalities.py`
- Run: `uv run pytest tests/acoustic/test_audio_alias.py`

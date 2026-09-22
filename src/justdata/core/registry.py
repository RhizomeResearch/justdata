import copy
from collections.abc import Mapping
from dataclasses import dataclass
import threading
from typing import Any, Callable, Dict, Tuple

PipelineFuncs = Tuple[Callable, Callable, Callable, Callable]


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    modality: str
    task: str | None = None
    pipeline_name: str | None = None
    preset: str | None = None


_REGISTRY_LOCK = threading.Lock()
_DATASETS: Dict[str, DatasetInfo] = {}
_PIPELINES: Dict[str, Callable[..., PipelineFuncs]] = {}
_PIPELINE_CONFIG_RESOLVERS: Dict[
    str, Callable[[dict[str, Any], bool], dict[str, Any]]
] = {}


def register_dataset(
    name: str,
    task_type: str | None = None,
    *,
    modality: str = "core",
    pipeline_name: str | None = None,
    preset: str | None = None,
):
    """Register dataset metadata at runtime."""
    with _REGISTRY_LOCK:
        if name in _DATASETS:
            existing = _DATASETS[name]
            raise ValueError(
                f"Dataset '{name}' is already registered with modality "
                f"'{existing.modality}' and task '{existing.task}'."
            )
        _DATASETS[name] = DatasetInfo(
            name=name,
            modality=modality,
            task=task_type,
            pipeline_name=pipeline_name,
            preset=preset,
        )


def get_dataset_info(name: str) -> DatasetInfo | None:
    with _REGISTRY_LOCK:
        exact = _DATASETS.get(name)
        entries = tuple(_DATASETS.items())

    if exact is not None:
        return exact

    for key, val in sorted(entries, key=lambda kv: len(kv[0]), reverse=True):
        if name.startswith(key):
            return val

    return None


def get_task_for_dataset(name: str) -> str | None:
    info = get_dataset_info(name)
    return info.task if info else None


def _pipeline_modality(pipeline_name: str | None) -> str | None:
    if pipeline_name and "/" in pipeline_name:
        return pipeline_name.split("/", 1)[0]
    return None


class DataPipeline:
    """Encapsulates modality-specific pipeline logic and configuration."""

    def __init__(
        self,
        pipeline_name: str,
        *,
        modality: str | None = None,
        _is_training_legacy: bool = False,
        _dataset_name: str | None = None,
        _preset_name: str | None = None,
        _preset_request: str | None = None,
        _apply_presets: bool = True,
        _strict_config: bool = False,
        _explicit_overrides: Mapping[str, Any] | None = None,
        **kwargs,
    ):
        self.pipeline_name = pipeline_name
        self.modality = modality or _pipeline_modality(pipeline_name)
        self.kwargs = copy.deepcopy(kwargs)
        self._is_training_legacy = _is_training_legacy
        self.dataset_name = _dataset_name
        self.preset_name = _preset_name
        self.preset_request = _preset_request
        self.apply_presets = _apply_presets
        self._strict_config = _strict_config
        self._explicit_overrides = copy.deepcopy(_explicit_overrides)

    def resolve_config(self, is_training: bool) -> dict[str, Any]:
        """Resolve a built-in pipeline into its serializable stage contract."""
        with _REGISTRY_LOCK:
            resolver = _PIPELINE_CONFIG_RESOLVERS.get(self.pipeline_name)
        if resolver is None:
            raise ValueError(
                f"Pipeline '{self.pipeline_name}' has no config_resolver; "
                "executed configuration export is unavailable."
            )
        resolved = resolver(copy.deepcopy(self.kwargs), is_training)
        if not isinstance(resolved, Mapping):
            raise TypeError(
                f"Config resolver for '{self.pipeline_name}' must return a mapping"
            )
        if not isinstance(resolved.get("configuration"), Mapping):
            raise ValueError(
                f"Config resolver for '{self.pipeline_name}' must return "
                "a configuration mapping"
            )
        stages = resolved.get("stages")
        if not isinstance(stages, Mapping):
            raise ValueError(
                f"Config resolver for '{self.pipeline_name}' must return "
                "a stages mapping"
            )
        expected_stages = {
            "preprocess",
            "augment",
            "late_augment",
            "postprocess",
        }
        if set(stages) != expected_stages:
            raise ValueError(
                f"Config resolver for '{self.pipeline_name}' must return exactly "
                f"these stages: {sorted(expected_stages)}"
            )
        for stage_name, stage in stages.items():
            if not isinstance(stage, Mapping) or not {"active", "config"}.issubset(
                stage
            ):
                raise ValueError(
                    f"Resolved stage '{stage_name}' must contain active and config"
                )
        return copy.deepcopy(dict(resolved))

    def build(self, is_training: bool) -> PipelineFuncs:
        """Return the 4 callables: preprocess, augment, batch augment, postprocess."""
        config = copy.deepcopy(self.kwargs)
        postproc = config.setdefault("postproc_kwargs", {})
        postproc["is_training"] = is_training

        if self._strict_config:
            self.resolve_config(is_training)

        with _REGISTRY_LOCK:
            factory = _PIPELINES.get(self.pipeline_name)
        if factory is None:
            raise ValueError(f"Pipeline '{self.pipeline_name}' not found.")

        return factory(**config)

    def __iter__(self):
        return iter(self.build(is_training=self._is_training_legacy))

    def __repr__(self):
        return (
            f"DataPipeline(pipeline_name={self.pipeline_name!r}, "
            f"modality={self.modality!r}, config={self.kwargs})"
        )


def register_pipeline(
    name: str,
    *,
    config_resolver: Callable[[dict[str, Any], bool], dict[str, Any]] | None = None,
):
    """
    Decorator to register a pipeline.

    A pipeline returns four callables:
    (preprocess, augment, batch_augment, postprocess).
    """

    def decorator(fn: Callable[..., PipelineFuncs]):
        with _REGISTRY_LOCK:
            if name in _PIPELINES:
                raise ValueError(
                    f"Pipeline '{name}' already registered by "
                    f"{_PIPELINES[name].__module__}.{_PIPELINES[name].__qualname__}"
                )
            _PIPELINES[name] = fn
            if config_resolver is not None:
                _PIPELINE_CONFIG_RESOLVERS[name] = config_resolver
        return fn

    return decorator


def list_pipelines() -> tuple[str, ...]:
    with _REGISTRY_LOCK:
        return tuple(sorted(_PIPELINES))


def has_pipeline(name: str) -> bool:
    with _REGISTRY_LOCK:
        return name in _PIPELINES


def get_pipeline(
    dataset: str | None = None,
    preset: str | None = None,
    task: str | None = None,
    pipeline_name: str | None = None,
    modality: str | None = None,
    apply_presets: bool = True,
    overrides: Mapping[str, Any] | None = None,
    **kwargs,
) -> DataPipeline:
    """
    Resolve and configure a registered pipeline.

    Dataset registration supplies the default modality, task, pipeline, and
    preset. Explicit ``pipeline_name`` wins over explicit ``task``; both win
    over dataset-driven resolution.
    """
    from justdata.core.presets import (
        get_resolved_preset,
        merge_explicit_overrides,
        merge_with_presets,
    )

    if overrides is not None and not isinstance(overrides, Mapping):
        raise TypeError("overrides must be a mapping or None")

    dataset_info = get_dataset_info(dataset) if dataset else None
    if (
        dataset_info is not None
        and modality is not None
        and modality != dataset_info.modality
    ):
        raise ValueError(
            f"Dataset '{dataset}' has modality '{dataset_info.modality}', "
            f"not '{modality}'."
        )
    effective_modality = modality or (dataset_info.modality if dataset_info else None)
    effective_pipeline = pipeline_name

    if effective_pipeline is None and task is not None:
        effective_pipeline = (
            task
            if "/" in task or effective_modality is None
            else f"{effective_modality}/{task}"
        )

    dataset_is_pipeline_name = False
    if effective_pipeline is None and dataset:
        if dataset_info is not None:
            effective_pipeline = dataset_info.pipeline_name
            if effective_pipeline is None and dataset_info.task is not None:
                effective_pipeline = f"{dataset_info.modality}/{dataset_info.task}"
        elif has_pipeline(dataset):
            effective_pipeline = dataset
            effective_modality = effective_modality or _pipeline_modality(dataset)
            dataset_is_pipeline_name = True

    if effective_pipeline is None:
        raise ValueError(
            f"Could not resolve pipeline for dataset='{dataset}' and preset='{preset}'. "
            "Please specify `task` or `pipeline_name`."
        )

    effective_modality = effective_modality or _pipeline_modality(effective_pipeline)
    pipeline_modality = _pipeline_modality(effective_pipeline)
    if (
        effective_modality is not None
        and pipeline_modality is not None
        and effective_modality != pipeline_modality
    ):
        raise ValueError(
            f"Pipeline '{effective_pipeline}' has modality '{pipeline_modality}', "
            f"not '{effective_modality}'."
        )

    preset_name = preset
    if preset_name is None and dataset_info is not None:
        preset_name = dataset_info.preset or dataset
    elif preset_name is None and dataset and not dataset_is_pipeline_name:
        preset_name = dataset

    if apply_presets and preset_name and effective_modality:
        kwargs = merge_with_presets(
            preset_name,
            kwargs,
            modality=effective_modality,
            overrides=overrides,
        )
    elif overrides is not None:
        kwargs = merge_explicit_overrides(kwargs, overrides)

    if overrides is not None:
        if "model_input" in overrides:
            raise ValueError(
                "overrides.model_input is derived from the executed stages"
            )
        for stage_name in (
            "preproc_kwargs",
            "aug_kwargs",
            "laug_kwargs",
            "postproc_kwargs",
        ):
            stage = kwargs.get(stage_name)
            if isinstance(stage, Mapping):
                runtime_owned = {"is_training"}
                if stage_name == "postproc_kwargs":
                    runtime_owned.add("num_classes")
                conflict = sorted(runtime_owned.intersection(stage))
                if conflict:
                    raise ValueError(f"{stage_name}.{conflict[0]} is runtime-owned")

    resolved_preset_name = None
    if apply_presets and preset_name and effective_modality:
        resolved_preset_name = get_resolved_preset(
            preset_name, modality=effective_modality
        ).name

    return DataPipeline(
        effective_pipeline,
        modality=effective_modality,
        _dataset_name=dataset,
        _preset_name=resolved_preset_name,
        _preset_request=preset_name,
        _apply_presets=apply_presets,
        _strict_config=overrides is not None,
        _explicit_overrides=overrides,
        **kwargs,
    )


def get_pipeline_for_dataset(
    dataset: str,
    task_type: str | None = None,
    pipeline_name: str | None = None,
    apply_presets: bool = True,
    is_training: bool = False,
    overrides: Mapping[str, Any] | None = None,
    **kwargs,
) -> DataPipeline:
    """Compatibility alias for call sites that still resolve by dataset."""
    return get_pipeline(
        dataset=dataset,
        task=task_type,
        pipeline_name=pipeline_name,
        apply_presets=apply_presets,
        overrides=overrides,
        _is_training_legacy=is_training,
        **kwargs,
    )

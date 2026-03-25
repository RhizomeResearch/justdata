from dataclasses import dataclass
import threading
from typing import Callable, Dict, Tuple

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
        _DATASETS[name] = DatasetInfo(
            name=name,
            modality=modality,
            task=task_type,
            pipeline_name=pipeline_name,
            preset=preset,
        )


def get_dataset_info(name: str) -> DatasetInfo | None:
    if name in _DATASETS:
        return _DATASETS[name]

    for key, val in sorted(_DATASETS.items(), key=lambda kv: len(kv[0]), reverse=True):
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
        **kwargs,
    ):
        self.pipeline_name = pipeline_name
        self.modality = modality or _pipeline_modality(pipeline_name)
        self.kwargs = kwargs
        self._is_training_legacy = _is_training_legacy

    def build(self, is_training: bool) -> PipelineFuncs:
        """Return the 4 callables: preprocess, augment, batch augment, postprocess."""
        import copy

        config = copy.deepcopy(self.kwargs)
        postproc = config.setdefault("postproc_kwargs", {})
        postproc["is_training"] = is_training

        if self.pipeline_name not in _PIPELINES:
            raise ValueError(f"Pipeline '{self.pipeline_name}' not found.")

        return _PIPELINES[self.pipeline_name](**config)

    def __iter__(self):
        return iter(self.build(is_training=self._is_training_legacy))

    def __repr__(self):
        return (
            f"DataPipeline(pipeline_name={self.pipeline_name!r}, "
            f"modality={self.modality!r}, config={self.kwargs})"
        )


def register_pipeline(name: str):
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
        return fn

    return decorator


def list_pipelines() -> tuple[str, ...]:
    return tuple(sorted(_PIPELINES))


def has_pipeline(name: str) -> bool:
    return name in _PIPELINES


def get_pipeline(
    dataset: str | None = None,
    preset: str | None = None,
    task: str | None = None,
    pipeline_name: str | None = None,
    modality: str | None = None,
    apply_presets: bool = True,
    **kwargs,
) -> DataPipeline:
    """
    Resolve and configure a registered pipeline.

    Dataset registration supplies the default modality, task, pipeline, and
    preset. Explicit ``pipeline_name`` wins over explicit ``task``; both win
    over dataset-driven resolution.
    """
    from justdata.core.presets import merge_with_presets

    dataset_info = get_dataset_info(dataset) if dataset else None
    effective_modality = modality or (dataset_info.modality if dataset_info else None)
    effective_pipeline = pipeline_name

    if effective_pipeline is None and task is not None:
        effective_pipeline = (
            task if "/" in task or effective_modality is None else f"{effective_modality}/{task}"
        )

    dataset_is_pipeline_name = False
    if effective_pipeline is None and dataset:
        if dataset_info is not None:
            effective_pipeline = dataset_info.pipeline_name
            if effective_pipeline is None and dataset_info.task is not None:
                effective_pipeline = f"{dataset_info.modality}/{dataset_info.task}"
        elif dataset in _PIPELINES:
            effective_pipeline = dataset
            effective_modality = effective_modality or _pipeline_modality(dataset)
            dataset_is_pipeline_name = True

    if effective_pipeline is None:
        raise ValueError(
            f"Could not resolve pipeline for dataset='{dataset}' and preset='{preset}'. "
            "Please specify `task` or `pipeline_name`."
        )

    effective_modality = effective_modality or _pipeline_modality(effective_pipeline)

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
        )

    return DataPipeline(
        effective_pipeline,
        modality=effective_modality,
        **kwargs,
    )


def get_pipeline_for_dataset(
    dataset: str,
    task_type: str | None = None,
    pipeline_name: str | None = None,
    apply_presets: bool = True,
    is_training: bool = False,
    **kwargs,
) -> DataPipeline:
    """Compatibility alias for call sites that still resolve by dataset."""
    return get_pipeline(
        dataset=dataset,
        task=task_type,
        pipeline_name=pipeline_name,
        apply_presets=apply_presets,
        _is_training_legacy=is_training,
        **kwargs,
    )

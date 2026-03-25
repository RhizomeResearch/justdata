from typing import Any, Callable, Dict, Literal, Tuple

KnownDataset = Literal[
    "cifar10",
    "cifar100",
    "imagenet",
    "imagenette",
    "stanford_dogs",
    "voc",
    "voc/2007",
    "voc/2012",
    "nyu_depth_v2_mini",
    "kitti_road",
]

TaskType = Literal[
    "classification", "object_detection", "segmentation", "depth_estimation"
]

_DATASET_TASK_MAP: Dict[str, str] = {
    "cifar10": "classification",
    "cifar100": "classification",
    "imagenet": "classification",
    "imagenette": "classification",
    "stanford_dogs": "classification",
    "voc": "object_detection",
    "voc/2007": "object_detection",
    "voc/2012": "object_detection",
    "nyu_depth_v2_mini": "depth_estimation",
    "kitti_road": "segmentation",
}


def register_dataset(name: str, task_type: str):
    """Register a new dataset -> task mapping at runtime."""
    _DATASET_TASK_MAP[name] = task_type


def get_task_for_dataset(name: str) -> str | None:
    if name in _DATASET_TASK_MAP:
        return _DATASET_TASK_MAP[name]
    # Check prefix match (e.g. "imagenet_a3" -> "imagenet")
    for key, val in _DATASET_TASK_MAP.items():
        if name.startswith(key):
            return val
    return None


PipelineFuncs = Tuple[Callable, Callable, Callable, Callable]


class DataPipeline:
    """Encapsulates task-specific logic and configuration."""

    def __init__(self, pipeline_name: str, _is_training_legacy: bool = False, **kwargs):
        self.pipeline_name = pipeline_name
        self.kwargs = kwargs
        self._is_training_legacy = _is_training_legacy

    def build(self, is_training: bool) -> PipelineFuncs:
        """Returns the 4 callables (preprocess, augment, late_augment, postprocess)."""
        import copy

        # Deep copy to avoid mutating the original configuration
        config = copy.deepcopy(self.kwargs)
        postproc = config.setdefault("postproc_kwargs", {})
        postproc["is_training"] = is_training

        if self.pipeline_name not in _PIPELINES:
            raise ValueError(f"Pipeline '{self.pipeline_name}' not found.")

        return _PIPELINES[self.pipeline_name](**config)

    def __iter__(self):
        """Allows legacy unpacking: a, b, c, d = get_pipeline(...)"""
        return iter(self.build(is_training=self._is_training_legacy))

    def __repr__(self):
        return f"DataPipeline(task={self.pipeline_name}, config={self.kwargs})"


_PIPELINES: Dict[str, Callable[..., PipelineFuncs]] = {}


def register_pipeline(name: str):
    """
    Decorator to register a new pipeline.
    A pipeline is a function that returns a tuple of 4 Callables:
    (preprocess, augment, late_augment, postprocess)
    """

    def decorator(fn: Callable[..., PipelineFuncs]):
        _PIPELINES[name] = fn
        return fn

    return decorator


def get_pipeline(
    dataset: str | None = None,
    preset: str | None = None,
    task: TaskType | None = None,
    pipeline_name: str | None = None,
    apply_presets: bool = True,
    **kwargs,
) -> DataPipeline:
    """
    Resolves and configures a pipeline.

    Args:
        dataset: Actual dataset name (used for task inference).
        preset: Preset name for default hyperparameters (defaults to `dataset`).
        task: Explicit task type (e.g. "classification").
        pipeline_name: Explicit pipeline name (alias for `task`).
        apply_presets: Whether to merge with registered presets.
        **kwargs: Overrides for pipeline/preset parameters.
    """
    from justdata.presets import merge_with_presets

    # 1. Resolve preset name (falls back to dataset name)
    preset_name = preset or dataset

    if apply_presets and preset_name:
        kwargs = merge_with_presets(preset_name, kwargs)

    # 2. Resolve task/pipeline name
    effective_task = pipeline_name or task
    if effective_task is None and dataset:
        effective_task = get_task_for_dataset(dataset)
        if effective_task is None and dataset in _PIPELINES:
            effective_task = dataset

    if effective_task is None:
        raise ValueError(
            f"Could not resolve task for dataset='{dataset}' and preset='{preset}'. "
            "Please specify `task` or `pipeline_name`."
        )

    return DataPipeline(effective_task, **kwargs)


def get_pipeline_for_dataset(
    dataset: str,
    task_type: TaskType | None = None,
    pipeline_name: str | None = None,
    apply_presets: bool = True,
    is_training: bool = False,
    **kwargs,
) -> DataPipeline:
    """Legacy entry point. Use `get_pipeline` instead."""
    return get_pipeline(
        dataset=dataset,
        task=task_type,
        pipeline_name=pipeline_name,
        apply_presets=apply_presets,
        _is_training_legacy=is_training,
        **kwargs,
    )


@register_pipeline("classification")
def default_classification_pipeline(
    preproc_kwargs: dict = None,
    aug_kwargs: dict = None,
    laug_kwargs: dict = None,
    postproc_kwargs: dict = None,
) -> PipelineFuncs:
    preproc_kwargs = preproc_kwargs or {}
    aug_kwargs = aug_kwargs or {}
    laug_kwargs = laug_kwargs or {}
    postproc_kwargs = postproc_kwargs or {}

    # Automatically enable one-hot labels if MixUp/CutMix is used to ensure
    # shape consistency between training and validation labels.
    if laug_kwargs.get("mixup_alpha", 0) > 0 or laug_kwargs.get("cutmix_alpha", 0) > 0:
        postproc_kwargs.setdefault("one_hot_labels", True)

    from justdata.tasks.classification import (
        make_augmentations,
        make_late_augmentations,
        make_postprocessing,
        make_preprocessing,
    )

    return (
        make_preprocessing(**preproc_kwargs),
        make_augmentations(**aug_kwargs),
        make_late_augmentations(**laug_kwargs),
        make_postprocessing(**postproc_kwargs),
    )


@register_pipeline("object_detection")
def default_object_detection_pipeline(**kwargs) -> PipelineFuncs:
    raise NotImplementedError


@register_pipeline("segmentation")
def default_segmentation_pipeline(
    preproc_kwargs: dict = None,
    aug_kwargs: dict = None,
    laug_kwargs: dict = None,
    postproc_kwargs: dict = None,
) -> PipelineFuncs:
    preproc_kwargs = preproc_kwargs or {}
    aug_kwargs = aug_kwargs or {}
    laug_kwargs = laug_kwargs or {}
    postproc_kwargs = postproc_kwargs or {}

    from justdata.tasks.segmentation import (
        make_augmentations,
        make_late_augmentations,
        make_postprocessing,
        make_preprocessing,
    )

    return (
        make_preprocessing(**preproc_kwargs),
        make_augmentations(**aug_kwargs),
        make_late_augmentations(**laug_kwargs),
        make_postprocessing(**postproc_kwargs),
    )


@register_pipeline("depth_estimation")
def default_depth_estimation_pipeline(**kwargs) -> PipelineFuncs:
    raise NotImplementedError

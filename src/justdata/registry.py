from typing import Callable, Dict, Literal, Tuple

KnownDataset = Literal[
    "cifar10",
    "cifar100",
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
    "imagenette": "classification",
    "stanford_dogs": "classification",
    "voc": "object_detection",
    "voc/2007": "object_detection",
    "voc/2012": "object_detection",
    "nyu_depth_v2_mini": "depth_estimation",
    "kitti_road": "segmentation",
}


def register_dataset(name: str, task_type: str):
    """Register a new dataset → task mapping at runtime."""
    _DATASET_TASK_MAP[name] = task_type


def get_task_for_dataset(name: str) -> str | None:
    return _DATASET_TASK_MAP.get(name)


PipelineFuncs = Tuple[Callable, Callable, Callable, Callable]

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


def get_pipeline(name: str, **kwargs) -> PipelineFuncs:
    """
    Get a specific registered pipeline by name.
    """
    if name not in _PIPELINES:
        raise ValueError(
            f"Pipeline '{name}' not found. Available pipelines: {list(_PIPELINES.keys())}"
        )
    return _PIPELINES[name](**kwargs)


def get_pipeline_for_dataset(
    dataset: str,
    task_type: TaskType | None = None,
    pipeline_name: str | None = None,
    apply_presets: bool = True,
    is_training: bool = False,
    **kwargs,
) -> PipelineFuncs:
    """
    Resolves the pipeline dynamically based on user choices, dataset, or task.
    If `pipeline_name` is provided, it has the highest priority (e.g. for quick experimentation).
    Otherwise, it infers the task from the dataset and falls back to the default pipeline for that task.
    If `apply_presets` is True, it merges the dataset's default preset settings with any kwargs.
    """
    from justdata.presets import merge_with_presets

    if apply_presets:
        kwargs = merge_with_presets(dataset, kwargs)

    postproc = kwargs.setdefault("postproc_kwargs", {})
    postproc.setdefault("is_training", is_training)

    if pipeline_name is None:
        # Infer task type if not explicitly provided
        task = task_type or get_task_for_dataset(dataset)
        if task is None:
            raise ValueError(
                f"Cannot infer task/pipeline for unknown dataset '{dataset}'. Please specify `task_type` or `pipeline_name`."
            )
        # By default, use the task type as the pipeline name
        pipeline_name = task

    return get_pipeline(pipeline_name, **kwargs)


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

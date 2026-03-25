from justdata.core.registry import PipelineFuncs, register_pipeline


@register_pipeline("vision/classification")
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

    if laug_kwargs.get("mixup_alpha", 0) > 0 or laug_kwargs.get("cutmix_alpha", 0) > 0:
        postproc_kwargs.setdefault("one_hot_labels", True)

    from justdata.vision.tasks.classification import (
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


@register_pipeline("vision/object_detection")
def default_object_detection_pipeline(**kwargs) -> PipelineFuncs:
    raise NotImplementedError


@register_pipeline("vision/segmentation")
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

    from justdata.vision.tasks.segmentation import (
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


@register_pipeline("vision/depth_estimation")
def default_depth_estimation_pipeline(**kwargs) -> PipelineFuncs:
    raise NotImplementedError

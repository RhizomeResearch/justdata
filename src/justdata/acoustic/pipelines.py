from justdata.core.registry import PipelineFuncs, register_pipeline


def default_pipeline(
    preproc_kwargs: dict = None,
    aug_kwargs: dict = None,
    laug_kwargs: dict = None,
    postproc_kwargs: dict = None,
    preprocess: dict = None,
    segment: dict = None,
    frontend: dict = None,
    layout: str = None,
    dtype: str = "float32",
    output_key: str = None,
    static_shape: tuple[int | None, ...] = None,
    input_duration: float = None,
    target_sample_rate: int = None,
    label_transform: dict = None,
    train_augment: dict = None,
    waveform_augmentations=None,
    spectrogram_augmentations=None,
    batch_augmentations=None,
    spectrogram_layout: str = None,
    augment_eval: bool = False,
    **kwargs,
) -> PipelineFuncs:
    preproc_kwargs = preproc_kwargs or {}
    aug_kwargs = aug_kwargs or {}
    laug_kwargs = laug_kwargs or {}
    postproc_kwargs = postproc_kwargs or {}

    if preprocess is not None:
        preproc_kwargs.setdefault("config", preprocess)

    is_training = postproc_kwargs.get("is_training", False)
    aug_kwargs.setdefault("is_training", is_training)
    aug_kwargs.setdefault("augment_eval", augment_eval)

    if train_augment is not None and is_training:
        aug_kwargs.setdefault("train_augment", train_augment)
        laug_kwargs.setdefault("train_augment", train_augment)
    if waveform_augmentations is not None:
        aug_kwargs.setdefault("waveform_augmentations", waveform_augmentations)
    if spectrogram_augmentations is not None:
        aug_kwargs.setdefault("spectrogram_augmentations", spectrogram_augmentations)
    if batch_augmentations is not None:
        laug_kwargs.setdefault("batch_augmentations", batch_augmentations)
    if train_augment is not None and is_training and "batch" in train_augment:
        laug_kwargs.setdefault("batch_augmentations", train_augment["batch"])
    if spectrogram_layout is not None:
        aug_kwargs.setdefault("spectrogram_layout", spectrogram_layout)
        laug_kwargs.setdefault("spectrogram_layout", spectrogram_layout)

    if segment is not None:
        if is_training:
            aug_kwargs.setdefault("segment_config", segment)
        else:
            postproc_kwargs.setdefault("segment_config", segment)

    if frontend is not None:
        postproc_kwargs.setdefault("frontend", frontend)
        postproc_kwargs.setdefault("layout", layout)
        postproc_kwargs.setdefault("dtype", dtype)
        postproc_kwargs.setdefault("output_key", output_key)
        postproc_kwargs.setdefault("static_shape", static_shape)
        postproc_kwargs.setdefault("input_duration", input_duration)
        postproc_kwargs.setdefault("target_sample_rate", target_sample_rate)
        postproc_kwargs.setdefault("preprocess", preprocess)

    if label_transform is not None:
        postproc_kwargs.setdefault("label_transform", label_transform)
        laug_kwargs.setdefault("label_transform", label_transform)

    from justdata.acoustic.tasks import (
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


register_pipeline("acoustic/default")(default_pipeline)


@register_pipeline("acoustic/classification")
def classification_pipeline(**kwargs) -> PipelineFuncs:
    return default_pipeline(**kwargs)


@register_pipeline("acoustic/identity")
def identity_pipeline(**kwargs) -> PipelineFuncs:
    return default_pipeline(**kwargs)

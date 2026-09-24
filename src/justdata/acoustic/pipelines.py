import copy
import dataclasses
from collections.abc import Mapping

from justdata.core.config_resolution import reject_unknown_keys
from justdata.core.registry import PipelineFuncs, register_pipeline


_ACOUSTIC_TOP_LEVEL = {
    "aug_kwargs",
    "augment_eval",
    "batch_augmentations",
    "dtype",
    "eval_views",
    "frontend",
    "input_duration",
    "label_transform",
    "laug_kwargs",
    "layout",
    "metadata",
    "metadata_mode",
    "name",
    "output_key",
    "postproc_kwargs",
    "preprocess",
    "preproc_kwargs",
    "segment",
    "spectrogram_augmentations",
    "spectrogram_layout",
    "static_shape",
    "target_sample_rate",
    "train_augment",
    "waveform_augmentations",
}
_FEATURE_DTYPES = {"float32", "float16", "bfloat16"}
# Position of the time axis in one example for each feature layout.
_TIME_AXIS = {"btf": 0, "bft": 1, "bcft": 2, "btfc": 0}


def _validate_dtype(dtype):
    if dtype not in _FEATURE_DTYPES:
        raise ValueError("pipeline.dtype must be 'float32', 'float16', or 'bfloat16'")


def _validate_dataclass_mapping(cls, value, *, path, nested=None):
    if value is None or dataclasses.is_dataclass(value):
        return
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping")
    field_names = {field.name for field in dataclasses.fields(cls)}
    reject_unknown_keys(value, field_names, path=path)
    for name, child_cls in (nested or {}).items():
        if name in value and value[name] is not None:
            _validate_dataclass_mapping(child_cls, value[name], path=f"{path}.{name}")


def _resolve_acoustic_config(config, is_training, *, implementation="default"):
    from justdata.acoustic.configs import (
        AudioPreprocessConfig,
        FeatureNormConfig,
        FrontendConfig,
        LabelTransformConfig,
        LogCompressionConfig,
        MelConfig,
        STFTConfig,
        SegmentStrategyConfig,
    )
    from justdata.acoustic.postprocessing import (
        default_output_key,
        expected_audio_static_shape,
        frontend_output_kind,
        preset_info,
    )

    reject_unknown_keys(config, _ACOUSTIC_TOP_LEVEL, path="pipeline")
    _validate_dataclass_mapping(
        AudioPreprocessConfig, config.get("preprocess"), path="preprocess"
    )
    _validate_dataclass_mapping(
        SegmentStrategyConfig, config.get("segment"), path="segment"
    )
    _validate_dataclass_mapping(
        FrontendConfig,
        config.get("frontend"),
        path="frontend",
        nested={
            "stft": STFTConfig,
            "mel": MelConfig,
            "log": LogCompressionConfig,
            "norm": FeatureNormConfig,
        },
    )
    _validate_dataclass_mapping(
        LabelTransformConfig,
        config.get("label_transform"),
        path="label_transform",
    )

    preprocess = copy.deepcopy(config.get("preprocess"))
    segment = copy.deepcopy(config.get("segment"))
    frontend = copy.deepcopy(config.get("frontend"))
    label_transform = copy.deepcopy(config.get("label_transform"))
    preproc = copy.deepcopy(config.get("preproc_kwargs") or {})
    aug = copy.deepcopy(config.get("aug_kwargs") or {})
    late = copy.deepcopy(config.get("laug_kwargs") or {})
    post = copy.deepcopy(config.get("postproc_kwargs") or {})
    augment_eval = bool(config.get("augment_eval", False))
    apply_augmentation = is_training or augment_eval

    preprocess_config = None
    if preprocess is not None:
        preproc.setdefault("config", preprocess)
        preprocess_config = AudioPreprocessConfig.from_dict(preprocess)
    aug.setdefault("is_training", is_training)
    aug.setdefault("augment_eval", augment_eval)
    late.setdefault("is_training", is_training)
    late.setdefault("augment_eval", augment_eval)
    late.setdefault("model_layout", config.get("layout"))

    train_augment = config.get("train_augment")
    if train_augment is not None and apply_augmentation:
        aug.setdefault("train_augment", copy.deepcopy(train_augment))
        late.setdefault("train_augment", copy.deepcopy(train_augment))
    for key, stage, stage_key in (
        ("waveform_augmentations", aug, "waveform_augmentations"),
        ("spectrogram_augmentations", late, "spectrogram_augmentations"),
        ("batch_augmentations", late, "batch_augmentations"),
    ):
        if config.get(key) is not None:
            stage.setdefault(stage_key, copy.deepcopy(config[key]))
    if config.get("spectrogram_layout") is not None:
        late.setdefault("spectrogram_layout", config["spectrogram_layout"])
    if train_augment is not None and apply_augmentation and "batch" in train_augment:
        late.setdefault("batch_augmentations", copy.deepcopy(train_augment["batch"]))

    if segment is not None:
        SegmentStrategyConfig.from_dict(segment)
        if apply_augmentation:
            aug.setdefault("segment_config", segment)
        else:
            post.setdefault("segment_config", segment)

    model_input = None
    if frontend is not None:
        frontend_config = FrontendConfig.from_dict(frontend)
        layout = config.get("layout")
        if layout is None:
            raise ValueError("pipeline.layout is required when frontend is configured")
        dtype = config.get("dtype", "float32")
        _validate_dtype(dtype)
        output_kind = frontend_output_kind(frontend_config)
        valid_layouts = (
            {"bt", "btc"}
            if output_kind == "waveform"
            else {"btf", "bft", "bcft", "btfc"}
        )
        if layout not in valid_layouts:
            raise ValueError(
                f"pipeline.layout {layout!r} is invalid for {output_kind} output"
            )
        input_duration = config.get("input_duration")
        if input_duration is not None and (
            isinstance(input_duration, bool)
            or not isinstance(input_duration, (int, float))
            or input_duration <= 0
        ):
            raise ValueError("pipeline.input_duration must be positive when set")
        target_sample_rate = config.get("target_sample_rate")
        if target_sample_rate is not None and (
            isinstance(target_sample_rate, bool)
            or not isinstance(target_sample_rate, int)
            or target_sample_rate <= 0
        ):
            raise ValueError(
                "pipeline.target_sample_rate must be a positive integer when set"
            )
        if (
            preprocess_config is not None
            and target_sample_rate is not None
            and preprocess_config.target_sample_rate != target_sample_rate
        ):
            raise ValueError(
                "pipeline.target_sample_rate must match preprocess.target_sample_rate"
            )
        effective_sample_rate = target_sample_rate
        if effective_sample_rate is None and preprocess_config is not None:
            effective_sample_rate = preprocess_config.target_sample_rate
        if (
            effective_sample_rate is not None
            and frontend_config.stft is not None
            and frontend_config.stft.sample_rate != effective_sample_rate
        ):
            raise ValueError(
                "frontend.stft.sample_rate must match the effective target sample rate"
            )
        configured_output_key = config.get("output_key")
        if configured_output_key is not None and not configured_output_key:
            raise ValueError("pipeline.output_key must be non-empty when set")
        output_key = config.get("output_key") or default_output_key(frontend_config)
        shape_info = preset_info(
            frontend=frontend_config,
            layout=layout,
            dtype=dtype,
            output_key=output_key,
            static_shape=config.get("static_shape"),
            input_duration=input_duration,
            target_sample_rate=target_sample_rate,
            preprocess=preprocess,
            segment=segment,
        )
        static_shape = expected_audio_static_shape(shape_info)
        post.setdefault("frontend", frontend_config.to_dict())
        post.setdefault("layout", layout)
        post.setdefault("dtype", dtype)
        post.setdefault("output_key", config.get("output_key"))
        post.setdefault("static_shape", config.get("static_shape"))
        post.setdefault("input_duration", config.get("input_duration"))
        post.setdefault("target_sample_rate", config.get("target_sample_rate"))
        post.setdefault("preprocess", preprocess)
        model_input = {
            "output_key": output_key,
            "layout": layout,
            "dtype": dtype,
            "static_shape": static_shape,
            "kind": frontend_output_kind(frontend_config),
            "normalization": frontend_config.norm.to_dict(),
        }

    if label_transform is not None:
        LabelTransformConfig.from_dict(label_transform)
        post.setdefault("label_transform", label_transform)
        late.setdefault("label_transform", label_transform)

    post["is_training"] = is_training
    preprocess_fields = {
        field.name for field in dataclasses.fields(AudioPreprocessConfig)
    }
    segment_fields = {field.name for field in dataclasses.fields(SegmentStrategyConfig)}
    reject_unknown_keys(preproc, {"config"} | preprocess_fields, path="preproc_kwargs")
    reject_unknown_keys(
        aug,
        {
            "segment_config",
            "waveform_augmentations",
            "spectrogram_augmentations",
            "train_augment",
            "is_training",
            "augment_eval",
            "spectrogram_key",
            "spectrogram_layout",
        },
        path="aug_kwargs",
    )
    reject_unknown_keys(
        late,
        {
            "batch_augmentations",
            "train_augment",
            "is_training",
            "augment_eval",
            "input_key",
            "input_kind",
            "label_mode",
            "label_transform",
            "spectrogram_augmentations",
            "spectrogram_key",
            "spectrogram_layout",
            "model_layout",
            "batch_mixstyle",
            "cutmix",
            "cutmix_spec",
            "mixstyle",
            "mixup",
            "wavmix",
        },
        path="laug_kwargs",
    )
    reject_unknown_keys(
        post,
        {
            "segment_config",
            "is_training",
            "frontend",
            "layout",
            "dtype",
            "output_key",
            "static_shape",
            "input_duration",
            "target_sample_rate",
            "preprocess",
            "label_transform",
        }
        | segment_fields,
        path="postproc_kwargs",
    )
    waveform_active = bool(
        aug.get("waveform_augmentations")
        or (aug.get("train_augment") and aug["train_augment"].get("waveform"))
    )
    late_active = bool(
        late.get("spectrogram_augmentations")
        or late.get("batch_augmentations")
        or late.get("train_augment")
    )
    return {
        "configuration": copy.deepcopy(config),
        "implementation": implementation,
        "stages": {
            "preprocess": {"active": bool(preproc), "config": preproc},
            "augment": {
                "active": apply_augmentation
                and (segment is not None or waveform_active),
                "config": aug,
            },
            "late_augment": {
                "active": apply_augmentation and late_active,
                "config": late,
            },
            "postprocess": {
                "active": bool(frontend is not None or label_transform is not None)
                or (segment is not None and not apply_augmentation),
                "config": post,
            },
        },
        "model_input": model_input,
        "requirements": {"num_classes": False},
    }


def _resolve_default_config(config, is_training):
    return _resolve_acoustic_config(config, is_training)


def _resolve_ast_config(config, is_training):
    from justdata.acoustic.compat.ast import ast_frontend
    from justdata.acoustic.configs import FrontendConfig
    from justdata.acoustic.schema import FEATURES

    resolved = _resolve_acoustic_config(config, is_training, implementation="ast")
    for stage_name in ("aug_kwargs", "laug_kwargs", "postproc_kwargs"):
        if config.get(stage_name):
            key = sorted(config[stage_name])[0]
            raise ValueError(f"{stage_name}.{key} is not supported by the AST pipeline")

    ignored_settings = (
        "augment_eval",
        "batch_augmentations",
        "eval_views",
        "spectrogram_augmentations",
        "spectrogram_layout",
        "waveform_augmentations",
    )
    for name in ignored_settings:
        if config.get(name):
            raise ValueError(f"pipeline.{name} is not supported by the AST pipeline")

    frontend = FrontendConfig.from_dict(config.get("frontend") or ast_frontend())
    layout = config.get("layout") or "btf"
    dtype = config.get("dtype", "float32")
    _validate_dtype(dtype)
    if layout not in _TIME_AXIS:
        raise ValueError(f"pipeline.layout {layout!r} is invalid for AST output")

    ast = {}
    metadata = config.get("metadata")
    train_augment = config.get("train_augment")
    if isinstance(metadata, Mapping) and isinstance(metadata.get("ast"), Mapping):
        ast.update(copy.deepcopy(metadata["ast"]))
    if isinstance(train_augment, Mapping) and isinstance(
        train_augment.get("ast"), Mapping
    ):
        ast.update(copy.deepcopy(train_augment["ast"]))

    static_shape = config.get("static_shape")
    if "target_length" in ast:
        target_length = int(ast["target_length"])
    elif static_shape is None:
        raise ValueError(
            "pipeline.static_shape or metadata.ast.target_length is required "
            "by the AST pipeline"
        )
    else:
        target_length = int(static_shape[_TIME_AXIS[layout]])

    if static_shape is not None:
        if static_shape[_TIME_AXIS[layout]] not in {None, target_length}:
            raise ValueError(
                "pipeline.static_shape time dimension conflicts with "
                "metadata.ast.target_length"
            )

    mean = float(ast.get("mean", -4.2677393))
    std = float(ast.get("std", 4.5689974))
    if std <= 0:
        raise ValueError("metadata.ast.std must be positive")
    output_key = config.get("output_key") or FEATURES
    ast_stage = {
        "frontend": frontend.to_dict(),
        "layout": layout,
        "dtype": dtype,
        "output_key": output_key,
        "static_shape": copy.deepcopy(static_shape),
        "target_length": target_length,
        "normalization": {
            "kind": "ast_affine",
            "mean": mean,
            "std": std,
            "divisor": 2.0,
        },
        "frequency_mask_max_width": int(ast.get("freqm", 0)),
        "time_mask_max_width": int(ast.get("timem", 0)),
        "waveform_noise_and_roll": bool(ast.get("noise", False)),
        "mixup_probability": float(ast.get("mixup", 0.0)),
        "segment": copy.deepcopy(config.get("segment")),
        "label_transform": copy.deepcopy(config.get("label_transform")),
    }
    frontend_stage = {
        key: copy.deepcopy(config.get(key))
        for key in (
            "frontend",
            "layout",
            "dtype",
            "output_key",
            "static_shape",
            "segment",
            "label_transform",
            "train_augment",
            "metadata",
        )
        if config.get(key) is not None
    }
    frontend_stage["ast"] = ast_stage
    # Training computes AST features during augmentation; evaluation does so
    # during postprocessing.
    resolved["stages"]["late_augment"] = {"active": False, "config": {}}
    if is_training:
        resolved["stages"]["augment"] = {
            "active": True,
            "config": {"ast_feature_stage": frontend_stage},
        }
        resolved["stages"]["postprocess"] = {
            "active": config.get("label_transform") is not None,
            "config": {"label_transform": copy.deepcopy(config.get("label_transform"))},
        }
    else:
        resolved["stages"]["augment"] = {"active": False, "config": {}}
        resolved["stages"]["postprocess"] = {
            "active": True,
            "config": {"ast_feature_stage": frontend_stage},
        }
    resolved["model_input"] = {
        "output_key": output_key,
        "layout": layout,
        "dtype": dtype,
        "static_shape": copy.deepcopy(static_shape),
        "kind": "features",
        "normalization": copy.deepcopy(ast_stage["normalization"]),
    }
    return resolved


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
    laug_kwargs.setdefault("is_training", is_training)
    laug_kwargs.setdefault("augment_eval", augment_eval)
    laug_kwargs.setdefault("model_layout", layout)

    if train_augment is not None and (is_training or augment_eval):
        aug_kwargs.setdefault("train_augment", train_augment)
        laug_kwargs.setdefault("train_augment", train_augment)
    if waveform_augmentations is not None:
        aug_kwargs.setdefault("waveform_augmentations", waveform_augmentations)
    if spectrogram_augmentations is not None:
        laug_kwargs.setdefault("spectrogram_augmentations", spectrogram_augmentations)
    if batch_augmentations is not None:
        laug_kwargs.setdefault("batch_augmentations", batch_augmentations)
    if (
        train_augment is not None
        and (is_training or augment_eval)
        and "batch" in train_augment
    ):
        laug_kwargs.setdefault("batch_augmentations", train_augment["batch"])
    if spectrogram_layout is not None:
        laug_kwargs.setdefault("spectrogram_layout", spectrogram_layout)

    if segment is not None:
        if is_training or augment_eval:
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


register_pipeline("acoustic/default", config_resolver=_resolve_default_config)(
    default_pipeline
)


@register_pipeline("acoustic/classification", config_resolver=_resolve_default_config)
def classification_pipeline(**kwargs) -> PipelineFuncs:
    return default_pipeline(**kwargs)


@register_pipeline("acoustic/ast_classification", config_resolver=_resolve_ast_config)
def ast_classification_pipeline(**kwargs) -> PipelineFuncs:
    from justdata.acoustic.compat.ast import ast_pipeline

    return ast_pipeline(**kwargs)


@register_pipeline("acoustic/identity", config_resolver=_resolve_default_config)
def identity_pipeline(**kwargs) -> PipelineFuncs:
    return default_pipeline(**kwargs)

import copy
import dataclasses
import math
from collections.abc import Mapping

from justdata.core.config_resolution import (
    reject_unknown_keys,
    resolve_callable_config,
)
from justdata.core.registry import PipelineFuncs, register_pipeline


_VISION_TOP_LEVEL = {
    "aug_kwargs",
    "augment_eval",
    "laug_kwargs",
    "metadata_mode",
    "model_input",
    "postproc_kwargs",
    "preproc_kwargs",
}
# Arguments that automatic augmentation policies receive from the pipeline.
_POLICY_OMIT = frozenset(
    {"image", "seed", "bboxes", "segmentation_mask", "segmentation_fill_value"}
)
# Options that only the recorded-geometry segmentation mode supports.
_RECORDED_GEOMETRY_OPTIONS = (
    "color_jitter_kwargs",
    "photometric_kwargs",
    "keep_original_mask",
)


def _configuration_snapshot(config):
    """Copy the supplied configuration; ``model_input`` is always derived."""
    return {
        key: copy.deepcopy(value)
        for key, value in config.items()
        if key != "model_input"
    }


def _image_model_input(size, *, permute_image, normalization):
    """Describe a float32 RGB model input, square when ``size`` is static."""
    static_shape = None
    if size is not None:
        static_shape = [3, size, size] if permute_image else [size, size, 3]
    return {
        "output_key": "image",
        "layout": "bchw" if permute_image else "bhwc",
        "dtype": "float32",
        "static_shape": static_shape,
        "normalization": normalization,
    }


def _resolve_photometric(config):
    if config is None:
        return None
    from justdata.vision.augmentations.color import (
        _validate_photometric_params,
        photometric_distortion,
    )

    resolved = resolve_callable_config(
        photometric_distortion,
        config,
        path="photometric_kwargs",
        omit={"image", "seed"},
    )
    try:
        _validate_photometric_params(**resolved)
    except ValueError as exc:
        raise ValueError(f"photometric_kwargs.{exc}") from exc
    return resolved


def _resolve_color_options(config):
    """Resolve the mutually exclusive post- and pre-geometry RGB augmentations."""
    from justdata.vision.augmentations.color import color_jitter

    jitter = config.get("color_jitter_kwargs")
    photometric = _resolve_photometric(config.get("photometric_kwargs"))
    if jitter is not None and photometric is not None:
        raise ValueError("color_jitter_kwargs and photometric_kwargs are exclusive")
    if jitter is not None:
        jitter = resolve_callable_config(
            color_jitter, jitter, path="color_jitter_kwargs", omit={"image", "seed"}
        )
    return jitter, photometric


def _validate_recorded_postprocess(post, flags):
    """Check recorded-geometry normalization and boolean postprocess flags."""
    normalization = _normalization_contract(post)
    if normalization["kind"] == "mean_std" and (
        any(not math.isfinite(v) for row in post["normalization_params"] for v in row)
        or any(v <= 0 for v in post["normalization_params"][1])
    ):
        raise ValueError("normalization_params must be finite with positive std")
    for key in flags:
        if type(post[key]) is not bool:
            raise ValueError(f"postproc_kwargs.{key} must be boolean")
    return normalization


def _recorded_geometry_contract(geometry, is_training):
    """Describe the version-1 geometry shared by semantic and panoptic views."""
    if not is_training:
        resize_policy = "long_side_cap"
    elif geometry["train_scale_range"] is not None:
        resize_policy = "uniform_fit_scale"
    else:
        resize_policy = "uniform_integer_short_side"
    return geometry | {
        "record_version": 1,
        "resize_policy": resize_policy,
        "image_interpolation": "bilinear",
        "mask_interpolation": "nearest",
        "resize_alignment": "half_pixel",
        "dimension_rounding": "half_up_min_one",
        "padding_placement": "bottom_right",
    }


def _reject_recorded_geometry_options(config):
    """Reject recorded-geometry options in the legacy segmentation mode."""
    if (config.get("postproc_kwargs") or {}).get("emit_semantic_targets", False):
        raise ValueError("emit_semantic_targets requires geometry_kwargs")
    for name in _RECORDED_GEOMETRY_OPTIONS:
        if name in config:
            raise ValueError(f"{name} requires geometry_kwargs")


def _validate_positive(value, *, path):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{path} must be positive; got {value!r}")


def _normalization_contract(postproc_kwargs):
    if not postproc_kwargs["normalize_image"]:
        return {"kind": "none"}
    mode = postproc_kwargs.get("normalization_mode", "mean_std")
    if mode == "per_image":
        return {"kind": "per_image", "per_channel": True}
    params = postproc_kwargs["normalization_params"]
    if params is None:
        raise ValueError(
            "postproc_kwargs.normalization_params cannot be None when "
            "normalization is enabled"
        )
    if len(params) != 2 or any(len(values) != 3 for values in params):
        raise ValueError(
            "postproc_kwargs.normalization_params must contain RGB mean and std"
        )
    return {"kind": "mean_std", "mean": params[0], "std": params[1]}


def _validate_classification_nested(aug):
    from justdata.vision.augmentations.auto import (
        rand_augment,
        trivial_augment,
        trivial_augment_wide,
    )
    from justdata.vision.augmentations.color import color_jitter
    from justdata.vision.augmentations.composed import (
        create_global_crops,
        create_local_crops,
    )
    from justdata.vision.augmentations.registry import get_crop_strategy

    nested = (
        ("ra_kwargs", rand_augment, _POLICY_OMIT),
        (
            "ta_kwargs",
            (
                trivial_augment_wide
                if aug["augment_type"] == "trivial_augment_wide"
                else trivial_augment
            ),
            _POLICY_OMIT,
        ),
        ("cj_kwargs", color_jitter, {"image", "seed"}),
        ("gc_kwargs", create_global_crops, {"image", "seed", "crops_number"}),
        ("lc_kwargs", create_local_crops, {"image", "seed", "crops_number"}),
        (
            "crop_kwargs",
            get_crop_strategy(aug["crop_type"]),
            {"image", "seed", "size", "interpolation", "padding", "pad_mode"},
        ),
    )
    for name, fn, omitted in nested:
        value = aug.get(name)
        if value is not None:
            resolve_callable_config(
                fn,
                value,
                path=f"aug_kwargs.{name}",
                omit=omitted,
            )

    if aug["mode"] not in {"sl", "ssl"}:
        raise ValueError("aug_kwargs.mode must be 'sl' or 'ssl'")


def _resolve_classification_config(config, is_training):
    from justdata.vision.stages import EvalViewConfig
    from justdata.vision.tasks.classification import (
        make_augmentations,
        make_late_augmentations,
        make_postprocessing,
        make_preprocessing,
    )

    reject_unknown_keys(config, _VISION_TOP_LEVEL, path="pipeline")
    preproc = resolve_callable_config(
        make_preprocessing, config.get("preproc_kwargs"), path="preproc_kwargs"
    )
    aug = resolve_callable_config(
        make_augmentations, config.get("aug_kwargs"), path="aug_kwargs"
    )
    supplied_late = copy.deepcopy(config.get("laug_kwargs") or {})
    late = resolve_callable_config(
        make_late_augmentations, supplied_late, path="laug_kwargs"
    )
    supplied_post = copy.deepcopy(config.get("postproc_kwargs") or {})
    supplied_post["is_training"] = is_training
    if supplied_late.get("enable", True) and (
        supplied_late.get("mixup_alpha", 0) > 0
        or supplied_late.get("cutmix_alpha", 0) > 0
    ):
        supplied_post.setdefault("one_hot_labels", True)
        supplied_post.setdefault("label_mode", late["label_mode"])
    post = resolve_callable_config(
        make_postprocessing, supplied_post, path="postproc_kwargs"
    )

    for name in (
        "ra_kwargs",
        "ta_kwargs",
        "cj_kwargs",
        "gc_kwargs",
        "lc_kwargs",
        "crop_kwargs",
    ):
        aug[name] = copy.deepcopy(aug[name] or {})
    aug["gc_kwargs"].setdefault("size", aug["image_size"])
    aug["gc_kwargs"].setdefault("scale", (0.4, 1.0))
    aug["lc_kwargs"].setdefault("size", int(aug["image_size"] * 0.425))
    aug["lc_kwargs"].setdefault("scale", (0.05, 0.4))
    if post["image_keys"] is None:
        post["image_keys"] = ["image", "images", "global_crops", "local_crops"]
    if post["label_keys"] is None:
        post["label_keys"] = ["label"]

    _validate_positive(aug["image_size"], path="aug_kwargs.image_size")
    _validate_positive(post["image_size"], path="postproc_kwargs.image_size")
    _validate_classification_nested(aug)
    if post["normalization_mode"] not in {"mean_std", "per_image"}:
        raise ValueError(
            "postproc_kwargs.normalization_mode must be 'mean_std' or 'per_image'"
        )
    if post["eval_view_config"] is not None:
        eval_view = post["eval_view_config"]
        if dataclasses.is_dataclass(eval_view):
            eval_view = dataclasses.asdict(eval_view)
        EvalViewConfig(**dict(eval_view))

    augment_eval = bool(config.get("augment_eval", False))
    apply_augmentation = is_training or augment_eval
    effective_size = (
        post["train_image_size"]
        if is_training and post["train_image_size"] is not None
        else post["image_size"]
    )
    val_resize = post["val_resize_size"]
    if val_resize == "auto":
        val_resize = int(post["image_size"] / 0.875)
    late_active = bool(
        apply_augmentation
        and late["enable"]
        and aug["mode"] != "ssl"
        and late["mode"] != "ssl"
        and (
            late["mixup_alpha"] > 0
            or late["cutmix_alpha"] > 0
            or late["random_erasing_prob"] > 0
        )
    )

    model_input = _image_model_input(
        effective_size,
        permute_image=post["permute_image"],
        normalization=_normalization_contract(post),
    )
    if is_training and aug["mode"] == "ssl":
        model_input = {
            "dtype": "float32",
            "normalization": _normalization_contract(post),
            "outputs": {
                "global_crops": {
                    "layout": "vchw" if post["permute_image"] else "vhwc",
                    "static_shape": [
                        aug["n_global_crops"],
                        3,
                        aug["gc_kwargs"]["size"],
                        aug["gc_kwargs"]["size"],
                    ]
                    if post["permute_image"]
                    else [
                        aug["n_global_crops"],
                        aug["gc_kwargs"]["size"],
                        aug["gc_kwargs"]["size"],
                        3,
                    ],
                },
                "local_crops": {
                    "active": bool(aug["local_crops"]),
                    "layout": "vchw" if post["permute_image"] else "vhwc",
                    "static_shape": [
                        aug["n_local_crops"],
                        3,
                        aug["lc_kwargs"]["size"],
                        aug["lc_kwargs"]["size"],
                    ]
                    if post["permute_image"]
                    else [
                        aug["n_local_crops"],
                        aug["lc_kwargs"]["size"],
                        aug["lc_kwargs"]["size"],
                        3,
                    ],
                },
            },
        }

    resizing_crops = {
        "random_resized",
        "random_resized_hvflip",
        "resize_random_hflip",
    }
    if aug["mode"] == "ssl":
        augment_geometry = {
            "crop": "random_resized",
            "interpolation": "bilinear",
            "antialias": True,
            "operations": {
                "global_crops": {
                    "crop": "random_resized",
                    "interpolation": "bilinear",
                    "antialias": True,
                },
                "local_crops": {
                    "active": bool(aug["local_crops"]),
                    "crop": "random_resized",
                    "interpolation": "bilinear",
                    "antialias": True,
                },
            },
            "padding_mode": None,
        }
    else:
        crop_resizes = aug["crop_type"] in resizing_crops
        augment_geometry = {
            "crop": aug["crop_type"],
            "interpolation": aug["interpolation"] if crop_resizes else None,
            "antialias": crop_resizes,
            "operations": {
                "sample_crop": {
                    "crop": aug["crop_type"],
                    "interpolation": aug["interpolation"] if crop_resizes else None,
                    "antialias": crop_resizes,
                }
            },
            "padding_mode": aug["pad_mode"],
        }

    return {
        "configuration": _configuration_snapshot(config),
        "stages": {
            "preprocess": {"active": True, "config": preproc},
            "augment": {
                "active": apply_augmentation and bool(aug["enable"]),
                "config": aug,
                "geometry": augment_geometry,
            },
            "late_augment": {
                "active": late_active,
                "config": late,
            },
            "postprocess": {
                "active": True,
                "config": post,
                "geometry": {
                    "output_size": effective_size,
                    "validation_resize_size": None if is_training else val_resize,
                    "interpolation": (
                        (
                            post["eval_view_config"].get("interpolation", "bilinear")
                            if isinstance(post["eval_view_config"], dict)
                            else post["eval_view_config"].interpolation
                        )
                        if post["eval_view_config"] is not None and not is_training
                        else "bilinear"
                    ),
                    "antialias": False,
                },
            },
        },
        "model_input": model_input,
        "requirements": {
            "num_classes": bool(
                post["one_hot_labels"]
                or (
                    late_active
                    and (late["mixup_alpha"] > 0 or late["cutmix_alpha"] > 0)
                )
            )
        },
    }


def _resolve_segmentation_config(config, is_training):
    if config.get("geometry_kwargs") is not None:
        return _resolve_dense_segmentation_config(config, is_training)
    _reject_recorded_geometry_options(config)

    from justdata.vision.augmentations.auto import (
        rand_augment,
        trivial_augment,
        trivial_augment_wide,
    )
    from justdata.vision.augmentations.registry import (
        get_augment_strategy,
        get_crop_strategy,
    )
    from justdata.vision.tasks.segmentation import (
        make_augmentations,
        make_late_augmentations,
        make_postprocessing,
        make_preprocessing,
    )

    reject_unknown_keys(
        config, _VISION_TOP_LEVEL | {"geometry_kwargs"}, path="pipeline"
    )
    preproc = resolve_callable_config(
        make_preprocessing, config.get("preproc_kwargs"), path="preproc_kwargs"
    )
    aug = resolve_callable_config(
        make_augmentations, config.get("aug_kwargs"), path="aug_kwargs"
    )
    late = resolve_callable_config(
        make_late_augmentations, config.get("laug_kwargs"), path="laug_kwargs"
    )
    supplied_post = copy.deepcopy(config.get("postproc_kwargs") or {})
    supplied_post["is_training"] = is_training
    post = resolve_callable_config(
        make_postprocessing, supplied_post, path="postproc_kwargs"
    )
    aug["ra_kwargs"] = copy.deepcopy(aug["ra_kwargs"] or {})
    aug["ta_kwargs"] = copy.deepcopy(aug["ta_kwargs"] or {})
    get_crop_strategy(aug["crop_type"])
    get_augment_strategy(aug["augment_type"])
    resolve_callable_config(
        rand_augment,
        aug["ra_kwargs"],
        path="aug_kwargs.ra_kwargs",
        omit=_POLICY_OMIT,
    )
    resolve_callable_config(
        (
            trivial_augment_wide
            if aug["augment_type"] == "trivial_augment_wide"
            else trivial_augment
        ),
        aug["ta_kwargs"],
        path="aug_kwargs.ta_kwargs",
        omit=_POLICY_OMIT,
    )
    _validate_positive(aug["image_size"], path="aug_kwargs.image_size")
    _validate_positive(post["image_size"], path="postproc_kwargs.image_size")
    if post["patch_align"]:
        _validate_positive(post["patch_size"], path="postproc_kwargs.patch_size")
    normalization = _normalization_contract(post | {"normalization_mode": "mean_std"})
    augment_eval = bool(config.get("augment_eval", False))
    apply_augmentation = is_training or augment_eval
    crop_resizes = aug["crop_type"] in {"random_resized", "random_resized_hvflip"}
    output_size = post["image_size"] if not post["patch_align"] or is_training else None
    return {
        "configuration": _configuration_snapshot(config),
        "stages": {
            "preprocess": {"active": True, "config": preproc},
            "augment": {
                "active": apply_augmentation and bool(aug["enable"]),
                "config": aug,
                "geometry": {
                    "crop_image_interpolation": "bilinear" if crop_resizes else None,
                    "crop_mask_interpolation": "nearest" if crop_resizes else None,
                    "policy_image_interpolation": "bilinear",
                    "policy_mask_interpolation": "nearest",
                    "mask_fill_value": aug["mask_fill_value"],
                    "padding_mode": aug["pad_mode"],
                },
            },
            "late_augment": {"active": False, "config": late},
            "postprocess": {
                "active": True,
                "config": post,
                "geometry": {
                    "patch_align": post["patch_align"] and not is_training,
                    "image_interpolation": (
                        "bicubic"
                        if post["patch_align"] and not is_training
                        else "bilinear"
                    ),
                    "mask_interpolation": "nearest",
                    "antialias": False,
                    "patch_padding": "bottom_right_constant",
                    "mask_padding_value": 0,
                },
            },
        },
        "model_input": _image_model_input(
            output_size,
            permute_image=post["permute_image"],
            normalization=normalization,
        ),
        "requirements": {"num_classes": False},
    }


@register_pipeline(
    "vision/classification", config_resolver=_resolve_classification_config
)
def default_classification_pipeline(
    preproc_kwargs: dict | None = None,
    aug_kwargs: dict | None = None,
    laug_kwargs: dict | None = None,
    postproc_kwargs: dict | None = None,
    **kwargs,
) -> PipelineFuncs:
    preproc_kwargs = preproc_kwargs or {}
    aug_kwargs = aug_kwargs or {}
    laug_kwargs = laug_kwargs or {}
    postproc_kwargs = postproc_kwargs or {}

    if laug_kwargs.get("enable", True) and (
        laug_kwargs.get("mixup_alpha", 0) > 0 or laug_kwargs.get("cutmix_alpha", 0) > 0
    ):
        postproc_kwargs.setdefault("one_hot_labels", True)
        if "label_mode" in laug_kwargs:
            postproc_kwargs.setdefault("label_mode", laug_kwargs["label_mode"])

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


@register_pipeline("vision/segmentation", config_resolver=_resolve_segmentation_config)
def default_segmentation_pipeline(
    preproc_kwargs: dict | None = None,
    aug_kwargs: dict | None = None,
    laug_kwargs: dict | None = None,
    postproc_kwargs: dict | None = None,
    geometry_kwargs: dict | None = None,
    **kwargs,
) -> PipelineFuncs:
    """Build segmentation stages, with optional recorded dense geometry."""
    if geometry_kwargs is not None:
        return _build_dense_segmentation_pipeline(
            kwargs
            | {
                "geometry_kwargs": geometry_kwargs,
                "preproc_kwargs": preproc_kwargs,
                "aug_kwargs": aug_kwargs,
                "laug_kwargs": laug_kwargs,
                "postproc_kwargs": postproc_kwargs,
            }
        )
    _reject_recorded_geometry_options(kwargs | {"postproc_kwargs": postproc_kwargs})

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


def _resolve_dense_segmentation_config(config, is_training):
    from justdata.vision.geometry import DenseGeometryConfig, _train_input_size
    from justdata.vision.tasks.segmentation import make_dense_postprocessing

    reject_unknown_keys(
        config,
        _VISION_TOP_LEVEL | {"geometry_kwargs", *_RECORDED_GEOMETRY_OPTIONS},
        path="pipeline",
    )
    for name in ("preproc_kwargs", "aug_kwargs", "laug_kwargs"):
        stage = config.get(name)
        if stage is not None and not isinstance(stage, Mapping):
            raise TypeError(f"{name} must be a mapping or None")
        if stage:
            raise ValueError(
                f"{name} cannot be combined with geometry_kwargs; "
                "configure geometry and color_jitter_kwargs explicitly"
            )
    if config.get("augment_eval", False):
        raise ValueError(
            "dense segmentation evaluation must use deterministic geometry"
        )
    geometry = dataclasses.asdict(
        DenseGeometryConfig(
            **resolve_callable_config(
                DenseGeometryConfig,
                config.get("geometry_kwargs"),
                path="geometry_kwargs",
            )
        )
    )
    post = resolve_callable_config(
        make_dense_postprocessing,
        (config.get("postproc_kwargs") or {}) | {"is_training": is_training},
        path="postproc_kwargs",
        omit={"geometry"},
    )
    normalization = _validate_recorded_postprocess(
        post, ("normalize_image", "permute_image", "emit_semantic_targets")
    )
    keep_original = config.get("keep_original_mask", True)
    if type(keep_original) is not bool:
        raise ValueError("keep_original_mask must be boolean")
    jitter, photometric = _resolve_color_options(config)
    geometry_contract = _recorded_geometry_contract(geometry, is_training) | {
        "image_padding_domain": "rgb_0_255_before_normalization",
        "mask_padding_value": geometry["ignore_value"],
    }
    size = (
        _train_input_size(geometry["train_crop_size"], geometry["patch_size"])
        if is_training
        else None
    )
    return {
        "configuration": _configuration_snapshot(config),
        "stages": {
            "preprocess": {
                "active": True,
                "config": {
                    "class_values": geometry["class_values"],
                    "ignore_value": geometry["ignore_value"],
                    "keep_original_mask": keep_original,
                },
            },
            "augment": {
                "active": is_training,
                "config": {
                    "geometry": geometry,
                    "color_jitter_kwargs": jitter,
                    "photometric_kwargs": photometric,
                },
                "geometry": geometry_contract if is_training else None,
            },
            "late_augment": {"active": False, "config": {}},
            "postprocess": {
                "active": True,
                "config": post,
                "geometry": None if is_training else geometry_contract,
                "target_set": (
                    {
                        "version": 1,
                        "class_values": geometry["class_values"],
                        "ignore_value": geometry["ignore_value"],
                        "capacity": len(geometry["class_values"]),
                        "slot_order": "class_values",
                        "pixel_validity": "source_annotation_non_ignore_and_example",
                    }
                    if post["emit_semantic_targets"]
                    else None
                ),
            },
        },
        "model_input": _image_model_input(
            size, permute_image=post["permute_image"], normalization=normalization
        ),
        "requirements": {"num_classes": False},
    }


def _build_dense_segmentation_pipeline(config) -> PipelineFuncs:
    """Build the recorded geometry mode of the segmentation pipeline."""
    from justdata.vision.geometry import DenseGeometryConfig
    from justdata.vision.tasks.segmentation import (
        make_dense_augmentations,
        make_dense_postprocessing,
        make_dense_preprocessing,
        make_late_augmentations,
    )

    is_training = (config.get("postproc_kwargs") or {}).get("is_training", False)
    resolved = _resolve_dense_segmentation_config(config, is_training)
    stages = resolved["stages"]
    geometry = DenseGeometryConfig(**stages["augment"]["config"]["geometry"])
    return (
        make_dense_preprocessing(**stages["preprocess"]["config"]),
        make_dense_augmentations(
            geometry,
            stages["augment"]["config"]["color_jitter_kwargs"],
            stages["augment"]["config"]["photometric_kwargs"],
        ),
        make_late_augmentations(),
        make_dense_postprocessing(geometry, **stages["postprocess"]["config"]),
    )


def _panoptic_options(class_values, thing_class_values, max_segments, void_value=0):
    from justdata.vision.encodings.panoptic_targets import _values

    classes, things = _values(
        class_values, thing_class_values, void_value, max_segments
    )
    return {
        "class_values": classes,
        "thing_class_values": things,
        "void_value": void_value,
        "max_segments": max_segments,
    }


def _resolve_panoptic_config(config, is_training):
    from justdata.vision.geometry import PanopticGeometryConfig, _train_input_size
    from justdata.vision.tasks.panoptic import make_panoptic_postprocessing

    reject_unknown_keys(
        config,
        _VISION_TOP_LEVEL
        | {
            "geometry_kwargs",
            "panoptic_kwargs",
            "color_jitter_kwargs",
            "photometric_kwargs",
            "keep_original_annotations",
        },
        path="pipeline",
    )
    for name in ("preproc_kwargs", "aug_kwargs", "laug_kwargs"):
        value = config.get(name)
        if value is not None and (not isinstance(value, Mapping) or value):
            raise ValueError(f"{name} must be empty for panoptic segmentation")
    if config.get("augment_eval", False):
        raise ValueError("panoptic evaluation must use deterministic geometry")
    geometry = dataclasses.asdict(
        PanopticGeometryConfig(
            **resolve_callable_config(
                PanopticGeometryConfig,
                config.get("geometry_kwargs"),
                path="geometry_kwargs",
            )
        )
    )
    panoptic = resolve_callable_config(
        _panoptic_options, config.get("panoptic_kwargs"), path="panoptic_kwargs"
    )
    panoptic = _panoptic_options(**panoptic)
    post = resolve_callable_config(
        make_panoptic_postprocessing,
        (config.get("postproc_kwargs") or {}) | {"is_training": is_training},
        path="postproc_kwargs",
        omit={"config", "geometry"},
    )
    normalization = _validate_recorded_postprocess(
        post, ("normalize_image", "permute_image", "emit_panoptic_targets")
    )
    keep_original = config.get("keep_original_annotations", False)
    if type(keep_original) is not bool:
        raise ValueError("keep_original_annotations must be boolean")
    jitter, photometric = _resolve_color_options(config)
    size = (
        _train_input_size(geometry["train_crop_size"], geometry["patch_size"])
        if is_training
        else None
    )
    geometry_contract = _recorded_geometry_contract(geometry, is_training) | {
        "void_value": panoptic["void_value"],
    }
    return {
        "configuration": {
            "geometry_kwargs": geometry,
            "panoptic_kwargs": panoptic,
            "color_jitter_kwargs": jitter,
            "photometric_kwargs": photometric,
            "keep_original_annotations": keep_original,
            "postproc_kwargs": post,
        },
        "stages": {
            "preprocess": {
                "active": True,
                "config": panoptic
                | {
                    "keep_original_annotations": keep_original,
                },
            },
            "augment": {
                "active": is_training,
                "config": {
                    "geometry": geometry,
                    "panoptic": panoptic,
                    "color_jitter_kwargs": jitter,
                    "photometric_kwargs": photometric,
                },
                "geometry": geometry_contract if is_training else None,
            },
            "late_augment": {"active": False, "config": {}},
            "postprocess": {
                "active": True,
                "config": post,
                "geometry": None if is_training else geometry_contract,
                "target_set": {
                    "version": 1,
                    "capacity": panoptic["max_segments"],
                    "class_values": panoptic["class_values"],
                    "thing_class_values": panoptic["thing_class_values"],
                    "void_value": panoptic["void_value"],
                    "crowd_policy": "exclude_supervision",
                    "stuff_policy": "merge_by_category",
                }
                if post["emit_panoptic_targets"]
                else None,
            },
        },
        "model_input": _image_model_input(
            size, permute_image=post["permute_image"], normalization=normalization
        ),
        "requirements": {"num_classes": False},
    }


@register_pipeline(
    "vision/panoptic_segmentation", config_resolver=_resolve_panoptic_config
)
def default_panoptic_pipeline(
    geometry_kwargs: dict | None = None,
    panoptic_kwargs: dict | None = None,
    color_jitter_kwargs: dict | None = None,
    photometric_kwargs: dict | None = None,
    keep_original_annotations: bool = False,
    postproc_kwargs: dict | None = None,
    **kwargs,
) -> PipelineFuncs:
    from justdata.vision.geometry import PanopticGeometryConfig
    from justdata.vision.tasks.panoptic import (
        make_panoptic_augmentations,
        make_panoptic_postprocessing,
        make_panoptic_preprocessing,
    )
    from justdata.vision.tasks.segmentation import make_late_augmentations

    config = kwargs | {
        "geometry_kwargs": geometry_kwargs,
        "panoptic_kwargs": panoptic_kwargs,
        "color_jitter_kwargs": color_jitter_kwargs,
        "photometric_kwargs": photometric_kwargs,
        "keep_original_annotations": keep_original_annotations,
        "postproc_kwargs": postproc_kwargs,
    }
    is_training = (postproc_kwargs or {}).get("is_training", False)
    resolved = _resolve_panoptic_config(config, is_training)
    stages = resolved["stages"]
    panoptic = resolved["configuration"]["panoptic_kwargs"]
    geometry = PanopticGeometryConfig(**resolved["configuration"]["geometry_kwargs"])
    return (
        make_panoptic_preprocessing(**stages["preprocess"]["config"]),
        make_panoptic_augmentations(
            panoptic,
            geometry,
            color_jitter_kwargs=resolved["configuration"]["color_jitter_kwargs"],
            photometric_kwargs=resolved["configuration"]["photometric_kwargs"],
        ),
        make_late_augmentations(),
        make_panoptic_postprocessing(
            panoptic, geometry, **stages["postprocess"]["config"]
        ),
    )

import copy
import dataclasses

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
        (
            "ra_kwargs",
            rand_augment,
            {"image", "seed", "bboxes", "segmentation_mask", "segmentation_fill_value"},
        ),
        (
            "ta_kwargs",
            (
                trivial_augment_wide
                if aug["augment_type"] == "trivial_augment_wide"
                else trivial_augment
            ),
            {"image", "seed", "bboxes", "segmentation_mask", "segmentation_fill_value"},
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
    layout = "bchw" if post["permute_image"] else "bhwc"
    shape = (
        [3, effective_size, effective_size]
        if post["permute_image"]
        else [effective_size, effective_size, 3]
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

    model_input = {
        "output_key": "image",
        "layout": layout,
        "dtype": "float32",
        "static_shape": shape,
        "normalization": _normalization_contract(post),
    }
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
        "configuration": {
            key: copy.deepcopy(value)
            for key, value in config.items()
            if key != "model_input"
        },
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

    reject_unknown_keys(config, _VISION_TOP_LEVEL, path="pipeline")
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
        omit={
            "image",
            "seed",
            "bboxes",
            "segmentation_mask",
            "segmentation_fill_value",
        },
    )
    resolve_callable_config(
        (
            trivial_augment_wide
            if aug["augment_type"] == "trivial_augment_wide"
            else trivial_augment
        ),
        aug["ta_kwargs"],
        path="aug_kwargs.ta_kwargs",
        omit={
            "image",
            "seed",
            "bboxes",
            "segmentation_mask",
            "segmentation_fill_value",
        },
    )
    _validate_positive(aug["image_size"], path="aug_kwargs.image_size")
    _validate_positive(post["image_size"], path="postproc_kwargs.image_size")
    if post["patch_align"]:
        _validate_positive(post["patch_size"], path="postproc_kwargs.patch_size")
    normalization = _normalization_contract(post | {"normalization_mode": "mean_std"})
    augment_eval = bool(config.get("augment_eval", False))
    apply_augmentation = is_training or augment_eval
    layout = "bchw" if post["permute_image"] else "bhwc"
    static_shape = None
    if not post["patch_align"] or is_training:
        static_shape = (
            [3, post["image_size"], post["image_size"]]
            if post["permute_image"]
            else [post["image_size"], post["image_size"], 3]
        )
    return {
        "configuration": {
            key: copy.deepcopy(value)
            for key, value in config.items()
            if key != "model_input"
        },
        "stages": {
            "preprocess": {"active": True, "config": preproc},
            "augment": {
                "active": apply_augmentation and bool(aug["enable"]),
                "config": aug,
                "geometry": {
                    "crop_image_interpolation": (
                        "bilinear"
                        if aug["crop_type"]
                        in {"random_resized", "random_resized_hvflip"}
                        else None
                    ),
                    "crop_mask_interpolation": (
                        "nearest"
                        if aug["crop_type"]
                        in {"random_resized", "random_resized_hvflip"}
                        else None
                    ),
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
        "model_input": {
            "output_key": "image",
            "layout": layout,
            "dtype": "float32",
            "static_shape": static_shape,
            "normalization": normalization,
        },
        "requirements": {"num_classes": False},
    }


@register_pipeline(
    "vision/classification", config_resolver=_resolve_classification_config
)
def default_classification_pipeline(
    preproc_kwargs: dict = None,
    aug_kwargs: dict = None,
    laug_kwargs: dict = None,
    postproc_kwargs: dict = None,
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
    preproc_kwargs: dict = None,
    aug_kwargs: dict = None,
    laug_kwargs: dict = None,
    postproc_kwargs: dict = None,
    **kwargs,
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

import copy
from typing import Any, Dict

from justdata.core.presets import (
    ResolvedPreset,
    get_dataset_presets as _get_dataset_presets,
    get_resolved_preset as _get_resolved_preset,
    merge_with_presets as _merge_with_presets,
    register_preset as _register_preset,
)


def _normalization_contract(postproc_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    if not postproc_kwargs.get("normalize_image", True):
        return {"kind": "none"}

    normalization_mode = postproc_kwargs.get("normalization_mode", "mean_std")
    if normalization_mode == "per_image":
        return {"kind": "per_image", "per_channel": True}
    if normalization_mode != "mean_std":
        return {"kind": normalization_mode}

    mean, std = postproc_kwargs.get(
        "normalization_params",
        ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    )
    return {
        "kind": "mean_std",
        "mean": tuple(mean),
        "std": tuple(std),
    }


def _with_model_input_contract(config: Dict[str, Any]) -> Dict[str, Any]:
    resolved = copy.deepcopy(config)
    postproc_kwargs = resolved.get("postproc_kwargs", {})
    image_size = postproc_kwargs.get("image_size")
    permute_image = postproc_kwargs.get("permute_image", True)
    layout = "bchw" if permute_image else "bhwc"
    static_shape = None
    if image_size is not None:
        static_shape = (
            (3, image_size, image_size)
            if permute_image
            else (image_size, image_size, 3)
        )

    contract = {
        "output_key": "image",
        "layout": layout,
        "dtype": "float32",
        "static_shape": static_shape,
        "normalization": _normalization_contract(postproc_kwargs),
    }
    contract.update(resolved.get("model_input", {}))
    resolved["model_input"] = contract
    return resolved


def register_preset(dataset: str, config: Dict[str, Any]):
    _register_preset(dataset, _with_model_input_contract(config), modality="vision")


def get_dataset_presets(dataset: str) -> Dict[str, Any]:
    return _get_dataset_presets(dataset, modality="vision")


def get_resolved_preset(dataset: str) -> ResolvedPreset:
    return _get_resolved_preset(dataset, modality="vision")


# ImageNet normalization stats (shared across ImageNet presets)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_IMAGENET_NORM = (_IMAGENET_MEAN, _IMAGENET_STD)

_CIFAR10_NORM = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
_CIFAR100_NORM = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))

# CIFAR-10 / CIFAR-100
# Modern branch: TrivialAugment, zero-padding crop.
# RandomCrop(32, padding=4, padding_mode='zeros') -> RandomHorizontalFlip
register_preset(
    "cifar100",
    {
        "preproc_kwargs": {},
        "aug_kwargs": {
            "image_size": 32,
            "crop_type": "random_pad",
            "padding": 4,
            "pad_mode": "CONSTANT",
            "augment_type": "trivial_augment_wide",
            "ra_kwargs": {
                "num_layers": 2,
                "magnitude": 9.0,
                "cutout_const": 14.0,
                "translate_const": 14.0,
            },
            "gc_kwargs": {"size": 32, "scale": (0.32, 1.0)},
            "lc_kwargs": {"size": 16, "scale": (0.05, 0.32)},
        },
        "laug_kwargs": {
            "mixup_alpha": 0.8,
            "cutmix_alpha": 1.0,
            "prob": 1.0,
            "switch_prob": 0.5,
            "random_erasing_prob": 0.25,
        },
        "postproc_kwargs": {
            "image_size": 32,
            "val_resize_size": None,
            "normalization_params": _CIFAR100_NORM,
        },
    },
)

register_preset(
    "cifar",
    {
        "preproc_kwargs": {},
        "aug_kwargs": {
            "image_size": 32,
            "crop_type": "random_pad",
            "padding": 4,
            "pad_mode": "CONSTANT",
            "augment_type": "trivial_augment_wide",
            "ra_kwargs": {
                "num_layers": 2,
                "magnitude": 9.0,
                "cutout_const": 14.0,
                "translate_const": 14.0,
            },
            "gc_kwargs": {"size": 32, "scale": (0.32, 1.0)},
            "lc_kwargs": {"size": 16, "scale": (0.05, 0.32)},
        },
        "laug_kwargs": {
            "mixup_alpha": 0.8,
            "cutmix_alpha": 1.0,
            "prob": 1.0,
            "switch_prob": 0.5,
            "random_erasing_prob": 0.25,
        },
        "postproc_kwargs": {
            "image_size": 32,
            "val_resize_size": None,
            "normalization_params": _CIFAR10_NORM,
        },
    },
)

# KITTI road segmentation. Keep this separate from the ImageNet classification
# default because segmentation augmentation accepts a deliberately smaller API.
register_preset(
    "kitti_road",
    {
        "preproc_kwargs": {},
        "aug_kwargs": {
            "image_size": 224,
            "crop_type": "random_resized",
            "augment_type": "rand_augment",
        },
        "laug_kwargs": {},
        "postproc_kwargs": {
            "image_size": 224,
            "normalization_params": _IMAGENET_NORM,
        },
    },
)

# ImageNet default  (ViT / ConvNeXt modern recipe)
# RandAugment(n=2, m=9), bicubic interpolation, 0.875 crop ratio validation.
register_preset(
    "_default",
    {
        "preproc_kwargs": {},
        "aug_kwargs": {
            "image_size": 224,
            "interpolation": "bicubic",
            "augment_type": "rand_augment",
            "ra_kwargs": {
                "num_layers": 2,
                "magnitude": 9.0,
                "cutout_const": 40.0,
                "translate_const": 101.0,
            },
            "gc_kwargs": {"size": 224, "scale": (0.32, 1.0)},
            "lc_kwargs": {"size": 96, "scale": (0.05, 0.32)},
        },
        "laug_kwargs": {
            "mixup_alpha": 0.8,
            "cutmix_alpha": 1.0,
            "prob": 1.0,
            "switch_prob": 0.5,
            "random_erasing_prob": 0.25,
        },
        "postproc_kwargs": {
            "image_size": 224,
            "normalization_params": _IMAGENET_NORM,
        },
    },
)

# ImageNet — Legacy ResNet branch (ColorJitter, no RandAugment)
# ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
register_preset(
    "imagenet_resnet",
    {
        "preproc_kwargs": {},
        "aug_kwargs": {
            "image_size": 224,
            "interpolation": "bicubic",
            "augment_type": "color_jitter",
            "cj_kwargs": {
                "brightness": 0.4,
                "contrast": 0.4,
                "saturation": 0.4,
                "hue": 0.1,
            },
        },
        "laug_kwargs": {
            "mixup_alpha": 0.2,
            "cutmix_alpha": 1.0,
            "prob": 1.0,
            "switch_prob": 0.5,
            "random_erasing_prob": 0.25,
            "bce_target": False,
        },
        "postproc_kwargs": {
            "image_size": 224,
            "normalization_params": _IMAGENET_NORM,
        },
    },
)

# RSB A1 — Heavy (ResNet-152/200, 600 epochs, BCE loss)
register_preset(
    "imagenet_a1",
    {
        "preproc_kwargs": {},
        "aug_kwargs": {
            "image_size": 224,
            "interpolation": "bicubic",
            "augment_type": "rand_augment",
            "ra_kwargs": {
                "num_layers": 2,
                "magnitude": 7.0,
                "cutout_const": 40.0,
                "translate_const": 101.0,
            },
        },
        "laug_kwargs": {
            "mixup_alpha": 0.2,
            "cutmix_alpha": 1.0,
            "prob": 1.0,
            "switch_prob": 0.5,
            "random_erasing_prob": 0.35,
        },
        "postproc_kwargs": {
            "image_size": 224,
            # crop_pct=1.0 -> no short-side resize + center crop, just resize
            "val_resize_size": None,
            "normalization_params": _IMAGENET_NORM,
        },
    },
)

# RSB A2 — Moderate (ResNet-50, 300 epochs, BCE loss)
register_preset(
    "imagenet_a2",
    {
        "preproc_kwargs": {},
        "aug_kwargs": {
            "image_size": 224,
            "interpolation": "bicubic",
            "augment_type": "rand_augment",
            "ra_kwargs": {
                "num_layers": 2,
                "magnitude": 6.0,
                "cutout_const": 40.0,
                "translate_const": 101.0,
            },
        },
        "laug_kwargs": {
            "mixup_alpha": 0.2,
            "cutmix_alpha": 1.0,
            "prob": 1.0,
            "switch_prob": 0.5,
            "random_erasing_prob": 0.25,
        },
        "postproc_kwargs": {
            "image_size": 224,
            # crop_pct=1.0 -> no short-side resize + center crop, just resize
            "val_resize_size": None,
            "normalization_params": _IMAGENET_NORM,
        },
    },
)

# RSB A3 — Light (ResNet-50 fast / ResNet-18, 100 epochs, CE loss)
# FixRes: train at 160, validate at 224 (resize to ~236, center crop 224).
register_preset(
    "imagenet_a3",
    {
        "preproc_kwargs": {},
        "aug_kwargs": {
            "image_size": 160,
            "interpolation": "bicubic",
            "augment_type": "rand_augment",
            "ra_kwargs": {
                "num_layers": 2,
                "magnitude": 6.0,
                "cutout_const": 40.0,
                "translate_const": 72.0,
            },
        },
        "laug_kwargs": {
            "mixup_alpha": 0.1,
            "cutmix_alpha": 1.0,
            "prob": 1.0,
            "switch_prob": 0.5,
            "random_erasing_prob": 0.0,
            "bce_target": False,
        },
        "postproc_kwargs": {
            "image_size": 224,
            # FixRes: train at 160, validate at 224 (resize to 236, center crop 224)
            "train_image_size": 160,
            "val_resize_size": 236,
            "normalization_params": _IMAGENET_NORM,
        },
    },
)

# DINOv2 Self-Supervised Learning
# Asymmetric multi-crop with per-crop blur/solarize probabilities.
register_preset(
    "dinov2",
    {
        "preproc_kwargs": {},
        "aug_kwargs": {
            "image_size": 224,
            "mode": "ssl",
            "n_global_crops": 2,
            "n_local_crops": 8,
            "gc_kwargs": {
                "size": 224,
                "scale": (0.32, 1.0),
                "brightness": 0.4,
                "contrast": 0.4,
                "saturation": 0.2,
                "hue": 0.1,
                "p_color_jitter": 0.8,
                "p_grayscale": 0.2,
                # Per-crop asymmetry (DINOv2):
                # crop 0: blur=1.0, solar=0.0
                # crop 1: blur=0.1, solar=0.2
                "p_gaussian_blur": (1.0, 0.1),
                "p_solarize": (0.0, 0.2),
            },
            "lc_kwargs": {
                "size": 96,
                "scale": (0.05, 0.32),
                "brightness": 0.4,
                "contrast": 0.4,
                "saturation": 0.2,
                "hue": 0.1,
                "p_color_jitter": 0.8,
                "p_grayscale": 0.2,
                "p_gaussian_blur": 0.5,
                "p_solarize": 0.0,
            },
        },
        "laug_kwargs": {},
        "postproc_kwargs": {
            "image_size": 224,
            "normalization_params": _IMAGENET_NORM,
        },
    },
)


def _register_wilds_preset(
    dataset: str,
    image_size: int,
    *,
    aug_kwargs: Dict[str, Any] | None = None,
    laug_kwargs: Dict[str, Any] | None = None,
    postproc_kwargs: Dict[str, Any] | None = None,
):
    resolved_aug_kwargs = {
        "enable": False,
        "image_size": image_size,
    }
    if aug_kwargs is not None:
        resolved_aug_kwargs.update(aug_kwargs)

    resolved_laug_kwargs = {
        "enable": False,
    }
    if laug_kwargs is not None:
        resolved_laug_kwargs.update(laug_kwargs)

    resolved_postproc_kwargs = {
        "image_size": image_size,
        "val_resize_size": None,
        "normalization_params": _IMAGENET_NORM,
    }
    if postproc_kwargs is not None:
        resolved_postproc_kwargs.update(postproc_kwargs)
        if (
            postproc_kwargs.get("normalization_mode") == "per_image"
            and "normalization_params" not in postproc_kwargs
        ):
            resolved_postproc_kwargs.pop("normalization_params", None)

    register_preset(
        dataset,
        {
            "preproc_kwargs": {},
            "aug_kwargs": resolved_aug_kwargs,
            "laug_kwargs": resolved_laug_kwargs,
            "postproc_kwargs": resolved_postproc_kwargs,
        },
    )


_register_wilds_preset("wilds:camelyon17", 96)
_register_wilds_preset("wilds:fmow", 224)
_register_wilds_preset("wilds:iwildcam", 448)
_register_wilds_preset(
    "wilds:rxrx1",
    256,
    aug_kwargs={
        "enable": True,
        "crop_type": "random_rot90_hflip",
        "augment_type": "none",
    },
    postproc_kwargs={
        "normalization_mode": "per_image",
    },
)

_register_wilds_preset(
    "wilds:camelyon17_strong",
    96,
    aug_kwargs={
        "enable": True,
        "crop_type": "random_rot90_hflip",
        "augment_type": "color_jitter",
        "cj_kwargs": {
            "brightness": 0.2,
            "contrast": 0.2,
            "saturation": 0.2,
            "hue": 0.05,
        },
    },
)
_register_wilds_preset(
    "wilds:fmow_strong",
    224,
    aug_kwargs={
        "enable": True,
        "crop_type": "resize_random_hflip",
        "augment_type": "rand_augment",
        "ra_kwargs": {
            "num_layers": 2,
            "magnitude": 9.0,
            "cutout_const": 40.0,
            "translate_const": 101.0,
        },
    },
)
_register_wilds_preset(
    "wilds:iwildcam_strong",
    448,
    aug_kwargs={
        "enable": True,
        "crop_type": "resize_random_hflip",
        "augment_type": "rand_augment",
        "ra_kwargs": {
            "num_layers": 2,
            "magnitude": 9.0,
            "cutout_const": 80.0,
            "translate_const": 203.0,
        },
    },
)
_register_wilds_preset(
    "wilds:rxrx1_strong",
    256,
    aug_kwargs={
        "enable": True,
        "crop_type": "random_rot90_hflip",
        "augment_type": "none",
    },
    laug_kwargs={
        "enable": True,
        "mixup_alpha": 0.0,
        "cutmix_alpha": 0.0,
        "random_erasing_prob": 0.25,
    },
    postproc_kwargs={
        "normalization_mode": "per_image",
    },
)


def merge_with_presets(dataset: str, user_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return _merge_with_presets(dataset, user_kwargs, modality="vision")

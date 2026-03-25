import copy
import threading
from typing import Any, Dict

_PRESETS: Dict[str, Dict[str, Any]] = {}
_PRESET_LOCK = threading.Lock()


def register_preset(dataset: str, config: Dict[str, Any]):
    with _PRESET_LOCK:
        _PRESETS[dataset.lower()] = config


def get_dataset_presets(dataset: str) -> Dict[str, Any]:
    dataset = dataset.lower()
    # Check exact match first
    if dataset in _PRESETS:
        return _PRESETS[dataset]
    # Check prefix match, longest first so "cifar100" beats "cifar"
    # for a query like "cifar100_corrupted".
    for key, val in sorted(_PRESETS.items(), key=lambda kv: len(kv[0]), reverse=True):
        if key != "_default" and dataset.startswith(key):
            return val
    return _PRESETS.get("_default", {})


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


def merge_with_presets(dataset: str, user_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Smartly merges user-provided kwargs with dataset-specific presets.
    If a user kwarg matches the global/ImageNet default, it is assumed to be an
    unmodified configuration default and the dataset-specific preset takes precedence.
    If it differs, it's treated as a conscious user override.
    """

    dataset_presets = copy.deepcopy(get_dataset_presets(dataset))
    default_presets = get_dataset_presets("imagenet")

    def smart_merge(base, user, default):
        result = base.copy()
        for k, v in user.items():
            if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                result[k] = smart_merge(
                    base[k], v, default.get(k, {}) if isinstance(default, dict) else {}
                )
            else:
                default_val = default.get(k) if isinstance(default, dict) else None
                # If the user value differs from the default preset, OR it's not in the default preset,
                # we consider it an explicit user override.
                # Tuples/Lists can be tricky, so let's normalize them for comparison
                val_to_compare = tuple(v) if isinstance(v, list) else v
                def_to_compare = (
                    tuple(default_val) if isinstance(default_val, list) else default_val
                )

                # Check for nested tuples inside like normalization_params
                if (
                    isinstance(val_to_compare, tuple)
                    and len(val_to_compare) > 0
                    and isinstance(val_to_compare[0], list)
                ):
                    val_to_compare = tuple(
                        tuple(x) if isinstance(x, list) else x for x in val_to_compare
                    )
                if (
                    isinstance(def_to_compare, tuple)
                    and len(def_to_compare) > 0
                    and isinstance(def_to_compare[0], list)
                ):
                    def_to_compare = tuple(
                        tuple(x) if isinstance(x, list) else x for x in def_to_compare
                    )

                if k not in default or val_to_compare != def_to_compare:
                    result[k] = v
        return result

    # The user_kwargs generally has keys like 'preproc_kwargs', 'aug_kwargs', etc.
    # So we do a top-level smart merge.
    return smart_merge(dataset_presets, user_kwargs, default_presets)

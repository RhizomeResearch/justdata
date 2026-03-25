from typing import Any, Dict

_PRESETS: Dict[str, Dict[str, Any]] = {}


def register_preset(dataset: str, config: Dict[str, Any]):
    _PRESETS[dataset.lower()] = config


def get_dataset_presets(dataset: str) -> Dict[str, Any]:
    dataset = dataset.lower()
    # Check exact match first
    if dataset in _PRESETS:
        return _PRESETS[dataset]
    # Check prefix match (e.g. "cifar" matches "cifar10")
    for key, val in _PRESETS.items():
        if key != "_default" and dataset.startswith(key):
            return val
    return _PRESETS.get("_default", {})


register_preset(
    "cifar100",
    {
        "preproc_kwargs": {},
        "aug_kwargs": {
            "image_size": 32,
            "crop_type": "random_pad",
            "padding": 4,
            "pad_mode": "REFLECT",
            "augment_type": "trivial_augment",
            "ra_kwargs": {"cutout_const": 14.0, "translate_const": 14.0},
            "ta_kwargs": {"cutout_const": 14.0, "translate_const": 14.0},
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
            "normalization_params": (
                (0.5071, 0.4867, 0.4408),
                (0.2675, 0.2565, 0.2761),
            ),
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
            "pad_mode": "REFLECT",
            "augment_type": "trivial_augment",
            "ra_kwargs": {"cutout_const": 14.0, "translate_const": 14.0},
            "ta_kwargs": {"cutout_const": 14.0, "translate_const": 14.0},
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
            "normalization_params": (
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010),
            ),
        },
    },
)

register_preset(
    "_default",
    {
        "preproc_kwargs": {},
        "aug_kwargs": {
            "image_size": 224,
            "ra_kwargs": {
                "magnitude": 7.0,
                "cutout_const": 40.0,
                "translate_const": 100.0,
            },
            "gc_kwargs": {"size": 224, "scale": (0.32, 1.0)},
            "lc_kwargs": {"size": 96, "scale": (0.05, 0.32)},
        },
        "laug_kwargs": {},
        "postproc_kwargs": {
            "image_size": 224,
            "normalization_params": (
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
            ),
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
    dataset_presets = get_dataset_presets(dataset)
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

import numpy as np
import pytest
import tensorflow as tf

import justdata.vision  # noqa: F401
from justdata.core.registry import get_pipeline
from justdata.vision.presets import get_dataset_presets, get_resolved_preset


EXPECTED_HASHES = {
    "cifar10": "27ea941b3a5d2604",
    "cifar100": "b3500b7104e5af73",
    "dinov2": "c346275bdd4b3906",
    "imagenet": "5366c87112cd2682",
    "imagenet_a1": "295a30e69b79c069",
    "imagenet_a2": "70f66a6fe3989e0d",
    "imagenet_a3": "0067216f95ff9557",
    "imagenet_resnet": "aba3867d4392cadf",
    "wilds:camelyon17": "f1b68d55baea105f",
    "wilds:camelyon17_strong": "212f1d01e33a9fac",
    "wilds:fmow": "2da092187b08c556",
    "wilds:fmow_strong": "d9d7c85bdded7418",
    "wilds:iwildcam": "5c4ac991be450aea",
    "wilds:iwildcam_strong": "44898930bce2bc62",
    "wilds:rxrx1": "38abee4af0d7aca2",
    "wilds:rxrx1_strong": "3c5a5609bff35f09",
}

EXPECTED_CONTRACTS = {
    "cifar10": {
        "image_size": 32,
        "train_image_size": None,
        "train_crop": "random_pad",
        "padding": 4,
        "val_resize_size": None,
        "normalization": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        "num_classes": 10,
    },
    "cifar100": {
        "image_size": 32,
        "train_image_size": None,
        "train_crop": "random_pad",
        "padding": 4,
        "val_resize_size": None,
        "normalization": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        "num_classes": 100,
    },
    "imagenet": {
        "image_size": 224,
        "train_image_size": None,
        "train_crop": "random_resized",
        "padding": 4,
        "val_resize_size": "auto",
        "normalization": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        "num_classes": 1000,
    },
    "imagenet_resnet": {
        "image_size": 224,
        "train_image_size": None,
        "train_crop": "random_resized",
        "padding": 4,
        "val_resize_size": "auto",
        "normalization": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        "num_classes": 1000,
    },
    "imagenet_a1": {
        "image_size": 224,
        "train_image_size": None,
        "train_crop": "random_resized",
        "padding": 4,
        "val_resize_size": None,
        "normalization": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        "num_classes": 1000,
    },
    "imagenet_a2": {
        "image_size": 224,
        "train_image_size": None,
        "train_crop": "random_resized",
        "padding": 4,
        "val_resize_size": None,
        "normalization": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        "num_classes": 1000,
    },
    "imagenet_a3": {
        "image_size": 224,
        "train_image_size": 160,
        "train_crop": "random_resized",
        "padding": 4,
        "val_resize_size": 236,
        "normalization": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        "num_classes": 1000,
    },
    "dinov2": {
        "image_size": 224,
        "train_image_size": None,
        "train_crop": "ssl_multi_crop",
        "padding": 4,
        "val_resize_size": "auto",
        "normalization": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        "num_classes": 1000,
    },
    "wilds:camelyon17": {
        "image_size": 96,
        "train_image_size": None,
        "train_crop": "disabled",
        "padding": 4,
        "val_resize_size": None,
        "normalization": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        "num_classes": 2,
    },
    "wilds:camelyon17_strong": {
        "image_size": 96,
        "train_image_size": None,
        "train_crop": "random_rot90_hflip",
        "padding": 4,
        "val_resize_size": None,
        "normalization": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        "num_classes": 2,
    },
    "wilds:fmow": {
        "image_size": 224,
        "train_image_size": None,
        "train_crop": "disabled",
        "padding": 4,
        "val_resize_size": None,
        "normalization": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        "num_classes": 62,
    },
    "wilds:fmow_strong": {
        "image_size": 224,
        "train_image_size": None,
        "train_crop": "resize_random_hflip",
        "padding": 4,
        "val_resize_size": None,
        "normalization": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        "num_classes": 62,
    },
    "wilds:iwildcam": {
        "image_size": 448,
        "train_image_size": None,
        "train_crop": "disabled",
        "padding": 4,
        "val_resize_size": None,
        "normalization": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        "num_classes": 182,
    },
    "wilds:iwildcam_strong": {
        "image_size": 448,
        "train_image_size": None,
        "train_crop": "resize_random_hflip",
        "padding": 4,
        "val_resize_size": None,
        "normalization": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        "num_classes": 182,
    },
    "wilds:rxrx1": {
        "image_size": 256,
        "train_image_size": None,
        "train_crop": "random_rot90_hflip",
        "padding": 4,
        "val_resize_size": None,
        "normalization": "per_image",
        "num_classes": 1139,
    },
    "wilds:rxrx1_strong": {
        "image_size": 256,
        "train_image_size": None,
        "train_crop": "random_rot90_hflip",
        "padding": 4,
        "val_resize_size": None,
        "normalization": "per_image",
        "num_classes": 1139,
    },
}


_TIMM_MIXING = {
    "laug_kwargs.random_erasing_prob": 0.25,
    "laug_kwargs.mixup_alpha": 0.8,
    "laug_kwargs.cutmix_alpha": 1.0,
}

# Published recipe values beyond EXPECTED_CONTRACTS, keyed by dotted preset path.
RECIPE_VALUES = {
    # RandomCrop(32, padding=4, zeros), TrivialAugmentWide, timm mixing defaults.
    "cifar10": {
        "aug_kwargs.pad_mode": "CONSTANT",
        "aug_kwargs.augment_type": "trivial_augment_wide",
        "laug_kwargs.switch_prob": 0.5,
        "model_input.output_key": "image",
        "model_input.dtype": "float32",
        "model_input.normalization.kind": "mean_std",
        **_TIMM_MIXING,
    },
    "cifar100": {"aug_kwargs.pad_mode": "CONSTANT"},
    # ViT/ConvNeXt: bicubic RandomResizedCrop and RandAugment(n=2, m=9).
    "imagenet": {
        "aug_kwargs.augment_type": "rand_augment",
        "aug_kwargs.interpolation": "bicubic",
        "aug_kwargs.ra_kwargs.num_layers": 2,
        "aug_kwargs.ra_kwargs.magnitude": 9.0,
        **_TIMM_MIXING,
    },
    # ResNet legacy: ColorJitter(0.4, 0.4, 0.4, 0.1).
    "imagenet_resnet": {
        "aug_kwargs.augment_type": "color_jitter",
        "aug_kwargs.cj_kwargs.brightness": 0.4,
        "aug_kwargs.cj_kwargs.contrast": 0.4,
        "aug_kwargs.cj_kwargs.saturation": 0.4,
        "aug_kwargs.cj_kwargs.hue": 0.1,
        "laug_kwargs.mixup_alpha": 0.2,
    },
    # ResNet Strikes Back A1/A2/A3; A3 trains at 160 px (FixRes).
    "imagenet_a1": {
        "aug_kwargs.ra_kwargs.num_layers": 2,
        "aug_kwargs.ra_kwargs.magnitude": 7.0,
        "laug_kwargs.random_erasing_prob": 0.35,
        "laug_kwargs.mixup_alpha": 0.2,
        "laug_kwargs.cutmix_alpha": 1.0,
    },
    "imagenet_a2": {
        "aug_kwargs.ra_kwargs.magnitude": 6.0,
        "laug_kwargs.random_erasing_prob": 0.25,
        "laug_kwargs.mixup_alpha": 0.2,
    },
    "imagenet_a3": {
        "aug_kwargs.image_size": 160,
        "aug_kwargs.ra_kwargs.magnitude": 6.0,
        "laug_kwargs.random_erasing_prob": 0.0,
        "laug_kwargs.mixup_alpha": 0.1,
    },
    # DINOv2 asymmetric multi-crop.
    "dinov2": {
        "aug_kwargs.gc_kwargs.p_gaussian_blur": (1.0, 0.1),
        "aug_kwargs.gc_kwargs.p_solarize": (0.0, 0.2),
        "aug_kwargs.gc_kwargs.p_color_jitter": 0.8,
        "aug_kwargs.gc_kwargs.brightness": 0.4,
        "aug_kwargs.gc_kwargs.contrast": 0.4,
        "aug_kwargs.gc_kwargs.saturation": 0.2,
        "aug_kwargs.gc_kwargs.hue": 0.1,
        "aug_kwargs.gc_kwargs.p_grayscale": 0.2,
        "aug_kwargs.lc_kwargs.size": 96,
        "aug_kwargs.lc_kwargs.scale": (0.05, 0.32),
        "aug_kwargs.lc_kwargs.p_gaussian_blur": 0.5,
        "aug_kwargs.lc_kwargs.p_solarize": 0.0,
    },
    # WILDS reference recipes; strong variants are explicit opt-ins.
    "wilds:camelyon17": {"laug_kwargs.enable": False},
    "wilds:fmow": {"laug_kwargs.enable": False},
    "wilds:iwildcam": {"laug_kwargs.enable": False},
    "wilds:rxrx1": {"aug_kwargs.enable": True, "laug_kwargs.enable": False},
    "wilds:camelyon17_strong": {"aug_kwargs.augment_type": "color_jitter"},
    "wilds:fmow_strong": {"aug_kwargs.ra_kwargs.magnitude": 9.0},
    "wilds:iwildcam_strong": {
        "aug_kwargs.image_size": 448,
        "aug_kwargs.ra_kwargs.translate_const": 203.0,
    },
    "wilds:rxrx1_strong": {
        "laug_kwargs.random_erasing_prob": 0.25,
        "laug_kwargs.mixup_alpha": 0.0,
        "laug_kwargs.cutmix_alpha": 0.0,
    },
}


def _preset_value(preset, path):
    for key in path.split("."):
        preset = preset[key]
    return preset


@pytest.mark.parametrize("name", sorted(RECIPE_VALUES))
def test_vision_presets_match_recipe_values(name):
    preset = get_dataset_presets(name)
    for path, expected in RECIPE_VALUES[name].items():
        actual = _preset_value(preset, path)
        if isinstance(expected, bool):
            assert actual is expected, path
        else:
            assert actual == expected, path


@pytest.mark.parametrize("name", sorted(EXPECTED_HASHES))
def test_vision_preset_contract_fields_and_hash(name):
    preset = get_dataset_presets(name)
    expected = EXPECTED_CONTRACTS[name]
    postproc = preset["postproc_kwargs"]
    aug = preset["aug_kwargs"]
    model_input = preset["model_input"]

    assert get_resolved_preset(name).hash() == EXPECTED_HASHES[name]
    assert postproc["image_size"] == expected["image_size"]
    assert postproc.get("train_image_size") == expected["train_image_size"]
    assert postproc.get("val_resize_size", "auto") == expected["val_resize_size"]
    if expected["normalization"] == "per_image":
        assert postproc["normalization_mode"] == "per_image"
        assert model_input["normalization"]["kind"] == "per_image"
        assert model_input["normalization"]["per_channel"] is True
    else:
        assert postproc["normalization_params"] == expected["normalization"]
        assert model_input["normalization"]["mean"] == expected["normalization"][0]
        assert model_input["normalization"]["std"] == expected["normalization"][1]
    assert model_input["layout"] == "bchw"
    assert model_input["static_shape"] == (
        3,
        expected["image_size"],
        expected["image_size"],
    )

    if expected["train_crop"] == "disabled":
        assert aug["enable"] is False
    elif expected["train_crop"] == "ssl_multi_crop":
        assert aug["mode"] == "ssl"
        assert aug["n_global_crops"] == 2
        assert aug["n_local_crops"] == 8
    else:
        assert aug.get("crop_type", "random_resized") == expected["train_crop"]
        assert aug.get("padding", 4) == expected["padding"]


@pytest.mark.parametrize("name", sorted(EXPECTED_HASHES))
def test_vision_validation_postprocessing_is_deterministic(name):
    num_classes = EXPECTED_CONTRACTS[name]["num_classes"]
    pipeline = get_pipeline(
        pipeline_name="vision/classification",
        preset=name,
        modality="vision",
        apply_presets=True,
    )
    preproc, _aug, _late_aug, postproc = pipeline.build(is_training=False)
    image = tf.reshape(
        tf.cast(tf.range(300 * 320 * 3) % 256, tf.uint8),
        [300, 320, 3],
    )
    sample = {"image": image, "label": tf.constant(3, dtype=tf.int64)}

    first = postproc(preproc(sample), num_classes=num_classes)
    second = postproc(preproc(sample), num_classes=num_classes)

    np.testing.assert_allclose(first["image"].numpy(), second["image"].numpy())

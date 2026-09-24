"""Preset pipelines from the augmentation recipe document, run end to end.

Preset values themselves are pinned in test_vision_preset_contracts.py.
"""

import numpy as np
import pytest
import tensorflow as tf

from justdata.core.registry import get_pipeline
from justdata.vision.augmentations import geometric
from justdata.vision.augmentations.auto import RAND_AUGMENT_OPS, rand_augment
from justdata.vision.augmentations.color import color_jitter
from justdata.vision.augmentations.composed import (
    create_global_crops,
    create_local_crops,
)
from justdata.vision.augmentations.registry import (
    get_augment_strategy,
    get_crop_strategy,
    get_crop_strategy_metadata,
    register_augment_strategy,
)
from justdata.vision.presets import get_dataset_presets
from justdata.vision.tasks.classification import (
    make_augmentations,
    make_late_augmentations,
    make_postprocessing,
    make_preprocessing,
)


@pytest.mark.parametrize(
    "strategy", ["rand_augment", "trivial_augment", "trivial_augment_wide"]
)
def test_excluding_operations_outside_fixed_pool_has_no_effect(
    strategy, cifar_image, seed
):
    augment = get_augment_strategy(strategy)
    expected = augment(cifar_image, seed)
    actual = augment(
        cifar_image,
        seed,
        exclude_ops=["Invert", "Cutout", "SolarizeAdd", "Grayscale"],
    )
    np.testing.assert_array_equal(actual, expected)


def test_randaugment_cutout_constant_remains_a_compatibility_noop(cifar_image, seed):
    expected = rand_augment(cifar_image, seed, cutout_const=0.0)
    actual = rand_augment(cifar_image, seed, cutout_const=1000.0)
    np.testing.assert_array_equal(actual, expected)


@pytest.fixture
def imagenet_image():
    """Simulated 480x640x3 uint8 image (larger than 224, like real ImageNet)."""
    np.random.seed(0)
    return tf.constant(np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8))


@pytest.fixture
def cifar_image():
    """32x32x3 uint8 image."""
    np.random.seed(0)
    return tf.constant(np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8))


@pytest.fixture
def cifar_sample(cifar_image):
    return {"image": cifar_image, "label": tf.constant(3, dtype=tf.int64)}


@pytest.fixture
def imagenet_sample(imagenet_image):
    return {"image": imagenet_image, "label": tf.constant(5, dtype=tf.int64)}


# Supervised Training


class TestAutoAugmentStrategies:
    """RandAugment, TrivialAugment, and TrivialAugmentWide contract tests."""

    @pytest.mark.parametrize(
        ("strategy", "image_fixture", "kwargs"),
        [
            ("rand_augment", "cifar_image", {}),
            ("rand_augment", "cifar_image", {"num_layers": 2, "magnitude": 9.0}),
            ("rand_augment", "cifar_image", {"num_layers": 3, "magnitude": 12.0}),
            ("trivial_augment", "cifar_image", {}),
            ("trivial_augment", "cifar_image", {"translate_const": 14.0}),
            # TA-Wide uses a fixed 32 px translate, not an image-proportional one.
            ("trivial_augment_wide", "cifar_image", {}),
            ("trivial_augment_wide", "imagenet_image", {}),
        ],
        ids=[
            "ra-default",
            "ra-n2-m9",
            "ra-n3-m12",
            "ta-default",
            "ta-cifar-translate",
            "taw-cifar",
            "taw-imagenet",
        ],
    )
    def test_policy_preserves_shape_and_uint8(
        self, request, seed, strategy, image_fixture, kwargs
    ):
        image = request.getfixturevalue(image_fixture)
        result = get_augment_strategy(strategy)(image, seed=seed, **kwargs)
        assert result.shape == image.shape
        assert result.dtype == tf.uint8

    @pytest.mark.parametrize(
        "operation",
        ["Rotate", "TranslateX", "TranslateY", "ShearX", "ShearY"],
    )
    def test_randaugment_geometric_ops_use_constant_fill(self, operation):
        image = tf.fill((15, 15, 3), tf.constant(7, dtype=tf.uint8))
        exclude_ops = [op for op in RAND_AUGMENT_OPS if op != operation]

        result = rand_augment(
            image,
            seed=tf.constant([3, 5]),
            num_layers=1,
            magnitude=30.0,
            translate_const=4,
            exclude_ops=exclude_ops,
        )

        assert result.shape == image.shape
        assert result.dtype == image.dtype
        np.testing.assert_array_equal(np.unique(result.numpy()), [7, 128])

    def test_ra_14_op_pool(self):
        """Default RA pool is the strict 14-op RA space (no Invert/Cutout/SolarizeAdd)."""
        ra_14_ops = {
            "Identity",
            "AutoContrast",
            "Equalize",
            "Rotate",
            "TranslateX",
            "TranslateY",
            "ShearX",
            "ShearY",
            "Brightness",
            "Color",
            "Contrast",
            "Sharpness",
            "Posterize",
            "Solarize",
        }
        assert set(RAND_AUGMENT_OPS) == ra_14_ops
        assert {"Invert", "Cutout", "SolarizeAdd", "Grayscale"}.isdisjoint(ra_14_ops)


class TestCIFARTrainingPipeline:
    """CIFAR-10/100: RandomCrop(32, padding=4, zeros) -> HFlip -> TrivialAugment."""

    def test_cifar_full_sl_pipeline(self, cifar_sample, seed):
        presets = get_dataset_presets("cifar10")
        aug = make_augmentations(**presets["aug_kwargs"])
        result = aug(cifar_sample, seed=seed)
        assert result["image"].shape == (32, 32, 3)
        assert "label" in result

    def test_cifar_late_augmentations(self, seed):
        presets = get_dataset_presets("cifar10")
        laug = make_late_augmentations(**presets["laug_kwargs"])

        # Late aug runs after postprocessing, so images are float32
        np.random.seed(0)
        images = tf.constant(np.random.rand(4, 32, 32, 3).astype(np.float32))
        labels = tf.constant([0, 1, 2, 3], dtype=tf.int64)
        batch = {"image": images, "label": labels}

        result = laug(batch, num_classes=10, seed=seed)
        assert result["image"].dtype == tf.float32
        # mixup/cutmix produces soft labels
        assert result["label"].shape == (4, 10)


class TestImageNetModernTrainingPipeline:
    """ViT/ConvNeXt: RandomResizedCrop(224, bicubic) -> HFlip -> RandAugment(n=2, m=9)."""

    def test_imagenet_full_sl_pipeline(self, imagenet_sample, seed):
        presets = get_dataset_presets("imagenet")
        preproc = make_preprocessing()
        aug = make_augmentations(**presets["aug_kwargs"])

        sample = preproc(imagenet_sample)
        result = aug(sample, seed=seed)
        assert result["image"].shape == (224, 224, 3)


class TestRandomResizedHVFlip:
    def test_strategy_metadata_declares_image_only_crop(self):
        metadata = get_crop_strategy_metadata("random_resized_hvflip")
        assert metadata.name == "random_resized_hvflip"
        assert metadata.domain == "image"
        assert metadata.is_training_only is True
        assert metadata.requires_labels is False

    def test_crop_specific_scale_and_ratio_are_forwarded(
        self,
        monkeypatch,
        cifar_image,
        seed,
    ):
        captured = {}

        def capture_crop(
            image,
            size,
            seed,
            scale,
            ratio,
            interpolation,
        ):
            captured.update(
                {
                    "size": size,
                    "seed": seed,
                    "scale": scale,
                    "ratio": ratio,
                    "interpolation": interpolation,
                }
            )
            return tf.image.resize(image, [size, size])

        monkeypatch.setattr(geometric, "random_resized_crop", capture_crop)
        crop = get_crop_strategy("random_resized_hvflip")
        result = crop(
            cifar_image,
            size=16,
            seed=seed,
            scale=(0.85, 1.0),
            ratio=(0.9, 1.1),
            interpolation="bicubic",
            horizontal_flip_probability=0.0,
            vertical_flip_probability=0.0,
        )

        assert result.shape == (16, 16, 3)
        assert captured["size"] == 16
        assert captured["scale"] == (0.85, 1.0)
        assert captured["ratio"] == (0.9, 1.1)
        assert captured["interpolation"] == "bicubic"
        assert captured["seed"].shape == (2,)

    @pytest.mark.parametrize(
        (
            "horizontal_probability",
            "vertical_probability",
            "expected_transform",
        ),
        [
            (0.0, 0.0, lambda image: image),
            (1.0, 0.0, tf.image.flip_left_right),
            (0.0, 1.0, tf.image.flip_up_down),
            (
                1.0,
                1.0,
                lambda image: tf.image.flip_up_down(tf.image.flip_left_right(image)),
            ),
        ],
    )
    def test_flip_probabilities_are_independent_and_forceable(
        self,
        monkeypatch,
        seed,
        horizontal_probability,
        vertical_probability,
        expected_transform,
    ):
        image = tf.reshape(tf.range(2 * 3 * 3), [2, 3, 3])
        monkeypatch.setattr(
            geometric,
            "random_resized_crop",
            lambda image, **_kwargs: image,
        )
        crop = get_crop_strategy("random_resized_hvflip")
        result = crop(
            image,
            size=(2, 3),
            seed=seed,
            horizontal_flip_probability=horizontal_probability,
            vertical_flip_probability=vertical_probability,
        )
        np.testing.assert_array_equal(
            result.numpy(),
            expected_transform(image).numpy(),
        )

    def test_fmow_m1_mapping_builds_and_is_seed_deterministic(
        self,
        imagenet_sample,
        seed,
    ):
        augmentation = make_augmentations(
            image_size=224,
            crop_type="random_resized_hvflip",
            interpolation="bicubic",
            crop_kwargs={
                "scale": (0.85, 1.0),
                "ratio": (0.90, 1.10),
                "horizontal_flip_probability": 0.5,
                "vertical_flip_probability": 0.5,
            },
            augment_type="none",
        )
        sample = {
            **imagenet_sample,
            "metadata": {"wilds_index": tf.constant(7, dtype=tf.int64)},
        }
        first = augmentation(sample, seed=seed)
        second = augmentation(sample, seed=seed)

        assert first["image"].shape == (224, 224, 3)
        np.testing.assert_array_equal(
            first["image"].numpy(),
            second["image"].numpy(),
        )
        assert first["label"].numpy() == imagenet_sample["label"].numpy()
        assert first["metadata"]["wilds_index"].numpy() == 7

    def test_augment_strategy_receives_seed_by_keyword(self, imagenet_sample, seed):
        name = "test_keyword_only_seed_augment"

        @register_augment_strategy(name)
        def keyword_only_seed(image, *, seed, **kwargs):
            return image

        augmentation = make_augmentations(image_size=224, augment_type=name)

        assert augmentation(imagenet_sample, seed=seed)["image"].shape == (224, 224, 3)

    def test_crop_kwargs_reject_legacy_top_level_key_conflicts(self):
        with pytest.raises(ValueError, match="interpolation"):
            make_augmentations(
                image_size=224,
                crop_type="random_resized_hvflip",
                interpolation="bicubic",
                crop_kwargs={"interpolation": "nearest"},
            )

    def test_existing_presets_do_not_gain_crop_kwargs(self):
        for name in (
            "cifar10",
            "imagenet",
            "imagenet_resnet",
            "wilds:fmow",
            "wilds:fmow_strong",
        ):
            assert "crop_kwargs" not in get_dataset_presets(name)["aug_kwargs"]


class TestWILDSCropStrategies:
    def test_wilds_crop_strategies_preserve_expected_shapes(self, imagenet_image, seed):
        hflip = get_crop_strategy("random_hflip")
        first = hflip(imagenet_image, size=224, seed=seed)
        second = hflip(imagenet_image, size=224, seed=seed)
        assert first.shape == imagenet_image.shape
        np.testing.assert_array_equal(first.numpy(), second.numpy())

        resize_hflip = get_crop_strategy("resize_random_hflip")
        resized = resize_hflip(imagenet_image, size=224, seed=seed)
        assert resized.shape == (224, 224, 3)

        rot_hflip = get_crop_strategy("random_rot90_hflip")
        square = imagenet_image[:224, :224]
        rotated = rot_hflip(square, size=224, seed=seed)
        assert rotated.shape == (224, 224, 3)


class TestImageNetLegacyTrainingPipeline:
    """ResNet legacy: RandomResizedCrop(224) -> HFlip -> ColorJitter(0.4,0.4,0.4,0.1)."""

    def test_resnet_full_sl_pipeline(self, imagenet_sample, seed):
        presets = get_dataset_presets("imagenet_resnet")
        preproc = make_preprocessing()
        aug = make_augmentations(**presets["aug_kwargs"])

        sample = preproc(imagenet_sample)
        result = aug(sample, seed=seed)
        assert result["image"].shape == (224, 224, 3)


class TestColorJitterIndependence:
    """color_jitter applies jitter with probability p, then grayscale independently."""

    def test_p_zero_skips_jitter(self, cifar_image, seed):
        result = color_jitter(
            cifar_image,
            seed=seed,
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1,
            p=0.0,
            p_grayscale=0.0,
        )
        # With p=0 and p_grayscale=0, image should be unchanged
        np.testing.assert_array_equal(result.numpy(), cifar_image.numpy())

    def test_zero_magnitudes_are_noops_for_float_images(self, seed):
        image = tf.reshape(tf.linspace(0.0, 255.0, 4 * 4 * 3), [4, 4, 3])

        result = color_jitter(
            image,
            seed=seed,
            brightness=0.0,
            contrast=0.0,
            saturation=0.0,
            hue=0.0,
            p=1.0,
            p_grayscale=0.0,
        )

        assert result.dtype == image.dtype
        np.testing.assert_allclose(result.numpy(), image.numpy())

    def test_brightness_matches_uint8_and_float_255_inputs(self):
        seed = tf.constant([11, 17], dtype=tf.int32)
        image_uint8 = tf.ones((8, 8, 3), dtype=tf.uint8) * 128
        image_float = tf.cast(image_uint8, tf.float32)

        kwargs = {
            "brightness": 0.4,
            "contrast": 0.0,
            "saturation": 0.0,
            "hue": 0.0,
            "p": 1.0,
            "p_grayscale": 0.0,
        }
        out_uint8 = color_jitter(image_uint8, seed=seed, **kwargs)
        out_float = color_jitter(image_float, seed=seed, **kwargs)

        assert out_uint8.dtype == tf.uint8
        assert out_float.dtype == tf.float32
        np.testing.assert_allclose(
            out_float.numpy(), tf.cast(out_uint8, tf.float32).numpy()
        )

    def test_brightness_on_float_255_is_not_pixel_delta_noop(self):
        seed = tf.constant([11, 17], dtype=tf.int32)
        image = tf.ones((8, 8, 3), dtype=tf.float32) * 128.0

        result = color_jitter(
            image,
            seed=seed,
            brightness=0.4,
            contrast=0.0,
            saturation=0.0,
            hue=0.0,
            p=1.0,
            p_grayscale=0.0,
        )

        mean_delta = tf.reduce_mean(tf.abs(result - image))
        assert float(mean_delta) > 1.0

    def test_grayscale_independent_of_jitter(self, cifar_image, seed):
        """p_grayscale should work even when p=0 (no jitter)."""
        result = color_jitter(
            cifar_image,
            seed=seed,
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1,
            p=0.0,
            p_grayscale=1.0,
        )
        # With p=0 (no jitter) and p_grayscale=1.0, all channels should be equal
        r, g, b = result[:, :, 0], result[:, :, 1], result[:, :, 2]
        np.testing.assert_array_equal(r.numpy(), g.numpy())
        np.testing.assert_array_equal(g.numpy(), b.numpy())


# Validation Pipelines


class TestCIFARValidation:
    """CIFAR validation: no resize, no crop, just normalize."""

    def test_cifar_val_pipeline(self, cifar_sample):
        presets = get_dataset_presets("cifar10")
        postproc = make_postprocessing(**presets["postproc_kwargs"], is_training=False)
        result = postproc(cifar_sample)
        # Should stay 32x32 (CHW after permute)
        assert result["image"].shape[1] == 32
        assert result["image"].shape[2] == 32


class TestImageNetValidation:
    """ImageNet validation: Resize(256) -> CenterCrop(224) -> Normalize."""

    def test_default_val_resize_auto_resolves_to_256(self):
        """The default val_resize_size='auto' resolves to int(224 / 0.875)."""
        resolved = get_pipeline(dataset="imagenet").resolve_config(False)
        geometry = resolved["stages"]["postprocess"]["geometry"]
        assert geometry["validation_resize_size"] == 256

    def test_imagenet_val_pipeline_shape(self, imagenet_sample):
        preproc = make_preprocessing()
        postproc = make_postprocessing(
            image_size=224,
            is_training=False,
            normalization_params=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        )
        sample = preproc(imagenet_sample)
        result = postproc(sample)
        # CHW output: (3, 224, 224)
        assert result["image"].shape == (3, 224, 224)
        assert result["image"].dtype == tf.float32


class TestFixResValidation:
    """RSB A3 FixRes: train 160, validate 224 with resize to 236."""

    def test_a3_val_pipeline_shape(self, imagenet_sample):
        presets = get_dataset_presets("imagenet_a3")
        preproc = make_preprocessing()
        postproc = make_postprocessing(**presets["postproc_kwargs"], is_training=False)
        sample = preproc(imagenet_sample)
        result = postproc(sample)
        # CHW output: (3, 224, 224)
        assert result["image"].shape == (3, 224, 224)


# RSB Recipes


class TestRSBRecipes:
    """RSB A1 and A3 presets run through the complete training stages."""

    def test_a1_full_pipeline_runs(self, imagenet_sample, seed):
        presets = get_dataset_presets("imagenet_a1")
        preproc = make_preprocessing()
        aug = make_augmentations(**presets["aug_kwargs"])
        laug = make_late_augmentations(**presets["laug_kwargs"])

        sample = preproc(imagenet_sample)
        sample = aug(sample, seed=seed)
        postproc = make_postprocessing(**presets["postproc_kwargs"], is_training=True)
        sample = postproc(sample)

        # Batch for late aug
        batch = {k: tf.expand_dims(v, 0) for k, v in sample.items()}
        result = laug(batch, num_classes=1000, seed=seed)
        assert result["image"].dtype == tf.float32

    def test_a3_train_resolution(self, imagenet_sample, seed):
        presets = get_dataset_presets("imagenet_a3")
        preproc = make_preprocessing()
        aug = make_augmentations(**presets["aug_kwargs"])

        sample = preproc(imagenet_sample)
        result = aug(sample, seed=seed)
        # Train at 160x160
        assert result["image"].shape == (160, 160, 3)


# DINOv2 SSL


class TestDINOv2Pipeline:
    """DINOv2 self-supervised learning with asymmetric multi-crop."""

    def test_asymmetric_global_crops_run(self, imagenet_image, seed):
        """create_global_crops with per-crop blur/solarize probabilities."""
        crops = create_global_crops(
            imagenet_image,
            crops_number=2,
            size=224,
            scale=(0.32, 1.0),
            seed=seed,
            p_gaussian_blur=(1.0, 0.1),
            p_solarize=(0.0, 0.2),
        )
        assert crops.shape == (2, 224, 224, 3)

    def test_uniform_global_crops_still_work(self, imagenet_image, seed):
        """Scalar blur/solarize probabilities still work (backwards compat)."""
        crops = create_global_crops(
            imagenet_image,
            crops_number=2,
            size=224,
            scale=(0.32, 1.0),
            seed=seed,
            p_gaussian_blur=0.5,
            p_solarize=0.1,
        )
        assert crops.shape == (2, 224, 224, 3)

    def test_local_crops_with_solarize(self, imagenet_image, seed):
        """Local crops accept p_solarize parameter."""
        crops = create_local_crops(
            imagenet_image,
            crops_number=4,
            size=96,
            scale=(0.05, 0.32),
            seed=seed,
            p_gaussian_blur=0.5,
            p_solarize=0.0,
        )
        assert crops.shape == (4, 96, 96, 3)

    def test_dinov2_full_ssl_pipeline(self, imagenet_sample, seed):
        presets = get_dataset_presets("dinov2")
        preproc = make_preprocessing()
        aug = make_augmentations(**presets["aug_kwargs"])
        postproc = make_postprocessing(**presets["postproc_kwargs"], is_training=True)

        sample = preproc(imagenet_sample)
        sample = aug(sample, seed=seed)

        assert "global_crops" in sample
        assert "local_crops" in sample
        assert sample["global_crops"].shape == (2, 224, 224, 3)
        assert sample["local_crops"].shape == (8, 96, 96, 3)

        sample = postproc(sample)
        # After normalization + permute: (2, 3, 224, 224)
        assert sample["global_crops"].shape[1] == 3


# Full Pipeline via Registry


class TestRegistryIntegration:
    """Test pipelines via the registry system."""

    def test_classification_pipeline_for_cifar(self):
        preproc, aug, laug, postproc = get_pipeline(dataset="cifar10").build(
            is_training=True
        )

        np.random.seed(0)
        sample = {
            "image": tf.constant(
                np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
            ),
            "label": tf.constant(1, dtype=tf.int64),
        }
        seed = tf.constant([1, 2], dtype=tf.int32)

        sample = preproc(sample)
        sample = aug(sample, seed=seed)
        sample = postproc(sample, num_classes=10)

        # CIFAR training: 32x32 -> CHW (3, 32, 32)
        assert sample["image"].shape == (3, 32, 32)

    def test_classification_pipeline_for_imagenet_default(self):
        """Default ImageNet pipeline (modern/ViT)."""
        preproc, aug, laug, postproc = get_pipeline(dataset="imagenette").build(
            is_training=True
        )

        np.random.seed(0)
        sample = {
            "image": tf.constant(
                np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            ),
            "label": tf.constant(0, dtype=tf.int64),
        }
        seed = tf.constant([1, 2], dtype=tf.int32)

        sample = preproc(sample)
        sample = aug(sample, seed=seed)
        assert sample["image"].shape == (224, 224, 3)

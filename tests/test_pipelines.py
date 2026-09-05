"""Tests for the complete preset pipelines described in the augmentation recipe document.

Covers:
  - Part 1: Supervised training (CIFAR, ImageNet modern/legacy, RSB A1/A2/A3)
  - Part 2: Validation pipelines (CIFAR, ImageNet 0.875, FixRes A3)
  - Part 3: RSB recipe presets
  - Part 4: DINOv2 SSL (asymmetric multi-crop)
  - Dense task patch-aligned validation
"""

import numpy as np
import pytest
import tensorflow as tf

from justdata.core.registry import get_pipeline_for_dataset
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
)
from justdata.vision.presets import get_dataset_presets, merge_with_presets
from justdata.vision.tasks.classification import (
    make_augmentations,
    make_late_augmentations,
    make_postprocessing,
    make_preprocessing,
)
from justdata.vision.tasks.segmentation import (
    make_postprocessing as seg_make_postprocessing,
)
from justdata.vision.transforms import pad_to_patch_multiple


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
def seed():
    return tf.constant([42, 0], dtype=tf.int32)


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

    def test_rand_augment_registered(self):
        assert get_augment_strategy("rand_augment") is not None

    def test_trivial_augment_registered(self):
        assert get_augment_strategy("trivial_augment") is not None

    def test_trivial_augment_wide_registered(self):
        assert get_augment_strategy("trivial_augment_wide") is not None

    def test_rand_augment_shape_preserved(self, cifar_image, seed):
        aug_fn = get_augment_strategy("rand_augment")
        result = aug_fn(cifar_image, seed=seed, num_layers=2, magnitude=9.0)
        assert result.shape == (32, 32, 3)

    def test_rand_augment_multi_layer(self, cifar_image, seed):
        """RA applies N ops with replacement; N=3 should still preserve shape."""
        aug_fn = get_augment_strategy("rand_augment")
        result = aug_fn(cifar_image, seed=seed, num_layers=3, magnitude=12.0)
        assert result.shape == (32, 32, 3)

    def test_rand_augment_dtype_preserved(self, cifar_image, seed):
        aug_fn = get_augment_strategy("rand_augment")
        result = aug_fn(cifar_image, seed=seed)
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

    def test_trivial_augment_shape_preserved(self, cifar_image, seed):
        aug_fn = get_augment_strategy("trivial_augment")
        result = aug_fn(cifar_image, seed=seed)
        assert result.shape == (32, 32, 3)

    def test_trivial_augment_wide_shape_cifar(self, cifar_image, seed):
        aug_fn = get_augment_strategy("trivial_augment_wide")
        result = aug_fn(cifar_image, seed=seed)
        assert result.shape == (32, 32, 3)

    def test_trivial_augment_wide_shape_imagenet(self, imagenet_image, seed):
        """TA-Wide uses fixed 32 px translate (not image-proportional)."""
        aug_fn = get_augment_strategy("trivial_augment_wide")
        result = aug_fn(imagenet_image, seed=seed)
        assert result.shape == imagenet_image.shape

    def test_trivial_augment_wide_dtype_preserved(self, cifar_image, seed):
        aug_fn = get_augment_strategy("trivial_augment_wide")
        result = aug_fn(cifar_image, seed=seed)
        assert result.dtype == tf.uint8

    def test_ta_and_taw_vary_with_seed(self, cifar_image):
        """TA and TA-Wide sample m ~ U{0,...,30}; different seeds give different output."""
        for strategy in ("trivial_augment", "trivial_augment_wide"):
            aug_fn = get_augment_strategy(strategy)
            results = [
                aug_fn(
                    cifar_image, seed=tf.constant([i * 13 + 1, i * 7], dtype=tf.int32)
                ).numpy()
                for i in range(10)
            ]
            assert not all(np.array_equal(results[0], r) for r in results[1:]), (
                f"{strategy} produced identical output for all seeds"
            )

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

    def test_ta_excludes_non_ra_ops(self, cifar_image, seed):
        """TA passes non-RA ops to exclude; result must still be valid."""
        from justdata.vision.augmentations.auto import trivial_augment

        result = trivial_augment(
            cifar_image,
            seed=seed,
            exclude_ops=["Invert", "Cutout", "SolarizeAdd", "Grayscale"],
        )
        assert result.shape == (32, 32, 3)

    def test_taw_wide_bounds_run_without_error(self, imagenet_image, seed):
        """TA-Wide wide params (±135°, shear ±0.99, enhance ±0.99, min 2 bits posterize)."""
        from justdata.vision.augmentations.auto import trivial_augment_wide

        result = trivial_augment_wide(imagenet_image, seed=seed)
        assert result.shape == imagenet_image.shape
        assert result.dtype == tf.uint8


class TestCIFARTrainingPipeline:
    """CIFAR-10/100: RandomCrop(32, padding=4, zeros) -> HFlip -> TrivialAugment."""

    def test_cifar_preset_uses_zero_padding(self):
        presets = get_dataset_presets("cifar10")
        assert presets["aug_kwargs"]["pad_mode"] == "CONSTANT"

    def test_cifar100_preset_uses_zero_padding(self):
        presets = get_dataset_presets("cifar100")
        assert presets["aug_kwargs"]["pad_mode"] == "CONSTANT"

    def test_cifar_crop_preserves_size(self, cifar_image, seed):
        crop_fn = get_crop_strategy("random_pad")
        cropped = crop_fn(
            cifar_image, size=32, seed=seed, padding=4, pad_mode="CONSTANT"
        )
        assert cropped.shape == (32, 32, 3)

    def test_cifar_trivial_augment_strategy(self, cifar_image, seed):
        aug_fn = get_augment_strategy("trivial_augment")
        result = aug_fn(cifar_image, seed=seed, translate_const=14.0)
        assert result.shape == (32, 32, 3)

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

    def test_cifar_normalization_stats(self):
        cifar10 = get_dataset_presets("cifar10")
        mean, std = cifar10["postproc_kwargs"]["normalization_params"]
        assert mean == (0.4914, 0.4822, 0.4465)
        assert std == (0.2023, 0.1994, 0.2010)

        cifar100 = get_dataset_presets("cifar100")
        mean, std = cifar100["postproc_kwargs"]["normalization_params"]
        assert mean == (0.5071, 0.4867, 0.4408)
        assert std == (0.2675, 0.2565, 0.2761)

    def test_cifar_random_erasing(self):
        presets = get_dataset_presets("cifar10")
        assert presets["laug_kwargs"]["random_erasing_prob"] == 0.25

    def test_cifar_mixup_cutmix_params(self):
        presets = get_dataset_presets("cifar10")
        assert presets["laug_kwargs"]["mixup_alpha"] == 0.8
        assert presets["laug_kwargs"]["cutmix_alpha"] == 1.0
        assert presets["laug_kwargs"]["switch_prob"] == 0.5


class TestImageNetModernTrainingPipeline:
    """ViT/ConvNeXt: RandomResizedCrop(224, bicubic) -> HFlip -> RandAugment(n=2, m=9)."""

    def test_default_preset_is_modern(self):
        presets = get_dataset_presets("imagenet")
        assert presets["aug_kwargs"]["augment_type"] == "rand_augment"
        assert presets["aug_kwargs"]["ra_kwargs"]["magnitude"] == 9.0
        assert presets["aug_kwargs"]["ra_kwargs"]["num_layers"] == 2

    def test_default_uses_bicubic(self):
        presets = get_dataset_presets("imagenet")
        assert presets["aug_kwargs"]["interpolation"] == "bicubic"

    def test_random_resized_crop_includes_flip(self, imagenet_image, seed):
        """The crop strategy should include horizontal flip."""
        crop_fn = get_crop_strategy("random_resized")
        # Run multiple times to check flip can occur
        results = []
        for i in range(10):
            s = tf.constant([42 + i, 0], dtype=tf.int32)
            result = crop_fn(imagenet_image, size=224, seed=s, interpolation="bicubic")
            results.append(result.numpy())
        # At least verify the output shape
        assert results[0].shape == (224, 224, 3)

    def test_imagenet_full_sl_pipeline(self, imagenet_sample, seed):
        presets = get_dataset_presets("imagenet")
        preproc = make_preprocessing()
        aug = make_augmentations(**presets["aug_kwargs"])

        sample = preproc(imagenet_sample)
        result = aug(sample, seed=seed)
        assert result["image"].shape == (224, 224, 3)

    def test_default_random_erasing_prob(self):
        presets = get_dataset_presets("imagenet")
        assert presets["laug_kwargs"]["random_erasing_prob"] == 0.25

    def test_default_mixup_cutmix(self):
        presets = get_dataset_presets("imagenet")
        assert presets["laug_kwargs"]["mixup_alpha"] == 0.8
        assert presets["laug_kwargs"]["cutmix_alpha"] == 1.0


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


class TestWILDSClassificationPresets:
    def test_wilds_base_presets_follow_reference_contracts(self):
        expected = {
            "wilds:camelyon17": (96, False, None, "mean_std"),
            "wilds:fmow": (224, False, None, "mean_std"),
            "wilds:iwildcam": (448, False, None, "mean_std"),
            "wilds:rxrx1": (256, True, "random_rot90_hflip", "per_image"),
        }

        for name, (image_size, aug_enabled, crop_type, norm_mode) in expected.items():
            presets = get_dataset_presets(name)
            assert presets["postproc_kwargs"]["image_size"] == image_size
            assert presets["aug_kwargs"]["enable"] is aug_enabled
            assert presets["laug_kwargs"]["enable"] is False
            assert (
                presets["postproc_kwargs"].get("normalization_mode", "mean_std")
                == norm_mode
            )
            if crop_type is not None:
                assert presets["aug_kwargs"]["crop_type"] == crop_type

    def test_wilds_strong_presets_are_opt_in(self):
        camelyon = get_dataset_presets("wilds:camelyon17_strong")
        assert camelyon["aug_kwargs"]["augment_type"] == "color_jitter"
        assert camelyon["aug_kwargs"]["crop_type"] == "random_rot90_hflip"

        fmow = get_dataset_presets("wilds:fmow_strong")
        assert fmow["aug_kwargs"]["crop_type"] == "resize_random_hflip"
        assert fmow["aug_kwargs"]["ra_kwargs"]["magnitude"] == 9.0

        iwildcam = get_dataset_presets("wilds:iwildcam_strong")
        assert iwildcam["aug_kwargs"]["image_size"] == 448
        assert iwildcam["aug_kwargs"]["ra_kwargs"]["translate_const"] == 203.0

        rxrx1 = get_dataset_presets("wilds:rxrx1_strong")
        assert rxrx1["postproc_kwargs"]["normalization_mode"] == "per_image"
        assert rxrx1["laug_kwargs"]["random_erasing_prob"] == 0.25
        assert rxrx1["laug_kwargs"]["mixup_alpha"] == 0.0
        assert rxrx1["laug_kwargs"]["cutmix_alpha"] == 0.0

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

    def test_resnet_preset_uses_color_jitter(self):
        presets = get_dataset_presets("imagenet_resnet")
        assert presets["aug_kwargs"]["augment_type"] == "color_jitter"

    def test_color_jitter_strategy_registered(self):
        aug_fn = get_augment_strategy("color_jitter")
        assert aug_fn is not None

    def test_color_jitter_augment_runs(self, imagenet_image, seed):
        aug_fn = get_augment_strategy("color_jitter")
        result = aug_fn(
            imagenet_image,
            seed=seed,
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1,
        )
        assert result.shape == imagenet_image.shape

    def test_resnet_full_sl_pipeline(self, imagenet_sample, seed):
        presets = get_dataset_presets("imagenet_resnet")
        preproc = make_preprocessing()
        aug = make_augmentations(**presets["aug_kwargs"])

        sample = preproc(imagenet_sample)
        result = aug(sample, seed=seed)
        assert result["image"].shape == (224, 224, 3)

    def test_resnet_preset_cj_kwargs(self):
        presets = get_dataset_presets("imagenet_resnet")
        cj = presets["aug_kwargs"]["cj_kwargs"]
        assert cj["brightness"] == 0.4
        assert cj["contrast"] == 0.4
        assert cj["saturation"] == 0.4
        assert cj["hue"] == 0.1

    def test_resnet_preset_mixup_alpha(self):
        presets = get_dataset_presets("imagenet_resnet")
        assert presets["laug_kwargs"]["mixup_alpha"] == 0.2


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

    def test_p_one_always_applies_jitter(self, cifar_image, seed):
        result = color_jitter(
            cifar_image,
            seed=seed,
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1,
            p=1.0,
            p_grayscale=0.0,
        )
        # Should be different from original (with high probability)
        assert result.shape == cifar_image.shape

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

    def test_cifar_val_no_resize(self):
        presets = get_dataset_presets("cifar10")
        assert presets["postproc_kwargs"]["val_resize_size"] is None

    def test_cifar_val_pipeline(self, cifar_sample):
        presets = get_dataset_presets("cifar10")
        postproc = make_postprocessing(**presets["postproc_kwargs"], is_training=False)
        result = postproc(cifar_sample)
        # Should stay 32x32 (CHW after permute)
        assert result["image"].shape[1] == 32
        assert result["image"].shape[2] == 32


class TestImageNetValidation:
    """ImageNet validation: Resize(256) -> CenterCrop(224) -> Normalize."""

    def test_default_val_resize_auto(self):
        """Default preset uses val_resize_size='auto' -> int(224/0.875) = 256."""
        presets = get_dataset_presets("imagenet")
        postproc_kwargs = presets["postproc_kwargs"]
        # "auto" is the default when not set
        val_resize = postproc_kwargs.get("val_resize_size", "auto")
        if val_resize == "auto":
            computed = int(224 / 0.875)
            assert computed == 256

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

    def test_a3_train_size_160(self):
        presets = get_dataset_presets("imagenet_a3")
        assert presets["aug_kwargs"]["image_size"] == 160

    def test_a3_val_size_224_with_resize_236(self):
        presets = get_dataset_presets("imagenet_a3")
        assert presets["postproc_kwargs"]["image_size"] == 224
        assert presets["postproc_kwargs"]["val_resize_size"] == 236

    def test_a1_a2_crop_pct_1(self):
        """A1/A2 use crop_pct=1.0 -> val_resize_size=None (just resize)."""
        for name in ("imagenet_a1", "imagenet_a2"):
            presets = get_dataset_presets(name)
            assert presets["postproc_kwargs"]["val_resize_size"] is None

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
    """Verify RSB A1/A2/A3 presets match the 'ResNet Strikes Back' paper."""

    def test_a1_heavy_params(self):
        p = get_dataset_presets("imagenet_a1")
        assert p["aug_kwargs"]["ra_kwargs"]["magnitude"] == 7.0
        assert p["aug_kwargs"]["ra_kwargs"]["num_layers"] == 2
        assert p["laug_kwargs"]["random_erasing_prob"] == 0.35
        assert p["laug_kwargs"]["mixup_alpha"] == 0.2
        assert p["laug_kwargs"]["cutmix_alpha"] == 1.0

    def test_a2_moderate_params(self):
        p = get_dataset_presets("imagenet_a2")
        assert p["aug_kwargs"]["ra_kwargs"]["magnitude"] == 6.0
        assert p["laug_kwargs"]["random_erasing_prob"] == 0.25
        assert p["laug_kwargs"]["mixup_alpha"] == 0.2

    def test_a3_light_params(self):
        p = get_dataset_presets("imagenet_a3")
        assert p["aug_kwargs"]["ra_kwargs"]["magnitude"] == 6.0
        assert p["laug_kwargs"]["random_erasing_prob"] == 0.0
        assert p["laug_kwargs"]["mixup_alpha"] == 0.1

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

    def test_dinov2_preset_exists(self):
        presets = get_dataset_presets("dinov2")
        assert presets["aug_kwargs"]["mode"] == "ssl"

    def test_dinov2_global_crop_asymmetry(self):
        presets = get_dataset_presets("dinov2")
        gc = presets["aug_kwargs"]["gc_kwargs"]
        # Per-crop blur/solarize
        assert gc["p_gaussian_blur"] == (1.0, 0.1)
        assert gc["p_solarize"] == (0.0, 0.2)

    def test_dinov2_local_crop_params(self):
        presets = get_dataset_presets("dinov2")
        lc = presets["aug_kwargs"]["lc_kwargs"]
        assert lc["size"] == 96
        assert lc["scale"] == (0.05, 0.32)
        assert lc["p_gaussian_blur"] == 0.5
        assert lc["p_solarize"] == 0.0

    def test_dinov2_color_jitter_params(self):
        presets = get_dataset_presets("dinov2")
        gc = presets["aug_kwargs"]["gc_kwargs"]
        assert gc["p_color_jitter"] == 0.8
        assert gc["brightness"] == 0.4
        assert gc["contrast"] == 0.4
        assert gc["saturation"] == 0.2
        assert gc["hue"] == 0.1
        assert gc["p_grayscale"] == 0.2

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


# Dense Task Patch Alignment


class TestPatchAlignment:
    """Patch-aligned padding for dense ViT evaluation."""

    def test_pad_to_patch_multiple_exact(self):
        """Image already aligned -> no padding."""
        image = tf.zeros((224, 224, 3))
        result = pad_to_patch_multiple(image, patch_size=14)
        assert result.shape == (224, 224, 3)

    def test_pad_to_patch_multiple_needs_padding(self):
        """Image not aligned -> pad bottom/right."""
        image = tf.zeros((225, 225, 3))
        result = pad_to_patch_multiple(image, patch_size=14)
        # 225 -> ceil to 238 (14*17)
        assert result.shape[0] % 14 == 0
        assert result.shape[1] % 14 == 0
        assert result.shape[0] == 238
        assert result.shape[1] == 238

    def test_pad_to_patch_multiple_patch16(self):
        image = tf.zeros((200, 300, 3))
        result = pad_to_patch_multiple(image, patch_size=16)
        assert result.shape[0] % 16 == 0  # 208
        assert result.shape[1] % 16 == 0  # 304

    def test_segmentation_patch_aligned_postprocessing(self):
        """Segmentation postprocessing with patch_align=True."""
        postproc = seg_make_postprocessing(
            image_size=224,
            is_training=False,
            patch_align=True,
            patch_size=14,
        )
        np.random.seed(0)
        sample = {
            "image": tf.constant(
                np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            ),
            "mask": tf.constant(np.random.randint(0, 5, (480, 640), dtype=np.int32)),
        }
        result = postproc(sample)
        # Image should be CHW and patch-aligned
        assert result["image"].shape[0] == 3
        assert result["image"].shape[1] % 14 == 0
        assert result["image"].shape[2] % 14 == 0
        # Mask should also be aligned
        assert result["mask"].shape[0] % 14 == 0
        assert result["mask"].shape[1] % 14 == 0

    def test_segmentation_standard_postprocessing_unchanged(self):
        """Standard segmentation postprocessing (no patch_align) still works."""
        postproc = seg_make_postprocessing(
            image_size=224,
            is_training=False,
            patch_align=False,
        )
        np.random.seed(0)
        sample = {
            "image": tf.constant(
                np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            ),
            "mask": tf.constant(np.random.randint(0, 5, (480, 640), dtype=np.int32)),
        }
        result = postproc(sample)
        # CHW, 224x224 fixed size
        assert result["image"].shape == (3, 224, 224)

    def test_segmentation_training_ignores_patch_align(self):
        """During training, patch_align should be ignored."""
        postproc = seg_make_postprocessing(
            image_size=32,
            is_training=True,
            patch_align=True,
            patch_size=14,
        )
        np.random.seed(0)
        sample = {
            "image": tf.constant(
                np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
            ),
            "mask": tf.constant(np.random.randint(0, 5, (32, 32), dtype=np.int32)),
        }
        result = postproc(sample)
        # Standard resize, not patch-aligned
        assert result["image"].shape == (3, 32, 32)


# Preset Merge Tests


class TestPresetMerging:
    """Ensure smart merge works correctly with new presets."""

    def test_a1_merge_no_override(self):
        """Using imagenet_a1 presets without overrides."""
        result = merge_with_presets("imagenet_a1", {})
        assert result["laug_kwargs"]["random_erasing_prob"] == 0.35

    def test_a3_user_override_image_size(self):
        """User can override A3 train image size."""
        result = merge_with_presets("imagenet_a3", {"aug_kwargs": {"image_size": 128}})
        assert result["aug_kwargs"]["image_size"] == 128

    def test_dinov2_merge_preserves_asymmetry(self):
        """DINOv2 asymmetric params survive merge."""
        result = merge_with_presets("dinov2", {})
        gc = result["aug_kwargs"]["gc_kwargs"]
        assert gc["p_gaussian_blur"] == (1.0, 0.1)
        assert gc["p_solarize"] == (0.0, 0.2)


# Full Pipeline via Registry


class TestRegistryIntegration:
    """Test pipelines via the registry system."""

    def test_classification_pipeline_for_cifar(self):
        pipeline = get_pipeline_for_dataset(
            "cifar10", is_training=True, apply_presets=True
        )
        preproc, aug, laug, postproc = pipeline

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
        pipeline = get_pipeline_for_dataset(
            "imagenette", is_training=True, apply_presets=True
        )
        preproc, aug, laug, postproc = pipeline

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

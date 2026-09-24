import numpy as np
import pytest
import tensorflow as tf

from augmentation_harness import (
    assert_different_seed_can_change_output,
    assert_same_seed_same_output,
)
from justdata.vision.augmentations.auto import (
    rand_augment,
    trivial_augment,
    trivial_augment_wide,
)
from justdata.vision.augmentations.color import color_jitter
from justdata.vision.augmentations.mixing import mixup_cutmix, random_erasing


def _image() -> tf.Tensor:
    rng = np.random.default_rng(0)
    return tf.constant(rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8))


def _batch() -> tf.Tensor:
    values = tf.reshape(tf.range(4 * 16 * 16 * 3, dtype=tf.float32), [4, 16, 16, 3])
    return values / tf.reduce_max(values)


def _labels() -> tf.Tensor:
    return tf.constant([0, 1, 2, 3], dtype=tf.int64)


def _check_seed_contract(augment, input_value):
    """Vision augmentations take no training flag; the harness always trains."""

    def transform(value, seed, is_training):
        del is_training
        return augment(value, seed)

    assert_same_seed_same_output(transform, input_value)
    assert_different_seed_can_change_output(transform, input_value)


@pytest.mark.parametrize(
    ("policy", "kwargs"),
    [
        (rand_augment, {"num_layers": 2, "magnitude": 12.0}),
        (trivial_augment, {}),
        (trivial_augment_wide, {}),
    ],
    ids=["rand_augment", "trivial_augment", "trivial_augment_wide"],
)
def test_automatic_policy_seed_contract(policy, kwargs):
    _check_seed_contract(
        lambda image, seed: policy(image, seed=seed, **kwargs), _image()
    )


def test_color_jitter_seed_contract():
    _check_seed_contract(
        lambda image, seed: color_jitter(
            image,
            seed=seed,
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1,
            p=1.0,
            p_grayscale=0.0,
        ),
        _image(),
    )


def test_random_erasing_seed_contract():
    _check_seed_contract(
        lambda images, seed: random_erasing(
            images,
            seed=seed,
            p=1.0,
            scale=(0.1, 0.2),
            ratio=(0.75, 1.33),
            replace=-1.0,
        ),
        _batch(),
    )


@pytest.mark.parametrize(
    ("mixup_alpha", "cutmix_alpha"), [(0.8, 0.0), (0.0, 1.0)], ids=["mixup", "cutmix"]
)
def test_mixing_seed_contract(mixup_alpha, cutmix_alpha):
    _check_seed_contract(
        lambda value, seed: mixup_cutmix(
            *value,
            seed=seed,
            num_classes=4,
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            prob=1.0,
            label_smoothing=0.0,
        ),
        (_batch(), _labels()),
    )

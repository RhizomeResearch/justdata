import numpy as np
import tensorflow as tf

from augmentation_harness import (
    assert_different_seed_can_change_output,
    assert_eval_disables_transform,
    assert_same_seed_same_output,
)
from justdata.vision.augmentations.auto import rand_augment, trivial_augment
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


def _check_seed_contract(transform, input_value):
    assert_same_seed_same_output(transform, input_value)
    assert_different_seed_can_change_output(transform, input_value)
    assert_eval_disables_transform(transform, input_value)


def test_rand_augment_seed_contract():
    def transform(image, seed, is_training):
        if not is_training:
            return image
        return rand_augment(image, seed=seed, num_layers=2, magnitude=12.0)

    _check_seed_contract(transform, _image())


def test_trivial_augment_seed_contract():
    def transform(image, seed, is_training):
        if not is_training:
            return image
        return trivial_augment(image, seed=seed)

    _check_seed_contract(transform, _image())


def test_color_jitter_seed_contract():
    def transform(image, seed, is_training):
        if not is_training:
            return image
        return color_jitter(
            image,
            seed=seed,
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1,
            p=1.0,
            p_grayscale=0.0,
        )

    _check_seed_contract(transform, _image())


def test_random_erasing_seed_contract():
    def transform(images, seed, is_training):
        if not is_training:
            return images
        return random_erasing(
            images,
            seed=seed,
            p=1.0,
            scale=(0.1, 0.2),
            ratio=(0.75, 1.33),
            replace=-1.0,
        )

    _check_seed_contract(transform, _batch())


def test_mixup_seed_contract():
    input_value = (_batch(), _labels())

    def transform(value, seed, is_training):
        if not is_training:
            return value
        images, labels = value
        return mixup_cutmix(
            images,
            labels,
            seed=seed,
            num_classes=4,
            mixup_alpha=0.8,
            cutmix_alpha=0.0,
            prob=1.0,
            label_smoothing=0.0,
        )

    _check_seed_contract(transform, input_value)


def test_cutmix_seed_contract():
    input_value = (_batch(), _labels())

    def transform(value, seed, is_training):
        if not is_training:
            return value
        images, labels = value
        return mixup_cutmix(
            images,
            labels,
            seed=seed,
            num_classes=4,
            mixup_alpha=0.0,
            cutmix_alpha=1.0,
            prob=1.0,
            label_smoothing=0.0,
        )

    _check_seed_contract(transform, input_value)

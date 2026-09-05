import numpy as np
import pytest
import tensorflow as tf

from justdata.vision.augmentations.color import solarize
from justdata.vision.augmentations.mixing import mixup_cutmix, random_erasing
from justdata.vision.tasks.classification import make_late_augmentations


def _assert_exact(actual, expected):
    tf.nest.assert_same_structure(actual, expected)
    for left, right in zip(tf.nest.flatten(actual), tf.nest.flatten(expected)):
        left, right = np.asarray(left), np.asarray(right)
        assert left.shape == right.shape
        assert left.dtype == right.dtype
        # Include signed zero and NaN payloads in floating-point comparisons.
        if left.dtype.kind in "OUS":
            np.testing.assert_array_equal(left, right)
        else:
            assert left.tobytes() == right.tobytes()


@pytest.mark.parametrize("dtype", [tf.uint8, tf.float16, tf.float32])
@pytest.mark.parametrize(
    "options",
    [
        {"cutmix_alpha": 0.0},
        {"mixup_alpha": 0.0},
        {},
        {"prob": 0.0},
        {"prob": 0.5, "bce_target": True},
        {"mixup_alpha": 0.0, "cutmix_alpha": 0.0},
        {"random_erasing_prob": 0.6},
    ],
)
def test_late_mixing_matches_channel_last_reference(dtype, options):
    values = tf.reshape(tf.range(3 * 9 * 11 * 3), [3, 9, 11, 3])
    images = tf.cast(values % 255, dtype)
    if dtype == tf.float32:
        images = tf.tensor_scatter_nd_update(
            images,
            [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 1, 0]],
            [float("nan"), float("inf"), -float("inf"), -0.0],
        )
    labels = tf.constant([0, 2, 4])
    nchw = tf.transpose(images, [0, 3, 1, 2])
    stage = make_late_augmentations(**options)
    mixing_options = {k: v for k, v in options.items() if k != "random_erasing_prob"}

    @tf.function(
        input_signature=[
            tf.TensorSpec([None, None, None, None], dtype),
            tf.TensorSpec([2], tf.int64),
        ]
    )
    def run(value, seed):
        return stage({"image": value, "label": labels}, num_classes=5, seed=seed)

    for offset in range(3):
        seed = tf.constant([offset, 19], tf.int64)
        seeds = tf.random.split(seed, 2)
        expected_images = images
        if options.get("random_erasing_prob", 0.0) > 0:
            expected_images = random_erasing(
                expected_images, seeds[0], p=options["random_erasing_prob"]
            )
        expected_labels = labels
        if options.get("mixup_alpha", 0.8) > 0 or options.get("cutmix_alpha", 1.0) > 0:
            expected_images, expected_labels = mixup_cutmix(
                expected_images, labels, seeds[1], num_classes=5, **mixing_options
            )
        expected = {
            "image": tf.transpose(expected_images, [0, 3, 1, 2]),
            "label": expected_labels,
        }
        _assert_exact(run(nchw, seed), expected)
        _assert_exact(run(images, seed), expected)


@pytest.mark.parametrize("permute_image", [False, True])
@pytest.mark.parametrize("shape", [(1, 3, 7, 9), (1, 7, 9, 3), (1, 3, 3, 3)])
@pytest.mark.parametrize("label_mode", ["single_label", "multi_label", "event_frames"])
def test_late_mixing_preserves_layout_detection_and_dense_labels(
    permute_image, shape, label_mode
):
    images = tf.reshape(tf.cast(tf.range(np.prod(shape)), tf.float32), shape)
    labels = tf.constant([[0.25, 0.5, 0.25]])
    sample = {"image": images, "label": labels, "metadata": {"id": tf.constant(["a"])}}
    seed = tf.constant([11, 37])
    stage = make_late_augmentations(permute_image=permute_image, label_mode=label_mode)
    is_nchw = shape[1] <= 4 and shape[-1] > 4
    channel_last = tf.transpose(images, [0, 2, 3, 1]) if is_nchw else images
    mixed, target = mixup_cutmix(
        channel_last, labels, tf.random.split(seed, 2)[1], 3, label_mode=label_mode
    )
    if permute_image or is_nchw:
        mixed = tf.transpose(mixed, [0, 3, 1, 2])
    _assert_exact(
        stage(sample, num_classes=3, seed=seed),
        sample | {"image": mixed, "label": target},
    )


def test_large_cutmix_preserves_exact_label_weights():
    images = tf.random.stateless_uniform(
        [32, 224, 224, 3], [17, 9], minval=-2.0, maxval=3.0
    )
    labels = tf.one_hot(tf.range(32), 1000)
    seed = tf.constant([9, 17], tf.int64)
    expected_images, expected_labels = mixup_cutmix(
        images, labels, tf.random.split(seed, 2)[1], 1000, mixup_alpha=0.0
    )
    expected = {
        "image": tf.transpose(expected_images, [0, 3, 1, 2]),
        "label": expected_labels,
    }
    stage = tf.function(make_late_augmentations(mixup_alpha=0.0))
    for value in (images, tf.transpose(images, [0, 3, 1, 2])):
        _assert_exact(
            stage({"image": value, "label": labels}, num_classes=1000, seed=seed),
            expected,
        )


@pytest.mark.parametrize("channels", [1, 3, 7])
@pytest.mark.parametrize("dtype", [tf.uint8, tf.float32])
def test_erasing_shares_the_same_spatial_mask_across_channels(channels, dtype):
    mono = tf.reshape(
        tf.cast(tf.range(1, 1 + 4 * 13 * 17) % 253 + 1, dtype), [4, 13, 17, 1]
    )
    images = tf.repeat(mono, channels, axis=-1)
    for probability in [0.0, 0.25, 1.0]:
        expected = tf.repeat(
            random_erasing(mono, [13, 2], p=probability), channels, axis=-1
        )
        actual = random_erasing(images, [13, 2], p=probability)
        _assert_exact(actual, expected)
        assert actual.shape == images.shape


@pytest.mark.parametrize("probability", [0.0, 0.2, 1.0, 2.0])
@pytest.mark.parametrize("dtype", [tf.uint8, tf.float32])
def test_solarize_matches_sampled_gate_exactly(probability, dtype):
    image = tf.cast(tf.reshape(tf.range(3 * 9 * 11) % 256, [9, 11, 3]), dtype)
    for offset in range(8):
        seed = tf.constant([offset, 19])
        selected = tf.random.stateless_uniform([], seed=seed) < probability
        expected = image
        if selected:
            expected = tf.where(
                image < tf.cast(127.5, dtype), image, tf.cast(255, dtype) - image
            )
        _assert_exact(solarize(image, seed=seed, p=probability), expected)

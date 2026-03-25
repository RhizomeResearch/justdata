import numpy as np
import pytest
import tensorflow as tf

from justdata.acoustic.augment.batch import audio_mixup


def test_mixup_single_label_converts_to_soft_one_hot():
    x = tf.reshape(tf.range(4 * 3, dtype=tf.float32), [4, 3])
    labels = tf.constant([0, 1, 2, 3], dtype=tf.int64)

    mixed_x, mixed_y = audio_mixup(
        x,
        labels,
        seed=[3, 0],
        alpha=10.0,
        prob=1.0,
        label_mode="single_label",
        num_classes=4,
    )

    assert mixed_x.shape == x.shape
    assert mixed_y.shape == (4, 4)
    assert mixed_y.dtype == tf.float32
    np.testing.assert_allclose(tf.reduce_sum(mixed_y, axis=1).numpy(), np.ones(4))
    assert np.any((mixed_y.numpy() > 0.0) & (mixed_y.numpy() < 1.0))


def test_mixup_multi_label_soft_multihot():
    x = tf.reshape(tf.range(4 * 3, dtype=tf.float32), [4, 3])
    labels = tf.constant(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=tf.float32,
    )

    _mixed_x, mixed_y = audio_mixup(
        x,
        labels,
        seed=[4, 0],
        alpha=10.0,
        prob=1.0,
        label_mode="multi_label",
    )

    assert mixed_y.shape == labels.shape
    assert np.any((mixed_y.numpy() > 0.0) & (mixed_y.numpy() < 1.0))


def test_mixup_event_frames_shape():
    x = tf.reshape(tf.range(2 * 5 * 3, dtype=tf.float32), [2, 5, 3])
    labels = tf.ones([2, 5, 4], dtype=tf.float32)

    _mixed_x, mixed_y = audio_mixup(
        x,
        labels,
        seed=[5, 0],
        alpha=2.0,
        prob=1.0,
        label_mode="event_frames",
    )

    assert mixed_y.shape == labels.shape


def test_mixup_same_seed_same_permutation():
    x = tf.reshape(tf.range(4 * 3, dtype=tf.float32), [4, 3])
    labels = tf.constant([0, 1, 2, 3], dtype=tf.int64)

    first = audio_mixup(
        x,
        labels,
        seed=[6, 0],
        alpha=2.0,
        prob=1.0,
        label_mode="single_label",
        num_classes=4,
    )
    second = audio_mixup(
        x,
        labels,
        seed=[6, 0],
        alpha=2.0,
        prob=1.0,
        label_mode="single_label",
        num_classes=4,
    )

    np.testing.assert_allclose(first[0].numpy(), second[0].numpy())
    np.testing.assert_allclose(first[1].numpy(), second[1].numpy())


def test_mixup_prob_zero_noop():
    x = tf.reshape(tf.range(2 * 3, dtype=tf.float32), [2, 3])
    labels = tf.constant([0, 1], dtype=tf.int64)

    mixed_x, mixed_y = audio_mixup(
        x,
        labels,
        seed=[7, 0],
        alpha=2.0,
        prob=0.0,
        label_mode="single_label",
    )

    np.testing.assert_array_equal(mixed_x.numpy(), x.numpy())
    np.testing.assert_array_equal(mixed_y.numpy(), labels.numpy())


def test_mixup_requires_num_classes_for_single_label():
    x = tf.reshape(tf.range(2 * 3, dtype=tf.float32), [2, 3])
    labels = tf.constant([0, 1], dtype=tf.int64)

    with pytest.raises(ValueError, match="num_classes"):
        audio_mixup(
            x,
            labels,
            seed=[8, 0],
            alpha=2.0,
            prob=1.0,
            label_mode="single_label",
        )

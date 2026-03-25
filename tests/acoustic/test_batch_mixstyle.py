import numpy as np
import tensorflow as tf

from justdata.acoustic.augment.batch import audio_batch_mixstyle
from justdata.acoustic.schema import FEATURES, LABEL
from justdata.acoustic.tasks import make_late_augmentations


def test_batch_mixstyle_preserves_labels():
    x = tf.reshape(tf.range(3 * 4 * 5, dtype=tf.float32), [3, 4, 5])
    labels = tf.constant([0, 1, 2], dtype=tf.int64)

    mixed_x, mixed_y = audio_batch_mixstyle(
        x,
        labels,
        seed=[31, 0],
        prob=1.0,
        mix="global",
        layout="btf",
    )

    assert mixed_x.shape == x.shape
    np.testing.assert_array_equal(mixed_y.numpy(), labels.numpy())


def test_audio_text_mixup_disabled_by_default():
    batch = {
        FEATURES: tf.reshape(tf.range(2 * 4 * 3, dtype=tf.float32), [2, 4, 3]),
        LABEL: tf.constant(["engine idling", "dog bark"]),
    }
    late_augment = make_late_augmentations(
        batch_augmentations={"mixup": {"alpha": 1.0, "prob": 1.0}},
        label_transform={"mode": "text"},
        spectrogram_layout="btf",
    )

    result = late_augment(batch, seed=tf.constant([32, 0], dtype=tf.int32))

    np.testing.assert_array_equal(result[FEATURES].numpy(), batch[FEATURES].numpy())
    np.testing.assert_array_equal(result[LABEL].numpy(), batch[LABEL].numpy())

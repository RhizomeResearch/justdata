import numpy as np
import tensorflow as tf

from justdata.acoustic.augment.batch import audio_cutmix_spec


def _paired_spectrogram(time: int = 6, frequency: int = 5) -> tf.Tensor:
    return tf.stack(
        [
            tf.zeros([time, frequency], dtype=tf.float32),
            tf.ones([time, frequency], dtype=tf.float32),
        ],
        axis=0,
    )


def test_cutmix_time_replaces_time_region():
    x = _paired_spectrogram()
    labels = tf.constant([[1.0, 0.0], [0.0, 1.0]], dtype=tf.float32)

    mixed_x, _mixed_y = audio_cutmix_spec(
        x,
        labels,
        seed=[11, 0],
        alpha=1.0,
        prob=1.0,
        axes="time",
        label_mode="multi_label",
        layout="btf",
    )

    changed = mixed_x.numpy()[0] != x.numpy()[0]
    changed_time = np.any(changed, axis=1)

    assert changed_time.any()
    np.testing.assert_array_equal(
        changed,
        np.broadcast_to(changed_time[:, np.newaxis], changed.shape),
    )


def test_cutmix_frequency_replaces_frequency_region():
    x = _paired_spectrogram()
    labels = tf.constant([[1.0, 0.0], [0.0, 1.0]], dtype=tf.float32)

    mixed_x, _mixed_y = audio_cutmix_spec(
        x,
        labels,
        seed=[12, 0],
        alpha=1.0,
        prob=1.0,
        axes="frequency",
        label_mode="multi_label",
        layout="btf",
    )

    changed = mixed_x.numpy()[0] != x.numpy()[0]
    changed_frequency = np.any(changed, axis=0)

    assert changed_frequency.any()
    np.testing.assert_array_equal(
        changed,
        np.broadcast_to(changed_frequency[np.newaxis, :], changed.shape),
    )


def test_cutmix_time_frequency_area_lambda():
    x = _paired_spectrogram(time=6, frequency=7)
    labels = tf.constant([[1.0, 0.0], [0.0, 1.0]], dtype=tf.float32)

    mixed_x, mixed_y = audio_cutmix_spec(
        x,
        labels,
        seed=[13, 0],
        alpha=1.0,
        prob=1.0,
        axes="time_frequency",
        label_mode="multi_label",
        layout="btf",
    )

    replaced = np.count_nonzero(mixed_x.numpy()[0] == 1.0)
    total = np.prod(x.shape[1:])
    expected_lambda = 1.0 - replaced / total

    np.testing.assert_allclose(mixed_y.numpy()[0], [expected_lambda, 1.0 - expected_lambda])


def test_cutmix_event_frame_splice_time_axis():
    x = _paired_spectrogram(time=5, frequency=4)
    labels = tf.stack(
        [
            tf.tile(tf.constant([[1.0, 0.0]], dtype=tf.float32), [5, 1]),
            tf.tile(tf.constant([[0.0, 1.0]], dtype=tf.float32), [5, 1]),
        ],
        axis=0,
    )

    mixed_x, mixed_y = audio_cutmix_spec(
        x,
        labels,
        seed=[14, 0],
        alpha=1.0,
        prob=1.0,
        axes="time",
        label_mode="event_frames",
        layout="btf",
    )

    changed_time = np.any(mixed_x.numpy()[0] != x.numpy()[0], axis=1)
    expected = labels.numpy()[0].copy()
    expected[changed_time] = labels.numpy()[1, changed_time]

    np.testing.assert_allclose(mixed_y.numpy()[0], expected)

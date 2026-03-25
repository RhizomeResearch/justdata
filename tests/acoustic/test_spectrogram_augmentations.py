import numpy as np
import pytest
import tensorflow as tf

from augmentation_harness import (
    assert_eval_disables_transform,
    assert_same_seed_same_output,
)
from justdata.acoustic.augment import (
    frequency_mask,
    frequency_mixstyle,
    mel_bin_shift,
    mixstyle,
    random_eq,
    spectrogram_time_roll,
    time_frequency_erasing,
    time_mask,
)
from justdata.acoustic.registry import (
    get_audio_spectrogram_augment,
    get_audio_spectrogram_augment_metadata,
)
from justdata.vision.augmentations.registry import get_augment_strategy


def _masked_output(transform, x):
    for offset in range(16):
        y = transform(x, seed=tf.constant([37 + offset, 11], dtype=tf.int32))
        if np.any(y.numpy() != x.numpy()):
            return y
    raise AssertionError("transform did not mask any element for the tested seeds")


def test_frequency_mask_masks_only_frequency_axis():
    x = tf.ones([6, 8], dtype=tf.float32)
    y = _masked_output(
        lambda value, seed: frequency_mask(
            value,
            seed=seed,
            max_width=3,
            fill_value="zero",
        ),
        x,
    )

    changed = y.numpy() != x.numpy()
    changed_freq = np.any(changed, axis=0)

    assert changed_freq.any()
    np.testing.assert_array_equal(changed, np.broadcast_to(changed_freq, changed.shape))


def test_time_mask_masks_only_time_axis():
    x = tf.ones([6, 8], dtype=tf.float32)
    y = _masked_output(
        lambda value, seed: time_mask(
            value,
            seed=seed,
            max_width=3,
            fill_value="zero",
        ),
        x,
    )

    changed = y.numpy() != x.numpy()
    changed_time = np.any(changed, axis=1)

    assert changed_time.any()
    np.testing.assert_array_equal(
        changed,
        np.broadcast_to(changed_time[:, np.newaxis], changed.shape),
    )


def test_mask_same_seed_same_output():
    transform = lambda x, seed, is_training: frequency_mask(
        x,
        seed=seed,
        max_width=3,
        fill_value="zero",
        is_training=is_training,
    )

    assert_same_seed_same_output(transform, tf.ones([8, 10], dtype=tf.float32))


def test_mask_eval_disabled():
    transform = lambda x, seed, is_training: time_mask(
        x,
        seed=seed,
        max_width=3,
        fill_value="zero",
        is_training=is_training,
    )

    assert_eval_disables_transform(transform, tf.ones([8, 10], dtype=tf.float32))


def test_time_roll_preserves_values():
    x = tf.reshape(tf.range(20, dtype=tf.float32), [4, 5])

    y = spectrogram_time_roll(x, seed=[1, 0], shift=2)

    np.testing.assert_array_equal(np.sort(y.numpy(), axis=None), np.sort(x.numpy(), axis=None))


def test_mel_bin_shift_zero_pads_edges():
    x = tf.reshape(tf.range(15, dtype=tf.float32), [3, 5])

    y = mel_bin_shift(x, seed=[2, 0], shift=2, zero_pad=True)

    np.testing.assert_array_equal(y.numpy()[:, :2], np.zeros([3, 2], dtype=np.float32))
    np.testing.assert_array_equal(y.numpy()[:, 2:], x.numpy()[:, :3])


def test_time_frequency_erasing_shape():
    x = tf.ones([7, 9, 2], dtype=tf.float32)

    y = time_frequency_erasing(
        x,
        seed=[3, 0],
        max_time_width=3,
        max_freq_width=4,
        fill_value="zero",
    )

    assert y.shape == x.shape


def test_random_eq_changes_frequency_bands():
    x = tf.ones([5, 12], dtype=tf.float32)

    y = random_eq(
        x,
        seed=[4, 0],
        num_bands=2,
        min_db=6.0,
        max_db=6.0,
        min_band_width=2,
        max_band_width=2,
    )

    ratio = y.numpy() / x.numpy()
    assert np.any(np.abs(ratio - 1.0) > 1e-5)
    np.testing.assert_allclose(
        ratio,
        np.broadcast_to(ratio[:1, :], ratio.shape),
        rtol=1e-6,
    )


def test_mixstyle_same_shape():
    x = tf.reshape(tf.range(2 * 4 * 3, dtype=tf.float32), [2, 4, 3])

    y = mixstyle(x, seed=[5, 0], alpha=0.4, layout="btf")

    assert y.shape == x.shape


def test_mixstyle_preserves_batch_size():
    x = tf.ones([4, 6, 5, 2], dtype=tf.float32)

    y = mixstyle(x, seed=[6, 0], alpha=0.4, layout="btfc")

    assert y.shape[0] == x.shape[0]


def test_frequency_mixstyle_stats_axes():
    x = np.zeros([2, 4, 3, 2], dtype=np.float32)
    for batch in range(2):
        for freq in range(3):
            x[batch, :, freq, :] = batch * 10.0 + freq

    y = frequency_mixstyle(tf.constant(x), seed=[7, 0], alpha=0.4, layout="btfc")
    per_frequency_mean = np.mean(y.numpy(), axis=(1, 3))

    assert per_frequency_mean.shape == (2, 3)
    assert not np.allclose(per_frequency_mean[:, 0], per_frequency_mean[:, 1])


def test_spectrogram_augments_have_audio_metadata_only():
    fn = get_audio_spectrogram_augment("frequency_mask")
    metadata = get_audio_spectrogram_augment_metadata("frequency_mask")

    assert fn.domain == "spectrogram"
    assert metadata.name == "frequency_mask"
    assert metadata.domain == "spectrogram"
    assert metadata.is_training_only is True
    assert metadata.requires_labels is False
    with pytest.raises(ValueError):
        get_augment_strategy("frequency_mask")

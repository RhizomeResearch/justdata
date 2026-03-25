import numpy as np
import tensorflow as tf

from justdata.acoustic.compat.ast import (
    ast_add_noise_and_roll,
    ast_frontend,
    ast_frequency_mask,
    ast_kaldi_fbank,
    ast_mix_waveforms,
    ast_normalize_fbank,
    ast_pad_or_crop_fbank,
    ast_time_mask,
)


def test_ast_kaldi_fbank_raw_frame_counts():
    frontend = ast_frontend()

    assert ast_kaldi_fbank(tf.zeros([160000, 1]), frontend).shape == (998, 128, 1)
    assert ast_kaldi_fbank(tf.zeros([80000, 1]), frontend).shape == (498, 128, 1)
    assert ast_kaldi_fbank(tf.zeros([16000, 1]), frontend).shape == (98, 128, 1)


def test_ast_pad_or_crop_fbank_right_pads_and_front_crops():
    fbank = tf.reshape(tf.range(6, dtype=tf.float32), [2, 3, 1])

    padded = ast_pad_or_crop_fbank(fbank, 4)
    cropped = ast_pad_or_crop_fbank(fbank, 1)

    assert padded.shape == (4, 3, 1)
    np.testing.assert_array_equal(padded.numpy()[:2], fbank.numpy())
    np.testing.assert_array_equal(padded.numpy()[2:], np.zeros([2, 3, 1], dtype=np.float32))
    np.testing.assert_array_equal(cropped.numpy(), fbank.numpy()[:1])


def test_ast_normalization_matches_reference_formula_without_epsilon():
    result = ast_normalize_fbank(tf.zeros([2, 3, 1]), mean=-4.0, std=2.0)

    np.testing.assert_allclose(result.numpy(), np.ones([2, 3, 1]), rtol=1e-6)


def test_ast_frequency_and_time_masks_apply_requested_spans():
    fbank = tf.ones([5, 6, 1], dtype=tf.float32)

    freq_masked = ast_frequency_mask(fbank, max_width=4, start=2, width=3)
    time_masked = ast_time_mask(fbank, max_width=4, start=1, width=2)

    assert np.all(freq_masked.numpy()[:, 2:5, :] == 0.0)
    assert np.all(freq_masked.numpy()[:, :2, :] == 1.0)
    assert np.all(time_masked.numpy()[1:3, :, :] == 0.0)
    assert np.all(time_masked.numpy()[0, :, :] == 1.0)


def test_ast_noise_roll_can_be_operation_exact_without_noise():
    fbank = tf.reshape(tf.range(12, dtype=tf.float32), [4, 3, 1])

    result = ast_add_noise_and_roll(fbank, noise_scale=0.0, roll_shift=1)

    np.testing.assert_array_equal(result.numpy(), np.roll(fbank.numpy(), 1, axis=0))


def test_ast_mix_waveforms_matches_reference_length_policy():
    waveform = tf.reshape(tf.constant([1.0, 3.0, 5.0, 7.0]), [4, 1])
    shorter_partner = tf.reshape(tf.constant([2.0, 4.0]), [2, 1])

    result = ast_mix_waveforms(waveform, shorter_partner, 0.25)

    main = waveform.numpy() - waveform.numpy().mean()
    partner = shorter_partner.numpy() - shorter_partner.numpy().mean()
    partner = np.pad(partner, [[0, 2], [0, 0]])
    expected = 0.25 * main + 0.75 * partner
    expected = expected - expected.mean()
    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6)

import numpy as np
import tensorflow as tf

import justdata.acoustic.corruptions  # noqa: F401
from justdata.acoustic.corruptions.dynamics import CLIPPING_THRESHOLD
from justdata.acoustic.corruptions.registry import apply_audio_corruption


SAMPLE_RATE = 32000
SEED = tf.constant([11, 29], dtype=tf.int32)


def _sine(freq_hz: float, n: int = 4096) -> tf.Tensor:
    t = tf.range(n, dtype=tf.float32) / float(SAMPLE_RATE)
    return tf.sin(2.0 * np.pi * freq_hz * t)[:, tf.newaxis]


def _snr_db(reference: tf.Tensor, corrupted: tf.Tensor) -> float:
    noise = tf.cast(corrupted - reference, tf.float32)
    signal_power = tf.reduce_mean(tf.square(reference))
    noise_power = tf.reduce_mean(tf.square(noise))
    return float(
        (10.0 * tf.math.log(signal_power / noise_power) / tf.math.log(10.0)).numpy()
    )


def _band_energy(x: tf.Tensor, low_hz: float, high_hz: float) -> float:
    x = tf.squeeze(tf.cast(x, tf.float32), axis=-1)
    spectrum = tf.abs(tf.signal.rfft(x))
    freqs = tf.linspace(0.0, SAMPLE_RATE / 2.0, tf.shape(spectrum)[0])
    mask = tf.logical_and(freqs >= low_hz, freqs <= high_hz)
    return float(tf.reduce_sum(tf.boolean_mask(tf.square(spectrum), mask)).numpy())


def test_white_noise_snr_decreases_with_severity():
    audio = _sine(1000.0)

    mild = apply_audio_corruption(audio, "additive_white_noise", 1, SEED)
    severe = apply_audio_corruption(audio, "additive_white_noise", 5, SEED)

    assert _snr_db(audio, severe) < _snr_db(audio, mild)


def test_pink_noise_not_equal_white_noise():
    audio = _sine(1000.0)

    white = apply_audio_corruption(audio, "additive_white_noise", 3, SEED)
    pink = apply_audio_corruption(audio, "additive_pink_noise", 3, SEED)

    assert not np.allclose(white.numpy(), pink.numpy())


def test_lowpass_reduces_high_frequency_energy():
    audio = _sine(1000.0) + 0.5 * _sine(10000.0)

    corrupted = apply_audio_corruption(
        audio,
        "low_pass",
        5,
        SEED,
        config={"sample_rate": SAMPLE_RATE},
    )

    assert (
        _band_energy(corrupted, 8000.0, 12000.0)
        < _band_energy(audio, 8000.0, 12000.0) * 0.1
    )


def test_highpass_reduces_low_frequency_energy():
    audio = _sine(500.0) + 0.5 * _sine(8000.0)

    corrupted = apply_audio_corruption(
        audio,
        "high_pass",
        5,
        SEED,
        config={"sample_rate": SAMPLE_RATE},
    )

    assert (
        _band_energy(corrupted, 100.0, 1000.0)
        < _band_energy(audio, 100.0, 1000.0) * 0.1
    )


def test_clipping_threshold_decreases_with_severity():
    audio = tf.linspace(-1.0, 1.0, 1024)[:, tf.newaxis]

    mild = apply_audio_corruption(audio, "clipping", 1, SEED)
    severe = apply_audio_corruption(audio, "clipping", 5, SEED)

    assert CLIPPING_THRESHOLD[5] < CLIPPING_THRESHOLD[1]
    assert float(tf.reduce_max(tf.abs(severe)).numpy()) < float(
        tf.reduce_max(tf.abs(mild)).numpy()
    )


def test_drc_reduces_dynamic_range():
    audio = tf.linspace(-1.0, 1.0, 4096)[:, tf.newaxis]

    compressed = apply_audio_corruption(audio, "dynamic_range_compression", 5, SEED)

    assert float(tf.reduce_max(tf.abs(compressed)).numpy()) < float(
        tf.reduce_max(tf.abs(audio)).numpy()
    )


def test_packet_dropout_inserts_gaps():
    audio = tf.ones([SAMPLE_RATE, 1], dtype=tf.float32)

    corrupted = apply_audio_corruption(
        audio,
        "packet_dropout_gaps",
        5,
        SEED,
        config={"sample_rate": SAMPLE_RATE},
    )

    assert int(tf.reduce_sum(tf.cast(tf.equal(corrupted, 0.0), tf.int32)).numpy()) > 0


def test_corruption_same_seed_same_output():
    audio = _sine(1000.0)

    first = apply_audio_corruption(audio, "additive_white_noise", 3, SEED)
    second = apply_audio_corruption(audio, "additive_white_noise", 3, SEED)

    np.testing.assert_allclose(first.numpy(), second.numpy())


def test_corruption_different_seed_can_change_output():
    audio = _sine(1000.0)

    first = apply_audio_corruption(audio, "additive_white_noise", 3, SEED)
    second = apply_audio_corruption(
        audio,
        "additive_white_noise",
        3,
        tf.constant([11, 30], dtype=tf.int32),
    )

    assert not np.allclose(first.numpy(), second.numpy())

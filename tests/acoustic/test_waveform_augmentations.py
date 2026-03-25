import numpy as np
import tensorflow as tf

from augmentation_harness import (
    assert_different_seed_can_change_output,
    assert_eval_disables_transform,
    assert_same_seed_same_output,
)
from justdata.acoustic.augment import (
    additive_noise,
    codec_simulation,
    dynamic_range_compression,
    make_colored_noise,
    polarity_inversion,
    random_clipping,
    random_crop,
    random_gain,
    rir_convolution,
    speed_perturb,
    time_shift,
)
from justdata.acoustic.schema import METADATA, SAMPLE_RATE, WAVEFORM
from justdata.acoustic.tasks import make_augmentations


def _waveform(length: int = 2048) -> tf.Tensor:
    t = tf.linspace(0.0, 1.0, length)
    x = 0.5 * tf.sin(2.0 * np.pi * 17.0 * t)
    return x[:, tf.newaxis]


def test_random_gain_same_seed_same_output():
    transform = lambda x, seed, is_training: random_gain(
        x,
        16000,
        seed=seed,
        prob=1.0,
        min_db=-6.0,
        max_db=6.0,
        is_training=is_training,
    )

    assert_same_seed_same_output(transform, _waveform())


def test_random_gain_different_seed_different_output():
    transform = lambda x, seed, is_training: random_gain(
        x,
        16000,
        seed=seed,
        prob=1.0,
        min_db=-6.0,
        max_db=6.0,
        is_training=is_training,
    )

    assert_different_seed_can_change_output(transform, _waveform())


def test_random_gain_db_scale_exact_for_forced_gain():
    audio = tf.constant([[0.25], [-0.5], [1.0]], dtype=tf.float32)

    result = random_gain(
        audio,
        16000,
        seed=[1, 0],
        prob=1.0,
        min_db=6.0,
        max_db=6.0,
    )

    np.testing.assert_allclose(result.numpy(), audio.numpy() * (10.0 ** (6.0 / 20.0)))


def test_time_shift_roll_preserves_energy():
    audio = _waveform(256)

    result = time_shift(
        audio,
        16000,
        seed=[2, 0],
        prob=1.0,
        max_shift_seconds=0.01,
        mode="roll",
    )

    np.testing.assert_allclose(
        tf.reduce_sum(tf.square(result)).numpy(),
        tf.reduce_sum(tf.square(audio)).numpy(),
        rtol=1e-6,
    )


def test_time_shift_zero_inserts_zeros():
    audio = tf.reshape(tf.range(10, dtype=tf.float32), [10, 1])

    result = time_shift(
        audio,
        seed=[3, 0],
        prob=1.0,
        mode="zero",
        shift_samples=3,
    )

    np.testing.assert_allclose(result.numpy()[:3, 0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(result.numpy()[3:, 0], np.arange(7, dtype=np.float32))


def test_random_crop_shape():
    audio = _waveform(200)

    result = random_crop(audio, seed=[4, 0], target_length=80)

    assert result.shape == (80, 1)


def test_additive_noise_snr_close():
    audio = _waveform(4096)

    result = additive_noise(
        audio,
        16000,
        seed=[5, 0],
        prob=1.0,
        snr_db_min=20.0,
        snr_db_max=20.0,
        noise_kind="white",
    )
    noise = result - audio
    snr_db = 10.0 * np.log10(
        tf.reduce_mean(tf.square(audio)).numpy()
        / tf.reduce_mean(tf.square(noise)).numpy()
    )

    assert abs(snr_db - 20.0) < 0.2


def test_pink_noise_spectrum_slope_negative():
    noise = make_colored_noise([8192, 1], seed=[6, 0], kind="pink").numpy()[:, 0]
    spectrum = np.fft.rfft(noise)
    power = np.abs(spectrum[1:]) ** 2
    frequencies = np.arange(1, power.shape[0] + 1)

    slope, _ = np.polyfit(np.log(frequencies), np.log(power + 1e-12), 1)

    assert slope < -0.5


def test_polarity_inversion_sign_flip():
    audio = tf.constant([[0.25], [-0.5], [1.0]], dtype=tf.float32)

    result = polarity_inversion(audio, seed=[7, 0], prob=1.0)

    np.testing.assert_allclose(result.numpy(), -audio.numpy())


def test_random_clipping_threshold_respected():
    audio = tf.constant([[-0.9], [-0.2], [0.2], [0.9]], dtype=tf.float32)

    result = random_clipping(
        audio,
        seed=[8, 0],
        prob=1.0,
        min_threshold=0.3,
        max_threshold=0.3,
    )

    assert np.max(np.abs(result.numpy())) <= 0.300001
    np.testing.assert_allclose(result.numpy()[:, 0], [-0.3, -0.2, 0.2, 0.3])


def test_dynamic_range_compression_reduces_peaks():
    audio = tf.constant([[0.05], [0.2], [0.9], [-0.95]], dtype=tf.float32)

    result = dynamic_range_compression(
        audio,
        seed=[9, 0],
        prob=1.0,
        threshold_db_min=-12.0,
        threshold_db_max=-12.0,
        ratio_min=4.0,
        ratio_max=4.0,
    )

    assert tf.reduce_max(tf.abs(result)).numpy() < tf.reduce_max(tf.abs(audio)).numpy()


def test_rir_convolution_shape():
    audio = tf.concat([_waveform(128), _waveform(128) * 0.5], axis=1)

    result = rir_convolution(
        audio,
        16000,
        seed=[10, 0],
        prob=1.0,
        rir=tf.constant([0.5, 1.0, 0.5], dtype=tf.float32),
    )

    assert result.shape == audio.shape


def test_speed_perturb_preserves_target_length_after_resample():
    audio = _waveform(512)

    result = speed_perturb(audio, 16000, seed=[11, 0], prob=1.0, rates=(1.2,))

    assert result.shape == audio.shape


def test_codec_simulation_no_nan():
    audio = _waveform(256)

    for codec in ("mulaw", "alaw", "mp3_proxy"):
        result = codec_simulation(audio, 16000, seed=[12, 0], prob=1.0, codec=codec)
        assert np.isfinite(result.numpy()).all()


def test_augment_disabled_eval():
    transform = lambda x, seed, is_training: random_gain(
        x,
        16000,
        seed=seed,
        prob=1.0,
        min_db=6.0,
        max_db=6.0,
        is_training=is_training,
    )

    assert_eval_disables_transform(transform, _waveform())


def test_augment_pipeline_preserves_metadata():
    sample = {
        WAVEFORM: _waveform(128),
        SAMPLE_RATE: tf.constant(16000, dtype=tf.int32),
        "label": tf.constant(3, dtype=tf.int64),
        METADATA: {
            "clip_id": tf.constant("clip-a"),
            "fold": tf.constant(2, dtype=tf.int32),
        },
    }
    augment = make_augmentations(
        waveform_augmentations={
            "random_gain": {"prob": 1.0, "min_db": 6.0, "max_db": 6.0}
        },
        is_training=True,
    )

    result = augment(sample, seed=tf.constant([13, 0], dtype=tf.int32))

    assert result["label"].numpy() == sample["label"].numpy()
    assert result[METADATA].keys() == sample[METADATA].keys()
    assert result[METADATA]["clip_id"].numpy() == sample[METADATA]["clip_id"].numpy()
    assert result[METADATA]["fold"].numpy() == sample[METADATA]["fold"].numpy()
    assert not np.array_equal(result[WAVEFORM].numpy(), sample[WAVEFORM].numpy())

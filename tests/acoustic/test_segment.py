import numpy as np
import tensorflow as tf

from justdata.acoustic.configs import SegmentStrategyConfig
from justdata.acoustic.segment import pad_waveform, segment_waveform


def _config(**overrides):
    data = {
        "clip_duration": 1.0,
        "train_mode": "random_crop",
        "eval_mode": "center_crop",
        "pad_mode": "zero",
        "pad_position": "right",
    }
    data.update(overrides)
    return SegmentStrategyConfig(**data)


def test_segment_random_crop_same_seed_same_crop():
    audio = tf.reshape(tf.range(20, dtype=tf.float32), [20, 1])
    config = _config(clip_duration=0.5)

    first = segment_waveform(audio, 10, config, is_training=True, seed=[123, 0])
    second = segment_waveform(audio, 10, config, is_training=True, seed=[123, 0])

    np.testing.assert_allclose(first["audio"].numpy(), second["audio"].numpy())


def test_segment_random_crop_different_seed_different_crop():
    audio = tf.reshape(tf.range(100, dtype=tf.float32), [100, 1])
    config = _config(clip_duration=1.0)

    first = segment_waveform(audio, 10, config, is_training=True, seed=[1, 0])
    second = segment_waveform(audio, 10, config, is_training=True, seed=[999, 0])

    assert not np.array_equal(first["audio"].numpy(), second["audio"].numpy())


def test_segment_eval_center_crop_deterministic():
    audio = tf.reshape(tf.range(10, dtype=tf.float32), [10, 1])
    config = _config(clip_duration=0.4, eval_mode="center_crop")

    first = segment_waveform(audio, 10, config, is_training=False, seed=None)
    second = segment_waveform(audio, 10, config, is_training=False, seed=[99, 1])

    np.testing.assert_allclose(first["audio"].numpy(), second["audio"].numpy())
    np.testing.assert_allclose(first["audio"].numpy()[:, 0], [3.0, 4.0, 5.0, 6.0])


def test_segment_eval_multi_crop_start_times_evenly_spaced():
    audio = tf.zeros([100, 1], dtype=tf.float32)
    config = _config(clip_duration=2.0, eval_mode="multi_crop", num_views=5)

    result = segment_waveform(audio, 10, config, is_training=False, seed=None)

    np.testing.assert_allclose(
        result["metadata"]["view_start_time"].numpy(),
        [0.0, 2.0, 4.0, 6.0, 8.0],
    )
    assert result["audio"].shape == (5, 20, 1)


def test_pad_zero():
    audio = tf.constant([[1.0], [2.0]], dtype=tf.float32)

    result = pad_waveform(audio, tf.constant(5), "zero")

    np.testing.assert_allclose(result.numpy()[:, 0], [1.0, 2.0, 0.0, 0.0, 0.0])


def test_pad_repeat():
    audio = tf.constant([[1.0], [2.0]], dtype=tf.float32)

    result = pad_waveform(audio, tf.constant(5), "repeat")

    np.testing.assert_allclose(result.numpy()[:, 0], [1.0, 2.0, 1.0, 2.0, 1.0])


def test_pad_reflect():
    audio = tf.constant([[1.0], [2.0], [3.0]], dtype=tf.float32)

    result = pad_waveform(audio, tf.constant(6), "reflect")

    np.testing.assert_allclose(result.numpy()[:, 0], [1.0, 2.0, 3.0, 2.0, 1.0, 2.0])


def test_pad_reflect_single_sample_zero_pads():
    audio = tf.constant([[1.0]], dtype=tf.float32)

    result = pad_waveform(audio, tf.constant(4), "reflect")

    np.testing.assert_allclose(result.numpy()[:, 0], [1.0, 0.0, 0.0, 0.0])


def test_dcase_keep_1s_does_not_tile():
    sample_rate = 44100
    audio = tf.ones([sample_rate, 1], dtype=tf.float32)
    config = _config(
        clip_duration=10.0,
        eval_mode="center_crop",
        pad_mode="repeat",
        duration_policy="keep_1s",
    )

    result = segment_waveform(audio, sample_rate, config, is_training=False, seed=None)

    assert result["audio"].shape == (sample_rate, 1)


def test_dcase_tile_policy_must_be_explicit():
    sample_rate = 4
    audio = tf.reshape(tf.range(4, dtype=tf.float32), [4, 1])
    implicit = _config(
        clip_duration=2.0,
        eval_mode="center_crop",
        pad_mode="repeat",
        duration_policy="keep_1s",
    )
    explicit = _config(
        clip_duration=2.0,
        eval_mode="center_crop",
        pad_mode="zero",
        duration_policy="tile_to_model_duration",
    )

    kept = segment_waveform(audio, sample_rate, implicit, is_training=False, seed=None)
    tiled = segment_waveform(audio, sample_rate, explicit, is_training=False, seed=None)

    assert kept["audio"].shape == (4, 1)
    np.testing.assert_allclose(tiled["audio"].numpy()[:, 0], [0, 1, 2, 3, 0, 1, 2, 3])


def test_view_metadata_start_end_times():
    audio = tf.reshape(tf.range(10, dtype=tf.float32), [10, 1])
    config = _config(clip_duration=2.0, eval_mode="center_crop")

    result = segment_waveform(audio, 2, config, is_training=False, seed=None)

    np.testing.assert_allclose(result["metadata"]["view_start_time"].numpy(), [1.5])
    np.testing.assert_allclose(result["metadata"]["view_end_time"].numpy(), [3.5])
    np.testing.assert_array_equal(result["metadata"]["view_index"].numpy(), [0])

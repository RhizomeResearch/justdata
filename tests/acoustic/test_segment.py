import numpy as np
import pytest
import tensorflow as tf

from justdata.acoustic.configs import SegmentStrategyConfig
from justdata.acoustic.segment import pad_waveform, segment_waveform
from justdata.acoustic.stages import make_segment_stage


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


@pytest.mark.parametrize(
    ("pad_mode", "pad_position", "expected"),
    [
        ("zero", "right", [1, 2, 0, 0, 0]),
        ("zero", "center", [0, 1, 2, 0, 0]),
        ("repeat", "right", [1, 2, 1, 2, 1]),
        ("repeat", "center", [2, 1, 2, 1, 2]),
        ("reflect", "right", [1, 2, 1, 2, 1]),
        ("reflect", "center", [2, 1, 2, 1, 2]),
    ],
)
def test_pad_position_exact_samples(pad_mode, pad_position, expected):
    audio = tf.constant([[1.0], [2.0]])

    result = pad_waveform(
        audio,
        tf.constant(5),
        pad_mode,
        pad_position=pad_position,
        seed=[7, 0],
    )

    np.testing.assert_allclose(result.numpy()[:, 0], expected)


@pytest.mark.parametrize("pad_mode", ["zero", "repeat", "reflect"])
@pytest.mark.parametrize("pad_position", ["right", "center", "random"])
def test_pad_empty_waveform_has_target_shape(pad_mode, pad_position):
    result = pad_waveform(
        tf.zeros([0, 1]),
        tf.constant(4),
        pad_mode,
        pad_position=pad_position,
        seed=[7, 0],
    )

    assert result.shape == (4, 1)
    np.testing.assert_allclose(result.numpy(), 0.0)


def test_random_pad_position_is_stateless_and_seeded():
    audio = tf.constant([[1.0], [2.0]])

    first = pad_waveform(
        audio, tf.constant(20), "zero", pad_position="random", seed=[3, 0]
    )
    second = pad_waveform(
        audio, tf.constant(20), "zero", pad_position="random", seed=[3, 0]
    )
    different = pad_waveform(
        audio, tf.constant(20), "zero", pad_position="random", seed=[4, 0]
    )

    np.testing.assert_array_equal(first.numpy(), second.numpy())
    assert not np.array_equal(first.numpy(), different.numpy())


@pytest.mark.parametrize(
    "overrides,field",
    [({"drop_short": True}, "drop_short"), ({"min_duration": 0.5}, "min_duration")],
)
def test_unsupported_short_clip_controls_fail_fast(overrides, field):
    with pytest.raises(ValueError, match=field):
        _config(**overrides)


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


@pytest.mark.parametrize(
    ("eval_mode", "samples", "expected_duration"),
    [
        ("center_crop", 2, 1.0),
        ("center_crop", 8, 1.0),
        ("full", 8, 4.0),
        ("multi_crop", 8, 1.0),
    ],
)
def test_segment_stage_refreshes_duration(eval_mode, samples, expected_duration):
    sample = {
        "waveform": tf.ones([samples, 1]),
        "sample_rate": tf.constant(2),
        "duration": tf.constant(samples / 2),
        "metadata": {
            "duration": tf.constant(samples / 2),
            "original_duration": tf.constant(9.0),
        },
    }
    stage = make_segment_stage(
        _config(clip_duration=1.0, eval_mode=eval_mode, num_views=3),
        is_training=False,
    )

    result = stage(sample)

    assert float(result["duration"].numpy()) == expected_duration
    assert float(result["metadata"]["duration"].numpy()) == expected_duration
    assert float(result["metadata"]["original_duration"].numpy()) == 9.0

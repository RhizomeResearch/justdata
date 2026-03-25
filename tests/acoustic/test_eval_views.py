import numpy as np
import tensorflow as tf

from justdata.acoustic.configs import SegmentStrategyConfig
from justdata.acoustic.eval_views import generate_eval_views, make_eval_views


def _config(**overrides):
    data = {
        "clip_duration": 1.0,
        "train_mode": "random_crop",
        "eval_mode": "multi_crop",
        "pad_mode": "zero",
        "pad_position": "right",
        "num_views": 3,
    }
    data.update(overrides)
    return SegmentStrategyConfig(**data)


def test_generate_eval_views_is_deterministic():
    audio = tf.reshape(tf.range(20, dtype=tf.float32), [20, 1])
    config = _config(clip_duration=0.5, eval_mode="multi_crop", num_views=3)

    first = generate_eval_views(audio, 10, config)
    second = generate_eval_views(audio, 10, config)

    np.testing.assert_allclose(first["audio"].numpy(), second["audio"].numpy())
    np.testing.assert_allclose(
        first["metadata"]["view_start_time"].numpy(),
        second["metadata"]["view_start_time"].numpy(),
    )


def test_make_eval_views_merges_view_metadata():
    sample = {
        "waveform": tf.zeros([20, 1], dtype=tf.float32),
        "sample_rate": tf.constant(10, dtype=tf.int32),
        "metadata": {"clip_id": tf.constant("a")},
    }
    config = _config(clip_duration=0.5, eval_mode="multi_crop", num_views=3)

    result = make_eval_views(config)(sample)

    assert "clip_id" in result["metadata"]
    np.testing.assert_array_equal(result["metadata"]["view_index"].numpy(), [0, 1, 2])
    assert result["waveform"].shape == (3, 5, 1)

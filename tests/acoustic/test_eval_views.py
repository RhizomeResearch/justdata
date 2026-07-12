import numpy as np
import pytest
import tensorflow as tf

from justdata.acoustic.configs import (
    FrontendConfig,
    MelConfig,
    STFTConfig,
    SegmentStrategyConfig,
)
from justdata.acoustic.eval_views import generate_eval_views, make_eval_views
from justdata.acoustic.postprocessing import make_model_input_stage, preset_info


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


@pytest.mark.parametrize(
    ("frontend", "layout", "output_key", "expected_shape"),
    [
        (FrontendConfig(name="raw_waveform"), "btc", "waveform", (3, 16, 1)),
        (
            FrontendConfig(
                name="stft_magnitude",
                stft=STFTConfig(
                    sample_rate=16,
                    n_fft=4,
                    win_length=4,
                    hop_length=2,
                ),
            ),
            "btfc",
            "features",
            (3, 9, 3, 1),
        ),
        (
            FrontendConfig(
                name="logmel",
                stft=STFTConfig(
                    sample_rate=16,
                    n_fft=4,
                    win_length=4,
                    hop_length=2,
                ),
                mel=MelConfig(n_mels=2),
            ),
            "btf",
            "features",
            (3, 9, 2),
        ),
    ],
)
def test_fixed_eval_views_compose_with_frontends(
    frontend, layout, output_key, expected_shape
):
    config = _config(clip_duration=1.0, num_views=3)
    sample = make_eval_views(config)(
        {
            "waveform": tf.reshape(tf.range(32, dtype=tf.float32), [32, 1]),
            "sample_rate": tf.constant(16),
            "metadata": {"clip_id": tf.constant("a")},
        }
    )
    preset = preset_info(
        frontend=frontend,
        layout=layout,
        dtype="float32",
        target_sample_rate=16,
        segment=config,
    )

    result = make_model_input_stage(frontend, layout=layout, preset=preset)(sample)

    assert result[output_key].shape == expected_shape
    np.testing.assert_array_equal(result["metadata"]["view_index"], [0, 1, 2])

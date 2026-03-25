from uuid import uuid4

import pytest
import tensorflow as tf

import justdata.acoustic.corruptions  # noqa: F401
from justdata.acoustic.corruptions.registry import (
    apply_audio_corruption,
    list_audio_corruptions,
    register_audio_corruption,
)


def test_register_audio_corruption_duplicate_raises():
    name = f"unit_audio_corruption_{uuid4().hex}"

    @register_audio_corruption(name)
    def first(audio_or_features, severity, seed, config=None):
        del severity, seed, config
        return audio_or_features

    with pytest.raises(ValueError, match="already registered"):

        @register_audio_corruption(name)
        def second(audio_or_features, severity, seed, config=None):
            del severity, seed, config
            return audio_or_features


def test_unknown_audio_corruption_raises():
    audio = tf.zeros([16, 1], dtype=tf.float32)

    with pytest.raises(ValueError, match="Audio corruption 'missing_corruption' not found"):
        apply_audio_corruption(
            audio,
            "missing_corruption",
            severity=1,
            seed=tf.constant([1, 2], dtype=tf.int32),
        )


def test_severity_clamped_or_rejected_consistently():
    audio = tf.zeros([16, 1], dtype=tf.float32)
    seed = tf.constant([1, 2], dtype=tf.int32)

    with pytest.raises(ValueError, match="severity"):
        apply_audio_corruption(audio, "additive_white_noise", severity=0, seed=seed)

    with pytest.raises(ValueError, match="severity"):
        apply_audio_corruption(audio, "additive_white_noise", severity=6, seed=seed)


def test_audio_corruptions_registered():
    names = list_audio_corruptions()

    expected = {
        "additive_white_noise",
        "additive_pink_noise",
        "additive_brown_noise",
        "background_noise",
        "reverb_rir",
        "band_pass",
        "low_pass",
        "high_pass",
        "equalization_tilt",
        "clipping",
        "dynamic_range_compression",
        "codec_compression",
        "resampling_degradation",
        "time_stretch",
        "pitch_shift",
        "packet_dropout_gaps",
        "device_ir",
    }
    assert expected.issubset(set(names))
    assert names == tuple(sorted(names))

import pytest
import tensorflow as tf

from justdata.acoustic.configs import FeatureNormConfig
from justdata.acoustic.normalization import apply_feature_normalization


def test_per_clip_mean_std_zero_mean():
    x = tf.reshape(tf.range(24, dtype=tf.float32), [4, 3, 2])

    result = apply_feature_normalization(
        x,
        FeatureNormConfig(
            kind="per_clip_mean_std",
            axes=("time", "frequency", "channel"),
        ),
    )

    assert abs(float(tf.reduce_mean(result))) < 1e-6


def test_per_frequency_mean_std_zeroes_time_axis():
    x = tf.reshape(tf.range(12, dtype=tf.float32), [3, 4, 1])

    result = apply_feature_normalization(
        x,
        FeatureNormConfig(kind="per_frequency_mean_std"),
    )

    assert float(tf.reduce_max(tf.abs(tf.reduce_mean(result, axis=0)))) < 1e-6


def test_checkpoint_mean_std_vector_length_checked():
    x = tf.ones([2, 4, 1], dtype=tf.float32)

    with pytest.raises(ValueError, match="mean length"):
        apply_feature_normalization(
            x,
            FeatureNormConfig(
                kind="checkpoint_mean_std",
                mean=(0.0, 0.0, 0.0),
                std=(1.0, 1.0, 1.0),
            ),
        )

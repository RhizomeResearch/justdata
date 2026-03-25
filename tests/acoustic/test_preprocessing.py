import numpy as np
import tensorflow as tf

from justdata.acoustic.configs import AudioPreprocessConfig
from justdata.acoustic.configs import SegmentStrategyConfig
from justdata.acoustic.pipelines import classification_pipeline
from justdata.acoustic.preprocessing import make_preprocessing
from justdata.acoustic.resampling import resample_waveform


def test_preprocessing_is_deterministic_same_input_same_output():
    config = AudioPreprocessConfig(
        target_sample_rate=4,
        resampler="identity",
        channel_strategy="mono_mean",
        remove_dc_offset=True,
        normalize_waveform="peak",
    )
    preprocess = make_preprocessing(config)
    sample = {
        "waveform": tf.constant([[0.0, 0.2], [0.4, 0.6], [0.8, 1.0]], dtype=tf.float32),
        "sample_rate": tf.constant(4, dtype=tf.int32),
    }

    first = preprocess(sample)
    second = preprocess(sample)

    np.testing.assert_allclose(first["waveform"].numpy(), second["waveform"].numpy())
    assert first["sample_rate"].numpy() == second["sample_rate"].numpy()


def test_preprocessing_does_not_random_crop():
    config = AudioPreprocessConfig(
        target_sample_rate=4,
        resampler="identity",
        channel_strategy="keep",
    )
    preprocess = make_preprocessing(config)
    sample = {
        "waveform": tf.reshape(tf.range(8, dtype=tf.float32), [8, 1]),
        "sample_rate": tf.constant(4, dtype=tf.int32),
    }

    result = preprocess(sample)

    assert result["waveform"].shape == (8, 1)
    np.testing.assert_allclose(result["waveform"].numpy()[:, 0], np.arange(8))


def test_resample_identity_preserves_values():
    audio = tf.constant([[0.0], [0.5], [1.0]], dtype=tf.float32)

    result = resample_waveform(audio, 16000, 16000, method="identity")

    np.testing.assert_allclose(result.numpy(), audio.numpy())


def test_resample_changes_length_expected_ratio():
    audio = tf.zeros([160, 1], dtype=tf.float32)

    result = resample_waveform(audio, 16000, 8000, method="tensorflow")

    assert result.shape == (80, 1)


def test_acoustic_pipeline_accepts_structured_preprocess_and_segment_configs():
    preprocess = AudioPreprocessConfig(
        target_sample_rate=4,
        resampler="identity",
        channel_strategy="keep",
    )
    segment = SegmentStrategyConfig(
        clip_duration=1.0,
        train_mode="random_crop",
        eval_mode="center_crop",
        pad_mode="zero",
        pad_position="right",
    )

    preproc, aug, _late_aug, postproc = classification_pipeline(
        preprocess=preprocess.to_dict(),
        segment=segment.to_dict(),
        postproc_kwargs={"is_training": True},
    )
    sample = {
        "waveform": tf.reshape(tf.range(8, dtype=tf.float32), [8, 1]),
        "sample_rate": tf.constant(4, dtype=tf.int32),
    }

    sample = preproc(sample)
    train_view = aug(sample, seed=[3, 0])
    eval_passthrough = postproc(train_view)

    assert train_view["waveform"].shape == (4, 1)
    assert eval_passthrough["waveform"].shape == (4, 1)

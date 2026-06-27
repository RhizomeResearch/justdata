import numpy as np
import tensorflow as tf

from justdata.acoustic.adapters import (
    adapt_acoustic_sample,
    speech_commands_adapter,
    to_float32_waveform,
)
from justdata.core.registry import get_pipeline


def _adapted_sample():
    return adapt_acoustic_sample(
        {
            "waveform": tf.ones([8], dtype=tf.float32),
            "sample_rate": tf.constant(4, dtype=tf.int32),
            "label": tf.constant(1, dtype=tf.int64),
            "source_id": tf.constant("source-a"),
            "clip_id": tf.constant("clip-a"),
        }
    )


def test_no_default_peak_normalization():
    audio = tf.constant([0.25, 0.5], dtype=tf.float32)

    result = to_float32_waveform(audio)

    np.testing.assert_allclose(result.numpy(), [0.25, 0.5])


def test_adapter_sets_duration():
    sample = _adapted_sample()

    assert np.isclose(sample["duration"].numpy(), 2.0)


def test_adapter_sets_original_sample_rate():
    sample = _adapted_sample()

    assert sample["metadata"]["original_sample_rate"].numpy() == 4


def test_adapter_preserves_source_id():
    sample = _adapted_sample()

    assert sample["metadata"]["source_id"].numpy() == b"source-a"


def test_speech_commands_adapter_supplies_tfds_sample_rate():
    sample = speech_commands_adapter(
        {
            "audio": tf.constant([0, 16384, -16384], dtype=tf.int16),
            "label": tf.constant(2, dtype=tf.int64),
        }
    )

    assert sample["sample_rate"].numpy() == 16000
    assert sample["waveform"].dtype == tf.float32
    assert sample["waveform"].shape == [3, 1]
    assert sample["label"].numpy() == 2
    assert sample["metadata"]["dataset"] == "speech_commands"


def test_speech_commands_adapter_preserves_explicit_sample_rate():
    sample = speech_commands_adapter(
        {
            "audio": tf.ones([4], dtype=tf.int16),
            "label": tf.constant(1, dtype=tf.int64),
            "sample_rate": tf.constant(8000, dtype=tf.int32),
        }
    )

    assert sample["sample_rate"].numpy() == 8000


def test_speech_commands_adapter_preserves_sampling_rate_alias():
    sample = speech_commands_adapter(
        {
            "audio": tf.ones([4], dtype=tf.int16),
            "label": tf.constant(1, dtype=tf.int64),
            "sampling_rate": tf.constant(22050, dtype=tf.int32),
        }
    )

    assert sample["sample_rate"].numpy() == 22050


def test_dataset_registrations_resolve_acoustic_pipeline():
    for dataset in ("esc50", "speech_commands", "audioset", "dcase2025_task1"):
        pipeline = get_pipeline(dataset=dataset)
        assert pipeline.pipeline_name == "acoustic/classification"
        assert pipeline.modality == "acoustic"

from unittest.mock import patch

import numpy as np
import pytest
import tensorflow as tf

from justdata.acoustic.metadata import MetadataSidecar
from justdata.acoustic.corruptions.datasets import create_audio_corruption_datasets
from justdata.acoustic.corruptions.registry import register_audio_corruption
from justdata.acoustic.schema import FEATURES, METADATA, SAMPLE_RATE, WAVEFORM


_ECHO_SAMPLE_RATE = "test_echo_sample_rate"


@register_audio_corruption(_ECHO_SAMPLE_RATE)
def _echo_sample_rate(audio_or_features, severity, seed, config=None):
    del severity, seed
    value = -1 if config is None else config.get("sample_rate", -1)
    return tf.ones_like(audio_or_features) * tf.cast(value, tf.float32)


def _base_dataset(num_examples: int = 4) -> tf.data.Dataset:
    waveform = np.ones((num_examples, 32, 1), dtype=np.float32)
    sample_rate = np.full((num_examples,), 16000, dtype=np.int32)
    labels = np.arange(num_examples, dtype=np.int64)
    clip_ids = np.asarray([f"clip-{i}" for i in range(num_examples)])
    ds = tf.data.Dataset.from_tensor_slices(
        {
            WAVEFORM: waveform,
            "sample_rate": sample_rate,
            "label": labels,
            METADATA: {
                "example_id": np.arange(num_examples, dtype=np.int64),
                "clip_id": clip_ids,
                "quality": np.arange(num_examples, dtype=np.float32),
            },
        }
    )
    return ds.apply(tf.data.experimental.assert_cardinality(num_examples))


def _waveform_dataset(waveforms, sample_rates=None) -> tf.data.Dataset:
    samples = {WAVEFORM: np.asarray(waveforms, dtype=np.float32)}
    if sample_rates is not None:
        samples[SAMPLE_RATE] = np.asarray(sample_rates, dtype=np.int32)
    return tf.data.Dataset.from_tensor_slices(samples)


def _postprocess(sample, num_classes=None):
    del num_classes
    return sample


def test_create_audio_corruption_datasets_adds_metadata():
    with patch(
        "justdata.core.loader.fetch_ds",
        return_value=_base_dataset(),
    ):
        datasets, n_batches = create_audio_corruption_datasets(
            corruption_types=["additive_white_noise"],
            severity=2,
            base_dataset="mock_audio",
            preset=None,
            split="validation",
            seed=7,
            batch_size=2,
            preprocess_fn=lambda x: x,
            augment_fn=lambda x, **kwargs: x,
            late_augment_fn=lambda x, **kwargs: x,
            postprocess_fn=_postprocess,
            cache_dataset=False,
        )

    assert n_batches == 2
    batch = next(iter(datasets[0]))
    assert batch[METADATA]["corruption"].numpy()[0] == b"additive_white_noise"
    assert batch[METADATA]["corruption_domain"].numpy()[0] == b"waveform"
    np.testing.assert_array_equal(batch[METADATA]["severity"].numpy(), [2, 2])


def test_create_audio_corruption_datasets_cardinality_matches_base():
    with patch(
        "justdata.core.loader.fetch_ds",
        return_value=_base_dataset(6),
    ):
        datasets, n_batches = create_audio_corruption_datasets(
            corruption_types=["additive_white_noise", "clipping"],
            severity=1,
            base_dataset="mock_audio",
            preset=None,
            split="validation",
            seed=7,
            batch_size=3,
            preprocess_fn=lambda x: x,
            augment_fn=lambda x, **kwargs: x,
            late_augment_fn=lambda x, **kwargs: x,
            postprocess_fn=_postprocess,
            cache_dataset=False,
        )

    assert len(datasets) == 2
    assert n_batches == 2
    assert int(tf.data.Dataset.cardinality(datasets[0]).numpy()) == 2
    assert int(tf.data.Dataset.cardinality(datasets[1]).numpy()) == 2


def test_audio_corruption_finalizer_applies_metadata_sidecar_and_padding(tmp_path):
    sidecar_path = tmp_path / "metadata.jsonl"
    with patch(
        "justdata.core.loader.fetch_ds",
        return_value=_base_dataset(3),
    ):
        dataset, _n = create_audio_corruption_datasets(
            corruption_types="identity",
            severity=1,
            base_dataset="mock_audio",
            preset=None,
            split="validation",
            seed=7,
            batch_size=2,
            preprocess_fn=lambda x: x,
            augment_fn=lambda x, **kwargs: x,
            late_augment_fn=lambda x, **kwargs: x,
            postprocess_fn=_postprocess,
            cache_dataset=False,
            metadata_mode="numeric_only",
            sidecar_metadata_path=str(sidecar_path),
        )

    final = list(dataset)[-1]
    assert "clip_id" not in final[METADATA]
    assert "corruption" not in final[METADATA]
    assert "quality" in final[METADATA]
    np.testing.assert_array_equal(final["padding_mask"].numpy(), [True, False])
    sidecar = MetadataSidecar.read_jsonl(str(sidecar_path))
    assert sidecar.records[0]["clip_id"] == "clip-0"
    assert sidecar.records[0]["corruption"] == "identity"


def test_spectrogram_corruption_runs_after_one_frontend_pass():
    calls = {"count": 0}

    def frontend(sample, num_classes=None):
        del num_classes

        def count(value):
            calls["count"] += 1
            return value

        features = tf.py_function(count, [sample[WAVEFORM] * 2], Tout=tf.float32)
        features.set_shape(sample[WAVEFORM].shape)
        return sample | {FEATURES: features}

    with patch(
        "justdata.core.loader.fetch_ds",
        return_value=_base_dataset(2),
    ):
        dataset, _n = create_audio_corruption_datasets(
            corruption_types="clipping",
            severity=1,
            base_dataset="mock_audio",
            preset=None,
            split="validation",
            seed=7,
            corruption_domain="spectrogram",
            batch_size=2,
            preprocess_fn=lambda x: x,
            augment_fn=lambda x, **kwargs: x,
            late_augment_fn=lambda x, **kwargs: x,
            postprocess_fn=frontend,
            cache_dataset=False,
        )

    batch = next(iter(dataset))
    np.testing.assert_allclose(batch[FEATURES].numpy(), 0.95)
    assert calls["count"] == 2


def test_waveform_corruption_uses_each_samples_rate_without_mutating_config():
    config = {"custom_option": "kept"}
    dataset = _waveform_dataset(
        np.ones((2, 8, 1)),
        sample_rates=[16000, 44100],
    )

    with patch("justdata.core.loader.fetch_ds", return_value=dataset):
        corrupted, _n = create_audio_corruption_datasets(
            corruption_types=_ECHO_SAMPLE_RATE,
            severity=1,
            base_dataset="mock_audio",
            preset=None,
            batch_size=2,
            preprocess_fn=lambda x: x,
            augment_fn=lambda x, **kwargs: x,
            late_augment_fn=lambda x, **kwargs: x,
            postprocess_fn=_postprocess,
            cache_dataset=False,
            corruption_configs={_ECHO_SAMPLE_RATE: config},
        )

    batch = next(iter(corrupted))
    np.testing.assert_array_equal(batch[WAVEFORM].numpy()[:, 0, 0], [16000, 44100])
    assert config == {"custom_option": "kept"}


def test_waveform_corruption_preserves_explicit_sample_rate_override():
    config = {"sample_rate": 8000}
    dataset = _waveform_dataset(np.ones((1, 8, 1)), sample_rates=[16000])

    with patch("justdata.core.loader.fetch_ds", return_value=dataset):
        corrupted, _n = create_audio_corruption_datasets(
            corruption_types=_ECHO_SAMPLE_RATE,
            severity=1,
            base_dataset="mock_audio",
            preset=None,
            batch_size=1,
            preprocess_fn=lambda x: x,
            augment_fn=lambda x, **kwargs: x,
            late_augment_fn=lambda x, **kwargs: x,
            postprocess_fn=_postprocess,
            cache_dataset=False,
            corruption_configs={_ECHO_SAMPLE_RATE: config},
        )

    batch = next(iter(corrupted))
    np.testing.assert_array_equal(batch[WAVEFORM].numpy()[:, 0, 0], [8000])
    assert config == {"sample_rate": 8000}


def test_waveform_corruption_requires_sample_rate():
    dataset = _waveform_dataset(np.ones((1, 8, 1)))

    with patch("justdata.core.loader.fetch_ds", return_value=dataset):
        with pytest.raises(ValueError, match="waveform.*sample key 'sample_rate'"):
            create_audio_corruption_datasets(
                corruption_types="identity",
                severity=1,
                base_dataset="mock_audio",
                preset=None,
                batch_size=1,
                preprocess_fn=lambda x: x,
                augment_fn=lambda x, **kwargs: x,
                late_augment_fn=lambda x, **kwargs: x,
                postprocess_fn=_postprocess,
                cache_dataset=False,
            )


@pytest.mark.parametrize("sample_rate", [16000, 32000])
def test_dataset_low_pass_uses_physical_frequency_axis(sample_rate):
    sample_count = 4096
    time = np.arange(sample_count, dtype=np.float32) / sample_rate
    waveform = np.sin(2.0 * np.pi * 3000.0 * time)[None, :, None]
    dataset = _waveform_dataset(waveform, sample_rates=[sample_rate])

    with patch("justdata.core.loader.fetch_ds", return_value=dataset):
        corrupted, _n = create_audio_corruption_datasets(
            corruption_types="low_pass",
            severity=4,
            base_dataset="mock_audio",
            preset=None,
            batch_size=1,
            preprocess_fn=lambda x: x,
            augment_fn=lambda x, **kwargs: x,
            late_augment_fn=lambda x, **kwargs: x,
            postprocess_fn=_postprocess,
            cache_dataset=False,
        )

    output = next(iter(corrupted))[WAVEFORM]
    assert float(tf.sqrt(tf.reduce_mean(tf.square(output))).numpy()) > 0.6


def test_dataset_packet_gap_uses_44100_hz_duration():
    dataset = _waveform_dataset(np.ones((1, 4410, 1)), sample_rates=[44100])

    with patch("justdata.core.loader.fetch_ds", return_value=dataset):
        corrupted, _n = create_audio_corruption_datasets(
            corruption_types="packet_dropout_gaps",
            severity=1,
            base_dataset="mock_audio",
            preset=None,
            seed=7,
            batch_size=1,
            preprocess_fn=lambda x: x,
            augment_fn=lambda x, **kwargs: x,
            late_augment_fn=lambda x, **kwargs: x,
            postprocess_fn=_postprocess,
            cache_dataset=False,
            corruption_configs={
                "packet_dropout_gaps": {"probability": 0.0, "gap_ms": 10.0}
            },
        )

    output = next(iter(corrupted))[WAVEFORM]
    zero_count = tf.reduce_sum(tf.cast(tf.equal(output, 0.0), tf.int32))
    assert int(zero_count.numpy()) == 441


def test_spectrogram_corruption_does_not_receive_waveform_sample_rate():
    def frontend(sample, num_classes=None):
        del num_classes
        return sample | {FEATURES: sample[WAVEFORM]}

    dataset = _waveform_dataset(np.ones((1, 8, 1)), sample_rates=[16000])
    with patch("justdata.core.loader.fetch_ds", return_value=dataset):
        corrupted, _n = create_audio_corruption_datasets(
            corruption_types=_ECHO_SAMPLE_RATE,
            severity=1,
            base_dataset="mock_audio",
            preset=None,
            corruption_domain="spectrogram",
            batch_size=1,
            preprocess_fn=lambda x: x,
            augment_fn=lambda x, **kwargs: x,
            late_augment_fn=lambda x, **kwargs: x,
            postprocess_fn=frontend,
            cache_dataset=False,
        )

    output = next(iter(corrupted))[FEATURES]
    np.testing.assert_array_equal(output.numpy()[:, 0, 0], [-1])

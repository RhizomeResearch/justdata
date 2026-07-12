from unittest.mock import patch

import numpy as np
import tensorflow as tf

from justdata.acoustic.metadata import MetadataSidecar
from justdata.acoustic.corruptions.datasets import create_audio_corruption_datasets
from justdata.acoustic.schema import FEATURES, METADATA, WAVEFORM


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

    assert int(n_batches.numpy()) == 2
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
    assert int(n_batches.numpy()) == 2
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

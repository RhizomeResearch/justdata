from unittest.mock import patch

import numpy as np
import tensorflow as tf

from justdata.acoustic.corruptions.datasets import create_audio_corruption_datasets
from justdata.acoustic.schema import METADATA, WAVEFORM


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
            METADATA: {"clip_id": clip_ids},
        }
    )
    return ds.apply(tf.data.experimental.assert_cardinality(num_examples))


def _postprocess(sample, num_classes=None):
    del num_classes
    return sample


def test_create_audio_corruption_datasets_adds_metadata():
    with patch(
        "justdata.acoustic.corruptions.datasets.load_ds",
        return_value=(_base_dataset(), {"postprocess_fn": _postprocess}),
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
        "justdata.acoustic.corruptions.datasets.load_ds",
        return_value=(_base_dataset(6), {"postprocess_fn": _postprocess}),
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
            postprocess_fn=_postprocess,
            cache_dataset=False,
        )

    assert len(datasets) == 2
    assert int(n_batches.numpy()) == 2
    assert int(tf.data.Dataset.cardinality(datasets[0]).numpy()) == 2
    assert int(tf.data.Dataset.cardinality(datasets[1]).numpy()) == 2

import wave
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf


def _write_wav(path: Path, data, sample_rate: int = 16000) -> Path:
    samples = np.asarray(data, dtype=np.int16)
    if samples.ndim == 1:
        channels = 1
    elif samples.ndim == 2:
        channels = samples.shape[1]
    else:
        raise ValueError("WAV fixture data must be rank 1 or 2")

    with wave.open(str(path), "wb") as f:
        f.setnchannels(channels)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(samples.tobytes())
    return path


@pytest.fixture
def write_wav_file():
    return _write_wav


@pytest.fixture
def make_synthetic_acoustic_ds():
    """Build a tiny canonical acoustic dataset with one padded final batch."""

    def factory(*, sample_rate: int, num_examples: int = 3):
        waveform = np.linspace(-0.75, 0.75, sample_rate, dtype=np.float32)
        waveforms = np.tile(waveform[None, :, None], (num_examples, 1, 1))
        samples = {
            "waveform": waveforms,
            "sample_rate": np.full(num_examples, sample_rate, dtype=np.int32),
            "label": np.arange(num_examples, dtype=np.int64),
            "metadata": {
                "device_id": np.arange(num_examples, dtype=np.int32),
                "device_name": np.asarray([f"mic-{i}" for i in range(num_examples)]),
            },
        }
        return tf.data.Dataset.from_tensor_slices(samples).apply(
            tf.data.experimental.assert_cardinality(num_examples)
        )

    return factory


def pytest_collection_modifyitems(items):
    acoustic = pytest.mark.acoustic
    for item in items:
        item.add_marker(acoustic)

import os
import wave
from io import BytesIO

import numpy as np
import pytest
import tensorflow as tf

from justdata.acoustic.sources import _as_waveform_np, load_huggingface_audio_splits


class FakeHFDataset:
    def __init__(self, records):
        self.records = records
        self.column_names = list(records[0]) if records else []
        self.cast_calls = []

    def __iter__(self):
        return iter(self.records)

    def __len__(self):
        return len(self.records)

    def cast_column(self, column, feature):
        self.cast_calls.append((column, feature))
        return self


class FakeAudioDecoder:
    def __init__(self, array, sampling_rate):
        self.array = array
        self.sampling_rate = sampling_rate

    def __getitem__(self, key):
        if key == "array":
            return self.array
        if key == "sampling_rate":
            return self.sampling_rate
        raise KeyError(key)


@pytest.mark.parametrize(
    "dtype", [np.int16, np.int32, np.uint8, np.float32, np.float64]
)
def test_waveform_numpy_conversion_keeps_values_and_ownership(dtype):
    from justdata.acoustic.adapters import (
        standardize_waveform_layout,
        to_float32_waveform,
    )

    original = np.arange(24, dtype=dtype).reshape(2, 12)
    value = FakeAudioDecoder(original, 16000)
    result, sample_rate, path = _as_waveform_np(
        value, decode_mode="hf_native", fallback_sample_rate=None
    )
    tensor = tf.convert_to_tensor(original)
    expected = (
        standardize_waveform_layout(
            to_float32_waveform(tensor, input_dtype=tensor.dtype), layout_hint="ct"
        )
        .numpy()
        .astype(np.float32)
    )
    assert result.dtype == np.float32
    assert result.shape == (12, 2)
    assert result.tobytes() == expected.tobytes()
    assert sample_rate == 16000
    assert path is None
    result[0, 0] = -99.0
    assert original[0, 0] == 0


def _wav_bytes(samples, sample_rate=8000):
    output = BytesIO()
    pcm = (np.asarray(samples) * 32767).astype("<i2")
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return output.getvalue()


def test_hf_audio_loader_uses_audio_column(monkeypatch, tmp_path):
    datasets = pytest.importorskip("datasets")
    calls = []
    fake = FakeHFDataset(
        [
            {
                "sound": {
                    "array": np.array([0.0, 0.5, -0.5], dtype=np.float32),
                    "sampling_rate": 8000,
                },
                "label": 3,
            }
        ]
    )

    def load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return fake

    monkeypatch.setattr(datasets, "load_dataset", load_dataset)

    ds = load_huggingface_audio_splits(
        "hf_audio:unit/audio",
        ["train"],
        data_dir=tmp_path,
        audio_column="sound",
    )[0]
    sample = next(iter(ds))

    assert calls[0][1]["cache_dir"] == os.fspath(tmp_path / "hf" / "acoustic")
    assert sample["waveform"].shape == (3, 1)
    assert sample["sample_rate"].numpy() == 8000
    assert sample["label"].numpy() == 3


def test_hf_audio_loader_accepts_decoder_object(monkeypatch):
    datasets = pytest.importorskip("datasets")
    channels_first = np.stack(
        [
            np.linspace(-1.0, 1.0, 12, dtype=np.float32),
            np.linspace(1.0, -1.0, 12, dtype=np.float32),
        ]
    )
    fake = FakeHFDataset(
        [{"audio": FakeAudioDecoder(channels_first, 16000), "label": 1}]
    )
    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: fake)

    ds = load_huggingface_audio_splits("hf_audio:unit/audio", ["train"])[0]
    sample = next(iter(ds))

    np.testing.assert_allclose(sample["waveform"].numpy(), channels_first.T)
    assert sample["waveform"].dtype.name == "float32"
    assert sample["sample_rate"].numpy() == 16000


@pytest.mark.parametrize("source", ["path", "bytes"])
def test_hf_audio_loader_justdata_decodes_without_torchcodec(
    monkeypatch, tmp_path, source
):
    datasets = pytest.importorskip("datasets")
    contents = _wav_bytes([0.0, 0.5, -0.5])
    path = tmp_path / "sample.wav"
    path.write_bytes(contents)
    audio = {
        "path": os.fspath(path) if source == "path" else None,
        "bytes": contents if source == "bytes" else None,
    }
    fake = FakeHFDataset([{"audio": audio, "label": 2}])
    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: fake)

    ds = load_huggingface_audio_splits(
        "hf_audio:unit/audio", ["train"], decode_mode="justdata"
    )[0]
    sample = next(iter(ds))

    assert fake.cast_calls
    assert fake.cast_calls[0][0] == "audio"
    assert fake.cast_calls[0][1].decode is False
    assert sample["waveform"].shape == (3, 1)
    assert sample["sample_rate"].numpy() == 8000


def test_hf_audio_loader_missing_sampling_rate_raises(monkeypatch):
    datasets = pytest.importorskip("datasets")
    fake = FakeHFDataset([{"audio": np.zeros(4, dtype=np.float32), "label": 0}])
    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: fake)

    ds = load_huggingface_audio_splits("hf_audio:unit/audio", ["train"])[0]

    with pytest.raises(
        tf.errors.InvalidArgumentError, match="must provide a sampling_rate"
    ):
        next(iter(ds))


def test_hf_audio_loader_missing_audio_column_raises(monkeypatch):
    datasets = pytest.importorskip("datasets")
    fake = FakeHFDataset(
        [
            {
                "other": {
                    "array": np.array([0.0], dtype=np.float32),
                    "sampling_rate": 8000,
                },
                "label": 0,
            }
        ]
    )
    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: fake)

    with pytest.raises(ValueError, match="must contain audio column"):
        load_huggingface_audio_splits(
            "hf_audio:unit/audio",
            ["train"],
            audio_column="sound",
        )


def test_hf_audio_loader_casts_requested_sampling_rate(monkeypatch):
    datasets = pytest.importorskip("datasets")
    fake = FakeHFDataset(
        [
            {
                "sound": {
                    "array": np.array([0.0], dtype=np.float32),
                    "sampling_rate": 16000,
                },
                "label": 0,
            }
        ]
    )
    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: fake)

    load_huggingface_audio_splits(
        "hf_audio:unit/audio",
        ["train"],
        audio_column="sound",
        hf_audio_sampling_rate=16000,
    )

    assert fake.cast_calls
    assert fake.cast_calls[0][0] == "sound"

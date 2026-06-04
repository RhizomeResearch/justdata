import os

import numpy as np
import pytest

from justdata.acoustic.sources import load_huggingface_audio_splits


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

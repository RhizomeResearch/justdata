import os
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

import justdata.vision  # noqa: F401
import justdata.core.sources as core_sources
from justdata.vision.sources import load_huggingface_vision_splits


class FakeHFVisionDataset:
    column_names = ["image", "label"]

    def __len__(self):
        return 1

    def iter(self, batch_size):
        yield {
            "image": [np.zeros((4, 4, 3), dtype=np.uint8)],
            "label": [0],
        }


def test_source_cache_dir_uses_uniform_default_root():
    assert core_sources.source_cache_dir(None, "zenodo", "123") == (
        Path.home() / ".cache" / "justdata" / "zenodo" / "123"
    )


def test_source_cache_dir_appends_parts_to_explicit_root(tmp_path):
    assert core_sources.source_cache_dir(tmp_path, "hf", "vision") == (
        tmp_path / "hf" / "vision"
    )


def test_tfds_source_uses_namespaced_cache_dir(monkeypatch, tmp_path):
    calls = []

    def load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return tf.data.Dataset.range(1)

    monkeypatch.setattr(core_sources.tfds, "load", load_dataset)

    core_sources.load_tfds_splits("mnist", ["train"], data_dir=tmp_path)

    assert calls[0][1]["data_dir"] == os.fspath(tmp_path / "tfds")


def test_hf_vision_source_uses_namespaced_cache_dir(monkeypatch, tmp_path):
    datasets = pytest.importorskip("datasets")
    calls = []

    def load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return FakeHFVisionDataset()

    monkeypatch.setattr(datasets, "load_dataset", load_dataset)

    ds = load_huggingface_vision_splits(
        "hf:unit/vision",
        ["train"],
        data_dir=tmp_path,
    )[0]

    assert calls[0][1]["cache_dir"] == os.fspath(tmp_path / "hf" / "vision")
    assert next(iter(ds))["image"].shape == (4, 4, 3)

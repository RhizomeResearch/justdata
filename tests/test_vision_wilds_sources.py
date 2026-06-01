import sys
import types

import numpy as np
import pytest

import justdata.vision  # noqa: F401
from justdata.core.loader import load_ds
from justdata.core.registry import get_pipeline, get_task_for_dataset
from justdata.vision.presets import get_dataset_presets
from justdata.vision.sources import load_wilds_vision_splits


class FakeWILDSSubset:
    metadata_fields = ("year", "region", "y", "from_source_domain")

    def __init__(self, split, *, unlabeled=False):
        self.split = split
        self.unlabeled = unlabeled
        self.indices = np.array([10, 12, 14], dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def __getitem__(self, idx):
        image = np.full((8, 8, 3), idx + 1, dtype=np.uint8)
        metadata = np.array([2016 + idx, 2, idx % 2, 1], dtype=np.int64)
        if self.unlabeled:
            return image, metadata
        return image, np.int64(idx % 2), metadata


class FakeWILDSDataset:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.version = kwargs.get("version", "1.0")
        self.split_scheme = kwargs.get("split_scheme", "official")

    def get_subset(self, split, transform=None):
        assert transform is None
        return FakeWILDSSubset(split, unlabeled=split.endswith("_unlabeled"))


def _install_fake_wilds(monkeypatch):
    calls = []

    def get_dataset(**kwargs):
        calls.append(kwargs)
        return FakeWILDSDataset(**kwargs)

    monkeypatch.setitem(
        sys.modules,
        "wilds",
        types.SimpleNamespace(get_dataset=get_dataset),
    )
    return calls


def test_wilds_source_parses_options_and_preserves_metadata(monkeypatch):
    calls = _install_fake_wilds(monkeypatch)

    ds = load_wilds_vision_splits(
        "wilds:fmow?split_scheme=time_after_2016&download=true&version=1.1",
        ["train"],
        data_dir="/tmp/wilds-data",
    )[0]
    sample = next(iter(ds))

    assert calls == [
        {
            "dataset": "fmow",
            "download": True,
            "split_scheme": "time_after_2016",
            "root_dir": "/tmp/wilds-data",
            "version": "1.1",
        }
    ]
    assert sample["image"].shape == (8, 8, 3)
    assert sample["label"].numpy() == 0
    assert sample["metadata"]["dataset"].numpy() == b"fmow"
    assert sample["metadata"]["split"].numpy() == b"train"
    assert sample["metadata"]["wilds_index"].numpy() == 10
    assert sample["metadata"]["wilds"]["year"].numpy() == 2016
    assert sample["metadata"]["wilds"]["from_source_domain"].numpy() == 1


def test_wilds_source_loads_unlabeled_splits_separately(monkeypatch):
    calls = _install_fake_wilds(monkeypatch)

    ds = load_wilds_vision_splits(
        "wilds:camelyon17",
        ["test_unlabeled"],
    )[0]
    sample = next(iter(ds))

    assert calls[0]["unlabeled"] is True
    assert "label" not in sample
    assert sample["metadata"]["split"].numpy() == b"test_unlabeled"


def test_wilds_source_rejects_mixed_labeled_and_unlabeled_splits(monkeypatch):
    _install_fake_wilds(monkeypatch)

    with pytest.raises(ValueError, match="separate load_ds calls"):
        load_wilds_vision_splits("wilds:camelyon17", ["train", "test_unlabeled"])


def test_wilds_source_rejects_unsupported_modalities():
    with pytest.raises(ValueError, match="object detection"):
        load_wilds_vision_splits("wilds:globalwheat", ["train"])


def test_wilds_registry_and_presets_support_query_names():
    dataset = "wilds:fmow?split_scheme=time_after_2016"

    assert get_task_for_dataset(dataset) == "classification"
    pipeline = get_pipeline(dataset=dataset)

    assert pipeline.pipeline_name == "vision/classification"
    assert pipeline.kwargs["postproc_kwargs"]["image_size"] == 224
    assert get_dataset_presets(dataset)["laug_kwargs"]["enable"] is False


def test_wilds_rxrx1_uses_per_image_256px_preset():
    pipeline = get_pipeline(dataset="wilds:rxrx1")
    presets = get_dataset_presets("wilds:rxrx1")

    assert pipeline.kwargs["postproc_kwargs"]["image_size"] == 256
    assert pipeline.kwargs["postproc_kwargs"]["normalization_mode"] == "per_image"
    assert presets["aug_kwargs"]["crop_type"] == "random_rot90_hflip"


def test_wilds_strong_presets_are_explicit_opt_ins():
    base = get_pipeline(dataset="wilds:fmow")
    strong = get_pipeline(dataset="wilds:fmow", preset="wilds:fmow_strong")

    assert base.kwargs["aug_kwargs"]["enable"] is False
    assert strong.kwargs["aug_kwargs"]["enable"] is True
    assert strong.kwargs["aug_kwargs"]["crop_type"] == "resize_random_hflip"


def test_load_ds_accepts_wilds_dataset_string_as_split_key(monkeypatch):
    _install_fake_wilds(monkeypatch)
    dataset = "wilds:camelyon17?split_scheme=official"
    pipeline = get_pipeline(dataset=dataset)

    ds, n = load_ds(
        dataset_names_arg=[dataset],
        splits_arg={dataset: ["train"]},
        dataset_type="validation",
        batch_size=2,
        seed=0,
        pipeline=pipeline,
        cache_dataset=False,
        metadata_mode="numeric_only",
    )
    batch = next(iter(ds))

    assert n == 2
    assert batch["image"].shape == (2, 3, 96, 96)
    np.testing.assert_array_equal(batch["label"].numpy(), [0, 1])
    np.testing.assert_array_equal(
        batch["metadata"]["wilds"]["year"].numpy(),
        [2016, 2017],
    )

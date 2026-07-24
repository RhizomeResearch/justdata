import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

import justdata.vision  # noqa: F401
from justdata.core.loader import fetch_ds, load_ds
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


class FakeFMoWSourceSubset(FakeWILDSSubset):
    def __init__(self, dataset, split):
        self.dataset = dataset
        self.split = split
        self.unlabeled = False
        self.indices = np.asarray(dataset.split_indices[split], dtype=np.int64)

    def __getitem__(self, idx):
        wilds_index = int(self.indices[idx])
        image = np.full((8, 8, 3), wilds_index + 1, dtype=np.uint8)
        metadata = np.array(
            [2010 + wilds_index, 2, wilds_index % 2, 1],
            dtype=np.int64,
        )
        return image, np.int64(wilds_index % 2), metadata


class FakeFMoWSourceDataset(FakeWILDSDataset):
    def __init__(self, **kwargs):
        import pandas as pd

        super().__init__(**kwargs)
        self.metadata = pd.DataFrame(
            {
                "img_path": [
                    "seq/airport/airport_seq/airport_seq_0_rgb.jpg",
                    (
                        "train/ground_transportation_station/"
                        "ground_transportation_station_000000000000123/"
                        "ground_transportation_station_000000000000123_0_rgb.jpg"
                    ),
                    ("val/airport/airport_0007/airport_0007_4_rgb.jpg"),
                    "unused/oil_or_gas_facility/unused/unused_0_rgb.jpg",
                    (
                        "train/airport/airport_000000000000000000001/"
                        "airport_000000000000000000001_2_rgb.jpg"
                    ),
                    (
                        "test/oil_or_gas_facility/oil_or_gas_facility_42/"
                        "oil_or_gas_facility_42_8_rgb.jpg"
                    ),
                ],
                "timestamp": [
                    "2001-01-01T00:00:00Z",
                    "2011-02-07T02:48:56+05:30",
                    "2013-04-05T06:07:08.123456-04:00",
                    "2004-01-01T00:00:00Z",
                    "2010-01-02T03:04:05Z",
                    "2012-03-04T05:06:07.890Z",
                ],
                "category": [
                    "airport",
                    "ground_transportation_station",
                    "airport",
                    "oil_or_gas_facility",
                    "airport",
                    "oil_or_gas_facility",
                ],
            }
        )
        self.full_idxs = np.array([4, 1, 5, 2], dtype=np.int64)
        self.public_size = len(self.full_idxs)
        self.split_indices = {
            "train": [2, 0, 3, 1],
            "val": [1, 3, 0],
            "id_val": [3, 2],
            "id_test": [0, 1],
        }

    def __len__(self):
        return self.public_size

    def get_subset(self, split, transform=None):
        assert transform is None
        return FakeFMoWSourceSubset(self, split)


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


def _install_fake_fmow_source(monkeypatch, dataset=None):
    calls = []
    dataset = dataset or FakeFMoWSourceDataset()

    def get_dataset(**kwargs):
        calls.append(kwargs)
        dataset.kwargs = kwargs
        dataset.version = kwargs.get("version", "1.0")
        dataset.split_scheme = kwargs.get("split_scheme", "official")
        return dataset

    monkeypatch.setitem(
        sys.modules,
        "wilds",
        types.SimpleNamespace(get_dataset=get_dataset),
    )
    return dataset, calls


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
            "root_dir": "/tmp/wilds-data/wilds",
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
    assert "wilds_source" not in sample["metadata"]


def test_wilds_fmow_source_metadata_uses_authoritative_index_mapping(monkeypatch):
    _install_fake_fmow_source(monkeypatch)
    selector = (
        "wilds:fmow?split_scheme=official&version=1.1&download=false"
        "&source_metadata=timestamp,location_id"
    )

    train, val = load_wilds_vision_splits(selector, ["train", "val"])
    expected = {
        0: (
            b"airport_000000000000000000001",
            b"2010-01-02T03:04:05Z",
        ),
        1: (
            b"ground_transportation_station_000000000000123",
            b"2011-02-07T02:48:56+05:30",
        ),
        2: (
            b"oil_or_gas_facility_42",
            b"2012-03-04T05:06:07.890Z",
        ),
        3: (
            b"airport_0007",
            b"2013-04-05T06:07:08.123456-04:00",
        ),
    }

    for dataset in (train, val):
        for sample in dataset.as_numpy_iterator():
            wilds_index = int(sample["metadata"]["wilds_index"])
            source = sample["metadata"]["wilds_source"]
            assert (source["location_id"], source["timestamp"]) == expected[wilds_index]

    train_by_index = {
        int(sample["metadata"]["wilds_index"]): sample["metadata"]["wilds_source"]
        for sample in train.as_numpy_iterator()
    }
    val_by_index = {
        int(sample["metadata"]["wilds_index"]): sample["metadata"]["wilds_source"]
        for sample in val.as_numpy_iterator()
    }
    for wilds_index in set(train_by_index) & set(val_by_index):
        assert train_by_index[wilds_index] == val_by_index[wilds_index]


def test_wilds_fmow_source_metadata_propagates_through_fetch_ds(monkeypatch):
    _dataset, calls = _install_fake_fmow_source(monkeypatch)
    selector = (
        "wilds:fmow?split_scheme=official&version=1.1&download=false"
        "&source_metadata=location_id,timestamp"
    )

    ds = fetch_ds([selector], {selector: ["train"]})
    sample = next(ds.as_numpy_iterator())

    assert calls == [
        {
            "dataset": "fmow",
            "download": False,
            "split_scheme": "official",
            "root_dir": os.fspath(Path.home() / ".cache" / "justdata" / "wilds"),
            "version": "1.1",
        }
    ]
    assert sample["metadata"]["wilds_index"] == 2
    assert sample["metadata"]["wilds_source"] == {
        "location_id": b"oil_or_gas_facility_42",
        "timestamp": b"2012-03-04T05:06:07.890Z",
    }


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("unknown", "Unsupported WILDS FMoW source metadata"),
        ("y", "Target-equivalent"),
        ("category", "Target-equivalent"),
    ],
)
def test_wilds_fmow_source_metadata_rejects_disallowed_fields(field, message):
    with pytest.raises(ValueError, match=message):
        load_wilds_vision_splits(
            f"wilds:fmow?source_metadata={field}",
            ["train"],
        )


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("", "non-empty field names"),
        ("location_id,", "non-empty field names"),
        ("location_id,location_id", "Duplicate"),
    ],
)
def test_wilds_fmow_source_metadata_rejects_invalid_lists(value, message):
    with pytest.raises(ValueError, match=message):
        load_wilds_vision_splits(
            f"wilds:fmow?source_metadata={value}",
            ["train"],
        )


def test_wilds_source_metadata_is_fmow_only():
    with pytest.raises(ValueError, match="only for the 'fmow' dataset"):
        load_wilds_vision_splits(
            "wilds:camelyon17?source_metadata=location_id",
            ["train"],
        )


@pytest.mark.parametrize(
    ("mapping", "message"),
    [
        (np.array([4.0, 1.0, 5.0, 2.0]), "one-dimensional integer"),
        (np.array([4, 1, 5], dtype=np.int64), "length does not match"),
        (np.array([4, 1, 4, 2], dtype=np.int64), "unique raw metadata row"),
        (np.array([4, 1, 99, 2], dtype=np.int64), "outside the valid range"),
    ],
)
def test_wilds_fmow_source_metadata_rejects_unreliable_mapping(
    monkeypatch,
    mapping,
    message,
):
    dataset = FakeFMoWSourceDataset()
    dataset.full_idxs = mapping
    _install_fake_fmow_source(monkeypatch, dataset)

    with pytest.raises(ValueError, match=message):
        load_wilds_vision_splits(
            "wilds:fmow?source_metadata=location_id",
            ["train"],
        )


def test_wilds_fmow_source_metadata_requires_authoritative_mapping(monkeypatch):
    dataset = FakeFMoWSourceDataset()
    del dataset.full_idxs
    _install_fake_fmow_source(monkeypatch, dataset)

    with pytest.raises(ValueError, match="dataset.full_idxs"):
        load_wilds_vision_splits(
            "wilds:fmow?source_metadata=location_id",
            ["train"],
        )


@pytest.mark.parametrize(
    ("field", "column", "value", "message"),
    [
        ("location_id", "img_path", "image.jpg", "sequence directory"),
        (
            "timestamp",
            "timestamp",
            "2010-01-02T03:04:05",
            "timezone offset is required",
        ),
    ],
)
def test_wilds_fmow_source_metadata_rejects_invalid_values(
    monkeypatch,
    field,
    column,
    value,
    message,
):
    dataset = FakeFMoWSourceDataset()
    dataset.metadata.loc[4, column] = value
    _install_fake_fmow_source(monkeypatch, dataset)

    with pytest.raises(ValueError, match=message):
        load_wilds_vision_splits(
            f"wilds:fmow?source_metadata={field}",
            ["train"],
        )


def test_wilds_fmow_source_metadata_rejects_missing_columns(monkeypatch):
    dataset = FakeFMoWSourceDataset()
    dataset.metadata = dataset.metadata.drop(columns=["timestamp"])
    _install_fake_fmow_source(monkeypatch, dataset)

    with pytest.raises(ValueError, match="missing column.*timestamp"):
        load_wilds_vision_splits(
            "wilds:fmow?source_metadata=timestamp",
            ["train"],
        )


def test_wilds_fmow_source_handles_mixed_iso8601_timestamps(monkeypatch):
    import pandas as pd

    original_to_datetime = pd.to_datetime
    parsed_formats = []

    def strict_to_datetime(arg, *args, **kwargs):
        if getattr(arg, "name", None) == "timestamp":
            parsed_formats.append(kwargs.get("format"))
            if kwargs.get("format") != "ISO8601":
                raise ValueError(
                    'time data "2011-02-07T02:48:56.643Z" does not match format'
                )
        return original_to_datetime(arg, *args, **kwargs)

    def get_dataset(**kwargs):
        timestamps = pd.Series(
            ["2011-01-01T00:00:00Z", "2011-02-07T02:48:56.643Z"],
            name="timestamp",
        )
        pd.to_datetime(timestamps)
        return FakeWILDSDataset(**kwargs)

    monkeypatch.setattr(pd, "to_datetime", strict_to_datetime)
    monkeypatch.setitem(
        sys.modules,
        "wilds",
        types.SimpleNamespace(get_dataset=get_dataset),
    )

    ds = load_wilds_vision_splits(
        "wilds:fmow?split_scheme=time_after_2016",
        ["train"],
    )[0]
    sample = next(iter(ds))

    assert parsed_formats == [None, "ISO8601"]
    assert sample["metadata"]["dataset"].numpy() == b"fmow"


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


@pytest.mark.parametrize("metadata_mode", ["full", "numeric_only"])
def test_load_ds_applies_metadata_mode_to_wilds_source_metadata(
    monkeypatch,
    metadata_mode,
):
    _install_fake_fmow_source(monkeypatch)
    dataset = "wilds:fmow?source_metadata=location_id,timestamp"
    pipeline = get_pipeline(dataset=dataset)

    ds, _n = load_ds(
        dataset_names_arg=[dataset],
        splits_arg={dataset: ["train"]},
        dataset_type="validation",
        batch_size=2,
        seed=0,
        pipeline=pipeline,
        cache_dataset=False,
        metadata_mode=metadata_mode,
    )
    batch = next(iter(ds))

    if metadata_mode == "numeric_only":
        assert "wilds_source" not in batch["metadata"]
    else:
        np.testing.assert_array_equal(
            batch["metadata"]["wilds_source"]["location_id"].numpy(),
            [
                b"oil_or_gas_facility_42",
                b"airport_000000000000000000001",
            ],
        )
        np.testing.assert_array_equal(
            batch["metadata"]["wilds_source"]["timestamp"].numpy(),
            [
                b"2012-03-04T05:06:07.890Z",
                b"2010-01-02T03:04:05Z",
            ],
        )

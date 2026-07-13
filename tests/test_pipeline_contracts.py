from dataclasses import dataclass
from unittest.mock import patch

import numpy as np
import pytest
import tensorflow as tf

import justdata.vision  # noqa: F401
from justdata.core import get_pipeline, load_ds
from justdata.vision import sources as vision_sources


@dataclass(frozen=True)
class VisionPipelineContract:
    dataset: str
    task: str
    dataset_type: str
    image_shape: tuple[int, ...]
    label_shape: tuple[int, ...]


VISION_PIPELINE_CONTRACTS = (
    VisionPipelineContract(
        "cifar10", "classification", "validation", (3, 32, 32), (10,)
    ),
    VisionPipelineContract("cifar10", "classification", "train", (3, 32, 32), (10,)),
    VisionPipelineContract("kitti_road", "segmentation", "validation", (3, 32, 32), ()),
    VisionPipelineContract("kitti_road", "segmentation", "train", (3, 32, 32), ()),
)


def _load_contract(contract, raw_ds, *, seed=17, as_numpy=False):
    pipeline = get_pipeline(
        dataset=contract.dataset,
        aug_kwargs={"image_size": 32},
        postproc_kwargs={"image_size": 32, "val_resize_size": None},
    )
    with patch("justdata.core.loader.fetch_ds", return_value=raw_ds):
        return load_ds(
            dataset_names_arg="synthetic",
            splits_arg=contract.dataset_type,
            dataset_type=contract.dataset_type,
            batch_size=2,
            seed=seed,
            num_classes=10,
            pipeline=pipeline,
            cache_dataset=False,
            deterministic=True,
            shuffle_buffer=1,
            metadata_mode="numeric_only",
            as_numpy=as_numpy,
        )


@pytest.mark.parametrize(
    "contract",
    VISION_PIPELINE_CONTRACTS,
    ids=lambda case: f"vision-{case.dataset}-{case.dataset_type}",
)
def test_vision_pipeline_contract_end_to_end(contract, make_synthetic_vision_ds):
    raw_ds = make_synthetic_vision_ds(task=contract.task)
    ds, cardinality = _load_contract(contract, raw_ds)
    batches = list(ds)

    assert cardinality == 2
    assert len(batches) == 2
    assert batches[0]["image"].shape == (2, *contract.image_shape)
    assert batches[0]["image"].dtype == tf.float32
    assert batches[0]["label"].shape == (2, *contract.label_shape)
    assert "camera_id" in batches[0]["metadata"]
    assert "camera_name" not in batches[0]["metadata"]
    np.testing.assert_array_equal(batches[-1]["padding_mask"], [True, False])
    if contract.task == "segmentation":
        assert batches[0]["mask"].shape == (2, 32, 32)
        assert batches[0]["mask"].dtype.is_integer


@pytest.mark.parametrize("dataset_type", ["validation", "train"])
def test_vision_pipeline_contract_is_seeded_and_supports_numpy(
    dataset_type, make_synthetic_vision_ds
):
    contract = next(
        case
        for case in VISION_PIPELINE_CONTRACTS
        if case.dataset == "cifar10" and case.dataset_type == dataset_type
    )

    def snapshot():
        raw_ds = make_synthetic_vision_ds(task="classification")
        iterator, _ = _load_contract(contract, raw_ds, seed=23, as_numpy=True)
        return list(iterator)

    first = snapshot()
    second = snapshot()
    assert isinstance(first[0]["image"], np.ndarray)
    for left, right in zip(first, second):
        np.testing.assert_allclose(left["image"], right["image"])
        np.testing.assert_allclose(left["label"], right["label"])


def test_zenodo_asymmetric_splits_keep_archive_wide_labels(tmp_path, monkeypatch):
    root = tmp_path / "extracted"
    for split, class_names in {"train": ("cat", "dog"), "validation": ("dog",)}.items():
        for class_name in class_names:
            path = root / split / class_name / f"{class_name}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            tf.io.write_file(str(path), tf.io.encode_png(tf.ones([2, 3, 3], tf.uint8)))

    monkeypatch.setattr(vision_sources, "_prepare_zenodo_archive", lambda *args: root)
    train, validation = vision_sources.load_zenodo_imagefolder_splits(
        "zenodo:123?file=images.zip", ["train", "validation"]
    )

    assert [int(sample["label"].numpy()) for sample in train] == [0, 1]
    assert [int(sample["label"].numpy()) for sample in validation] == [1]

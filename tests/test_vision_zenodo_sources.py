import hashlib
import zipfile

import numpy as np
import pytest
import tensorflow as tf

import justdata.vision  # noqa: F401
from justdata.core.loader import load_ds
from justdata.core.registry import get_pipeline, get_task_for_dataset
from justdata.vision import sources as vision_sources


DATASET = "zenodo:123?file=tiny.zip"


def _png_bytes(value: int) -> bytes:
    image = tf.fill([4, 4, 3], tf.constant(value, dtype=tf.uint8))
    return bytes(tf.io.encode_png(image).numpy())


def _write_tiny_imagefolder_zip(tmp_path):
    archive_path = tmp_path / "tiny.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for split in ("train", "val"):
            for index, class_name in enumerate(("class_a", "class_b")):
                archive.writestr(
                    f"tiny-root/{split}/{class_name}/{split}_{class_name}.png",
                    _png_bytes(index + 1),
                )

    checksum = hashlib.md5(archive_path.read_bytes()).hexdigest()
    return archive_path, checksum


def _install_tiny_zenodo_record(monkeypatch, archive_path, checksum):
    def fetch_record(record_id):
        assert record_id == "123"
        return {
            "files": [
                {
                    "key": "tiny.zip",
                    "checksum": f"md5:{checksum}",
                    "links": {"content": archive_path.as_uri()},
                }
            ]
        }

    monkeypatch.setattr(vision_sources, "_fetch_zenodo_record", fetch_record)


def test_zenodo_source_parses_generic_imagefolder_spec():
    spec = vision_sources._parse_zenodo_spec(DATASET)

    assert spec.record_id == "123"
    assert spec.filename == "tiny.zip"

    with pytest.raises(ValueError, match="requires"):
        vision_sources._parse_zenodo_spec("zenodo:123")

    with pytest.raises(ValueError, match="Unsupported Zenodo option"):
        vision_sources._parse_zenodo_spec("zenodo:123?file=tiny.zip&root=data")


def test_zenodo_prefix_resolves_to_vision_classification_pipeline():
    assert get_task_for_dataset(DATASET) == "classification"

    pipeline = get_pipeline(dataset=DATASET, apply_presets=False)

    assert pipeline.pipeline_name == "vision/classification"


def test_zenodo_source_reports_missing_requested_file(monkeypatch, tmp_path):
    monkeypatch.setattr(
        vision_sources,
        "_fetch_zenodo_record",
        lambda record_id: {"files": [{"key": "other.zip"}]},
    )

    with pytest.raises(ValueError, match="does not contain file"):
        vision_sources.load_zenodo_imagefolder_splits(
            DATASET,
            ["train"],
            data_dir=tmp_path / "cache",
        )


def test_zenodo_imagefolder_source_loads_archive_splits(monkeypatch, tmp_path):
    archive_path, checksum = _write_tiny_imagefolder_zip(tmp_path)
    _install_tiny_zenodo_record(monkeypatch, archive_path, checksum)

    train_ds, val_ds = vision_sources.load_zenodo_imagefolder_splits(
        DATASET,
        ["train", "val"],
        data_dir=tmp_path / "cache",
    )

    assert (tmp_path / "cache" / "zenodo" / "123" / "files" / "tiny.zip").is_file()
    assert (
        tmp_path
        / "cache"
        / "zenodo"
        / "123"
        / "extracted"
        / "tiny"
        / "tiny-root"
    ).is_dir()
    assert int(tf.data.Dataset.cardinality(train_ds).numpy()) == 2
    assert int(tf.data.Dataset.cardinality(val_ds).numpy()) == 2

    samples = list(train_ds.as_numpy_iterator())

    assert [int(sample["label"]) for sample in samples] == [0, 1]
    assert samples[0]["image"].shape == (4, 4, 3)
    assert samples[0]["image"].dtype == np.uint8
    assert samples[0]["metadata"]["dataset"] == DATASET.encode()
    assert samples[0]["metadata"]["record_id"] == b"123"
    assert samples[0]["metadata"]["archive"] == b"tiny.zip"
    assert samples[0]["metadata"]["split"] == b"train"
    assert samples[0]["metadata"]["class_name"] == b"class_a"
    assert samples[0]["metadata"]["class_index"] == 0
    assert samples[0]["metadata"]["example_id"] == (
        b"train/class_a/train_class_a.png"
    )


def test_load_ds_accepts_zenodo_imagefolder_source(monkeypatch, tmp_path):
    archive_path, checksum = _write_tiny_imagefolder_zip(tmp_path)
    _install_tiny_zenodo_record(monkeypatch, archive_path, checksum)
    pipeline = get_pipeline(
        dataset=DATASET,
        apply_presets=False,
        aug_kwargs={"image_size": 8, "enable": False},
        laug_kwargs={"enable": False},
        postproc_kwargs={"image_size": 8, "normalize_image": False},
    )

    ds, n = load_ds(
        dataset_names_arg=[DATASET],
        splits_arg={DATASET: ["val"]},
        dataset_type="validation",
        batch_size=2,
        seed=0,
        pipeline=pipeline,
        num_classes=2,
        cache_dataset=False,
        metadata_mode="numeric_only",
        data_dir=tmp_path / "cache",
    )
    batch = next(iter(ds))

    assert int(n.numpy()) == 1
    assert batch["image"].shape == (2, 3, 8, 8)
    np.testing.assert_array_equal(batch["label"].numpy(), [0, 1])
    np.testing.assert_array_equal(batch["metadata"]["class_index"].numpy(), [0, 1])
    np.testing.assert_array_equal(batch["metadata"]["hard_label"].numpy(), [0, 1])
    np.testing.assert_array_equal(batch["padding_mask"].numpy(), [True, True])

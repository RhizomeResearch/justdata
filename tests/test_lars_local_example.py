"""Real-format archive fixtures for the local LaRS inventory example."""

import json
import runpy
import stat
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile, ZipInfo

import numpy as np
import pytest
import tensorflow as tf

from justdata.core import MetadataSidecar, open_inventory


EXAMPLE = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "examples/vision/lars_local_inventory.py")
)
build_inventory = EXAMPLE["build_inventory"]


def _archives(
    tmp_path, *, missing=None, invalid_mask=None, bad_size=False, rgb_mask=False
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    images_path = tmp_path / "images.zip"
    annotations_path = tmp_path / "annotations.zip"
    with (
        ZipFile(images_path, "w") as images,
        ZipFile(annotations_path, "w") as annotations,
    ):
        for split, stems in (
            ("train", ("train_0001", "train_0002")),
            ("val", ("val_0001",)),
        ):
            images.writestr(f"{split}/image_list.txt", "\n".join(stems) + "\n")
            scene = []
            panoptic_images = []
            panoptic_annotations = []
            for image_id, stem in enumerate(stems, 1):
                image = tf.fill([4, 6, 3], tf.constant(image_id * 30, tf.uint8))
                mask = np.asarray(
                    [[0, 1, 2, 255, 0, 1]] * 4,
                    dtype=np.uint8,
                )
                panoptic_ids = np.asarray(
                    [[65547, 3, 5, 0, 131083, 3]] + [[1, 3, 5, 0, 1, 3]] * 3,
                    dtype=np.int32,
                )
                panoptic_rgb = np.stack(
                    [
                        (panoptic_ids % 256).astype(np.uint8),
                        ((panoptic_ids // 256) % 256).astype(np.uint8),
                        ((panoptic_ids // 65536) % 256).astype(np.uint8),
                    ],
                    axis=-1,
                )
                if invalid_mask == stem:
                    mask[0, 0] = 7
                encoded_mask = (
                    np.repeat(mask[..., None], 3, axis=-1)
                    if rgb_mask and stem == "val_0001"
                    else mask[..., None]
                )
                image_name = f"{stem}.jpg"
                images.writestr(
                    f"{split}/images/{image_name}", tf.io.encode_jpeg(image).numpy()
                )
                for folder, content in (
                    ("semantic_masks", tf.io.encode_png(encoded_mask).numpy()),
                    (
                        "panoptic_masks",
                        tf.io.encode_png(panoptic_rgb).numpy(),
                    ),
                ):
                    name = f"{split}/{folder}/{stem}.png"
                    if name != missing:
                        annotations.writestr(name, content)
                scene.append(
                    {
                        "file_name": image_name,
                        "labels": {"scene_type": "river_like", "fog": False},
                    }
                )
                panoptic_images.append(
                    {
                        "id": image_id,
                        "file_name": image_name,
                        "height": 5 if bad_size and stem == "val_0001" else 4,
                        "width": 6,
                    }
                )
                panoptic_annotations.append(
                    {
                        "image_id": image_id,
                        "file_name": f"{stem}.png",
                        "segments_info": [
                            {"id": 1, "category_id": 1, "iscrowd": 0},
                            {"id": 3, "category_id": 3, "iscrowd": 0},
                            {"id": 5, "category_id": 5, "iscrowd": 0},
                            {"id": 65547, "category_id": 11, "iscrowd": 0},
                            {
                                "id": 131083,
                                "category_id": 11,
                                "iscrowd": int(split == "val"),
                            },
                        ],
                    }
                )
            annotations.writestr(
                f"{split}/image_annotations.json",
                json.dumps({"annotations": scene}),
            )
            annotations.writestr(
                f"{split}/panoptic_annotations.json",
                json.dumps(
                    {
                        "images": panoptic_images,
                        "annotations": panoptic_annotations,
                        "categories": [
                            {
                                "id": 1,
                                "name": "static obstacle",
                                "supercategory": "obstacle",
                                "isthing": 0,
                            },
                            {
                                "id": 3,
                                "name": "water",
                                "supercategory": "water",
                                "isthing": 0,
                            },
                            {
                                "id": 5,
                                "name": "sky",
                                "supercategory": "sky",
                                "isthing": 0,
                            },
                            {
                                "id": 11,
                                "name": "boat",
                                "supercategory": "obstacle",
                                "isthing": 1,
                            },
                        ],
                    }
                ),
            )
    return images_path, annotations_path


def test_example_admits_only_requested_split_and_preserves_annotations(tmp_path):
    images, annotations = _archives(tmp_path)
    output = tmp_path / "admitted"
    admitted = build_inventory(images, annotations, "val", output)
    assert admitted.report["requested_count"] == 1
    reopened = open_inventory(output / "snapshot")
    sample = next(reopened.dataset.as_numpy_iterator())
    assert sample["metadata"]["example_id"] == b"lars/v1.0.0/val/val_0001"
    assert sample["image"].shape == (4, 6, 3)
    np.testing.assert_array_equal(sample["mask"][0], [0, 1, 2, 255, 0, 1])

    sidecar = MetadataSidecar.read_jsonl(str(output / "metadata.jsonl"))
    source = next(iter(sidecar.records.values()))["source_record"]
    assert source["split"] == "val"
    assert source["metadata"]["scene_attributes"]["scene_type"] == "river_like"
    assert len(source["metadata"]["panoptic_segments"]) == 5
    assert {asset["role"] for asset in source["assets"]} == {
        "image",
        "semantic",
        "panoptic",
    }
    summary = json.loads((output / "source.json").read_text())
    assert summary["split"] == "val"
    assert summary["selected_count"] == 1
    assert summary["available_count"] == 1
    assert len(summary["images_archive_sha256"]) == 64
    assert stat.S_IMODE(output.stat().st_mode) & 0o077 == 0


def test_example_limit_follows_author_list_order(tmp_path):
    images, annotations = _archives(tmp_path)
    admitted = build_inventory(
        images, annotations, "train", tmp_path / "admitted", limit=1
    )
    assert admitted.report["retained_count"] == 1
    assert admitted.report["splits"][0]["retained_ids"] == [
        "lars/v1.0.0/train/train_0001"
    ]


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ({"missing": "val/semantic_masks/val_0001.png"}, "semantic_masks disagree"),
        ({"invalid_mask": "val_0001"}, "unknown class"),
        ({"bad_size": True}, "dimensions disagree"),
        ({"rgb_mask": True}, "8-bit grayscale"),
    ],
)
def test_example_rejects_invalid_pairs_without_publishing(tmp_path, change, expected):
    images, annotations = _archives(tmp_path, **change)
    output = tmp_path / "admitted"
    with pytest.raises(ValueError, match=expected):
        build_inventory(images, annotations, "val", output)
    assert not output.exists()


def test_example_rejects_duplicate_and_link_members(tmp_path):
    images, annotations = _archives(tmp_path)
    with pytest.warns(UserWarning, match="Duplicate name"):
        with ZipFile(images, "a") as archive:
            archive.writestr("val/images/val_0001.jpg", b"duplicate")
    with pytest.raises(ValueError, match="duplicate ZIP member"):
        build_inventory(images, annotations, "val", tmp_path / "duplicate")

    images, annotations = _archives(tmp_path / "other")
    with ZipFile(images, "a") as archive:
        link = ZipInfo("val/images/link.jpg")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(link, b"target")
    with pytest.raises(ValueError, match="Unsafe or duplicate ZIP member"):
        build_inventory(images, annotations, "val", tmp_path / "link")

    images, annotations = _archives(tmp_path / "traversal")
    with ZipFile(images, "a") as archive:
        archive.writestr("val/../unexpected.jpg", b"unexpected")
    with pytest.raises(ValueError, match="Unsafe or duplicate ZIP member"):
        build_inventory(images, annotations, "val", tmp_path / "unsafe")


def test_example_rejects_test_targets_and_existing_output(tmp_path):
    images, annotations = _archives(tmp_path)
    with pytest.raises(ValueError, match="train and val"):
        build_inventory(images, annotations, "test", tmp_path / "test")
    output = tmp_path / "present"
    output.mkdir()
    with pytest.raises(FileExistsError):
        build_inventory(images, annotations, "val", output)


def test_example_command_loads_a_realistic_validation_view(tmp_path):
    images, annotations = _archives(tmp_path)
    output = tmp_path / "command-output"
    result = subprocess.run(
        [
            sys.executable,
            str(
                Path(__file__).resolve().parents[1]
                / "examples/vision/lars_local_inventory.py"
            ),
            "--images-archive",
            str(images),
            "--annotations-archive",
            str(annotations),
            "--split",
            "val",
            "--output-dir",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    printed = json.loads(result.stdout)
    assert printed["retained"] == 1
    assert printed["first_batch_ids"] == ["lars/v1.0.0/val/val_0001"]
    assert printed["target_shape"][1] == 3
    assert printed["original_sizes"] == [[4, 6]]
    assert printed["valid_pixels"] == [20]
    assert (output / "executed_config.json").is_file()


def test_example_panoptic_command_preserves_instances_and_crowd(tmp_path):
    images, annotations = _archives(tmp_path)
    output = tmp_path / "panoptic-output"
    result = subprocess.run(
        [
            sys.executable,
            str(
                Path(__file__).resolve().parents[1]
                / "examples/vision/lars_local_inventory.py"
            ),
            "--images-archive",
            str(images),
            "--annotations-archive",
            str(annotations),
            "--split",
            "val",
            "--labels",
            "panoptic",
            "--output-dir",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    printed = json.loads(result.stdout)
    assert printed["labels"] == "panoptic"
    assert printed["present_targets"] == [4]
    assert printed["category_ids"] == [1, 3, 5, 11]
    assert printed["valid_pixels"] == [19]
    assert json.loads((output / "source.json").read_text())["max_segments"] == 5
    sample = next(open_inventory(output / "snapshot").dataset.as_numpy_iterator())
    assert sorted(np.unique(sample["panoptic_mask"]).tolist()) == [
        0,
        1,
        3,
        5,
        65547,
        131083,
    ]

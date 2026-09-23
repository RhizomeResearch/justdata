"""Admit one LaRS v1.0.0 split from local archives and inspect a dense view.

The two archives are supplied by the caller. This example reads annotated
keyframes only; LaRS temporal context lives in a separate archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import stat
from pathlib import Path, PurePosixPath, PureWindowsPath
from zipfile import ZipFile, ZipInfo

import tensorflow as tf

import justdata.vision  # noqa: F401
from justdata.core import (
    InventoryAsset,
    InventoryRecord,
    InventorySource,
    MetadataSidecar,
    admit_inventory,
    get_pipeline,
    load_inventory,
)
from justdata.vision.encodings import (
    decode_panoptic_rgb,
    panoptic_map_to_semantic,
    validate_panoptic_sample,
)


_MAX_MEMBERS = 100_000
_MAX_EXPANDED_BYTES = 100 * 1024**3
_MAX_MEMBER_BYTES = 32 * 1024**2
_MAX_PIXELS = 10_000_000
_STEM = re.compile(r"[A-Za-z0-9_-]+\Z")
_CLASSES = (0, 1, 2)
_IGNORE = 255


def _archive_members(archive: ZipFile) -> dict[str, ZipInfo]:
    """Reject ambiguous or unsafe members before reading any payload."""
    members = {}
    expanded_bytes = 0
    for member in archive.infolist():
        name = member.filename
        path = PurePosixPath(name)
        windows_path = PureWindowsPath(name)
        mode = stat.S_IFMT(member.external_attr >> 16)
        if (
            not name
            or "\\" in name
            or path.is_absolute()
            or windows_path.is_absolute()
            or windows_path.drive
            or any(part in {"", ".", ".."} for part in name.rstrip("/").split("/"))
            or mode == stat.S_IFLNK
            or mode not in (0, stat.S_IFREG, stat.S_IFDIR)
            or (mode == stat.S_IFDIR and not member.is_dir())
            or name.rstrip("/") in members
        ):
            raise ValueError(f"Unsafe or duplicate ZIP member: {name!r}")
        members[name.rstrip("/")] = member
        expanded_bytes += member.file_size
    if len(members) > _MAX_MEMBERS or expanded_bytes > _MAX_EXPANDED_BYTES:
        raise ValueError("LaRS archive exceeds the member or expansion budget")
    return members


def _read_member(archive: ZipFile, members: dict[str, ZipInfo], name: str) -> bytes:
    member = members.get(name)
    if member is None or member.is_dir():
        raise ValueError(f"Missing LaRS archive member: {name}")
    if member.file_size > _MAX_MEMBER_BYTES:
        raise ValueError(f"LaRS archive member exceeds 32 MiB: {name}")
    return archive.read(member)


def _json_member(archive: ZipFile, members: dict[str, ZipInfo], name: str) -> dict:
    value = json.loads(_read_member(archive, members, name))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {name}")
    return value


def _unique_index(rows: list[dict], key: str, expected: set[str], name: str) -> dict:
    if not isinstance(rows, list):
        raise ValueError(f"Expected a list in {name}")
    indexed = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get(key), str):
            raise ValueError(f"Invalid entry in {name}")
        if row[key] in indexed:
            raise ValueError(f"Duplicate {key} in {name}: {row[key]}")
        indexed[row[key]] = row
    if set(indexed) != expected:
        raise ValueError(f"LaRS listing and {name} disagree")
    return indexed


def _asset_names(members: dict[str, ZipInfo], prefix: str) -> set[str]:
    return {
        name
        for name, member in members.items()
        if name.startswith(prefix) and not member.is_dir()
    }


def _png_size(payload: bytes, *, grayscale: bool) -> tuple[int, int]:
    if (
        len(payload) < 29
        or payload[:8] != b"\x89PNG\r\n\x1a\n"
        or payload[12:16] != b"IHDR"
    ):
        raise ValueError("Invalid PNG header")
    if grayscale and (payload[24], payload[25]) != (8, 0):
        raise ValueError("LaRS semantic mask must be 8-bit grayscale")
    width = int.from_bytes(payload[16:20], "big")
    height = int.from_bytes(payload[20:24], "big")
    if not 0 < width * height <= _MAX_PIXELS:
        raise ValueError("LaRS PNG dimensions exceed the pixel budget")
    return height, width


def _category_contract(categories):
    if not isinstance(categories, list) or not categories:
        raise ValueError("LaRS panoptic categories must be a nonempty list")
    rows = sorted(categories, key=lambda row: row["id"])
    ids = tuple(row["id"] for row in rows)
    things = tuple(row["id"] for row in rows if row.get("isthing") == 1)
    mapping = {"obstacle": 0, "water": 1, "sky": 2}
    if (
        any(type(value) is not int or value < 0 for value in ids)
        or len(set(ids)) != len(ids)
        or any(row.get("isthing") not in (0, 1) for row in rows)
        or any(row.get("supercategory") not in mapping for row in rows)
    ):
        raise ValueError("Invalid LaRS panoptic category table")
    return ids, things, tuple(mapping[row["supercategory"]] for row in rows)


def _read_pair(
    payloads: dict[str, bytes],
    metadata: dict,
    *,
    labels="semantic",
    class_values=(),
    thing_class_values=(),
    semantic_values=(),
    max_segments=1,
) -> dict:
    expected_size = (metadata["height"], metadata["width"])
    jpeg_shape = tf.io.extract_jpeg_shape(payloads["image"]).numpy()
    if tuple(jpeg_shape[:2]) != expected_size or int(jpeg_shape[2]) not in (1, 3):
        raise ValueError("LaRS image dimensions disagree with annotations")
    if not 0 < expected_size[0] * expected_size[1] <= _MAX_PIXELS:
        raise ValueError("LaRS image dimensions exceed the pixel budget")
    if _png_size(payloads["semantic"], grayscale=True) != expected_size:
        raise ValueError("LaRS semantic mask dimensions disagree with image")
    if _png_size(payloads["panoptic"], grayscale=False) != expected_size:
        raise ValueError("LaRS panoptic mask dimensions disagree with image")

    image = tf.io.decode_jpeg(payloads["image"], channels=3)
    mask = tf.squeeze(tf.io.decode_png(payloads["semantic"], channels=1), -1)
    panoptic = tf.io.decode_png(payloads["panoptic"], channels=3)
    if tuple(panoptic.shape[:2]) != expected_size:
        raise ValueError("LaRS panoptic mask dimensions disagree with image")
    allowed = tf.constant((*_CLASSES, _IGNORE), dtype=mask.dtype)
    if not bool(tf.reduce_all(tf.reduce_any(mask[..., None] == allowed, -1))):
        raise ValueError("LaRS semantic mask contains an unknown class")
    if labels == "semantic":
        return {"image": image, "mask": mask}
    panoptic_mask = decode_panoptic_rgb(panoptic)
    rows = metadata["panoptic_segments"]
    if any(
        not isinstance(row, dict)
        or type(row.get("id")) is not int
        or type(row.get("category_id")) is not int
        or row.get("iscrowd") not in (0, 1)
        for row in rows
    ):
        raise ValueError("Invalid LaRS panoptic segment table")
    sample = validate_panoptic_sample(
        {
            "image": image,
            "panoptic_mask": panoptic_mask,
            "segments": {
                "segment_ids": tf.constant([row["id"] for row in rows], tf.int64),
                "category_ids": tf.constant(
                    [row["category_id"] for row in rows], tf.int32
                ),
                "is_crowd": tf.constant(
                    [bool(row["iscrowd"]) for row in rows], tf.bool
                ),
                "valid_mask": tf.ones([len(rows)], tf.bool),
            },
        },
        class_values=class_values,
        thing_class_values=thing_class_values,
        max_segments=max_segments,
    )
    projected = panoptic_map_to_semantic(
        panoptic_mask,
        sample["segments"],
        class_values=class_values,
        thing_class_values=thing_class_values,
        semantic_values=semantic_values,
        max_segments=max_segments,
    )
    if not bool(tf.reduce_all(projected == tf.cast(mask, tf.int32))):
        raise ValueError("LaRS panoptic and semantic annotations disagree")
    return sample


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build_inventory(
    images_archive: Path,
    annotations_archive: Path,
    split: str,
    output_dir: Path,
    *,
    limit: int | None = None,
    labels: str = "semantic",
):
    """Create one strict, self-contained snapshot for train or validation."""
    if split not in {"train", "val"}:
        raise ValueError("LaRS annotations are available for train and val")
    if labels not in {"semantic", "panoptic"}:
        raise ValueError("labels must be semantic or panoptic")
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("limit must be a positive integer")
    output_dir = Path(output_dir).absolute()
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")

    with ZipFile(images_archive) as images, ZipFile(annotations_archive) as annotations:
        image_members = _archive_members(images)
        annotation_members = _archive_members(annotations)
        stems = (
            _read_member(images, image_members, f"{split}/image_list.txt")
            .decode("utf-8")
            .splitlines()
        )
        if (
            not stems
            or len(stems) != len(set(stems))
            or any(not _STEM.fullmatch(stem) for stem in stems)
        ):
            raise ValueError("LaRS image_list.txt has invalid or duplicate IDs")
        names = {f"{stem}.jpg" for stem in stems}
        if _asset_names(image_members, f"{split}/images/") != {
            f"{split}/images/{name}" for name in names
        }:
            raise ValueError("LaRS image list and image archive disagree")
        for folder in ("semantic_masks", "panoptic_masks"):
            expected = {f"{split}/{folder}/{stem}.png" for stem in stems}
            if _asset_names(annotation_members, f"{split}/{folder}/") != expected:
                raise ValueError(f"LaRS image list and {folder} disagree")

        scene = _json_member(
            annotations, annotation_members, f"{split}/image_annotations.json"
        )
        panoptic = _json_member(
            annotations, annotation_members, f"{split}/panoptic_annotations.json"
        )
        class_values, thing_class_values, semantic_values = _category_contract(
            panoptic.get("categories")
        )
        scenes = _unique_index(
            scene.get("annotations"), "file_name", names, "scene annotations"
        )
        images_by_name = _unique_index(
            panoptic.get("images"), "file_name", names, "panoptic images"
        )
        images_by_id = {}
        for item in images_by_name.values():
            image_id = item.get("id")
            if type(image_id) is not int or image_id in images_by_id:
                raise ValueError("Invalid or duplicate LaRS panoptic image ID")
            images_by_id[image_id] = item
        annotations_by_id = {}
        for item in panoptic.get("annotations", []):
            image_id = item.get("image_id")
            if image_id in annotations_by_id:
                raise ValueError("Duplicate LaRS panoptic annotation")
            annotations_by_id[image_id] = item
        if set(annotations_by_id) != set(images_by_id):
            raise ValueError("LaRS panoptic image and annotation IDs disagree")

        selected = stems[:limit]
        max_segments = max(
            1,
            *(
                len(
                    annotations_by_id[images_by_name[f"{stem}.jpg"]["id"]][
                        "segments_info"
                    ]
                )
                for stem in selected
            ),
        )
        images_digest = _digest(Path(images_archive))
        annotations_digest = _digest(Path(annotations_archive))
        selected_digest = hashlib.sha256(
            "\n".join(selected).encode("utf-8")
        ).hexdigest()
        inventory_digest = hashlib.sha256(
            f"{images_digest}:{annotations_digest}:{split}:{selected_digest}:{labels}:v2".encode(
                "ascii"
            )
        ).hexdigest()
        expected_bytes = sum(
            images_by_name[f"{stem}.jpg"]["height"]
            * images_by_name[f"{stem}.jpg"]["width"]
            * 4
            + sum(
                archive_members[f"{split}/{folder}/{stem}.{suffix}"].file_size
                for archive_members, folder, suffix in (
                    (image_members, "images", "jpg"),
                    (annotation_members, "semantic_masks", "png"),
                    (annotation_members, "panoptic_masks", "png"),
                )
            )
            for stem in selected
        )
        if expected_bytes > shutil.disk_usage(output_dir.parent).free:
            raise ValueError("Insufficient free space for LaRS assets and snapshot")

        output_dir.mkdir(mode=0o700)
        try:
            asset_dir = output_dir / "assets"
            asset_dir.mkdir()
            records = []
            for stem in selected:
                image_name = f"{stem}.jpg"
                info = images_by_name[image_name]
                if (
                    type(info.get("height")) is not int
                    or type(info.get("width")) is not int
                    or not 0 < info["height"] * info["width"] <= _MAX_PIXELS
                ):
                    raise ValueError(f"Invalid LaRS image dimensions: {image_name}")
                annotation = annotations_by_id[info["id"]]
                if annotation.get("file_name") != f"{stem}.png" or not isinstance(
                    annotation.get("segments_info"), list
                ):
                    raise ValueError(f"Invalid LaRS panoptic annotation: {image_name}")
                assets = {}
                for role, archive, members, member_name in (
                    ("image", images, image_members, f"{split}/images/{image_name}"),
                    (
                        "semantic",
                        annotations,
                        annotation_members,
                        f"{split}/semantic_masks/{stem}.png",
                    ),
                    (
                        "panoptic",
                        annotations,
                        annotation_members,
                        f"{split}/panoptic_masks/{stem}.png",
                    ),
                ):
                    payload = _read_member(archive, members, member_name)
                    path = asset_dir / f"{role}-{stem}{Path(member_name).suffix}"
                    path.write_bytes(payload)
                    assets[role] = InventoryAsset(
                        path, hashlib.sha256(payload).hexdigest()
                    )
                records.append(
                    InventoryRecord(
                        f"lars/v1.0.0/{split}/{stem}",
                        assets,
                        {
                            "height": info["height"],
                            "width": info["width"],
                            "source_split": split,
                            "source_name": image_name,
                            "panoptic_image_id": info["id"],
                            "scene_attributes": scenes[image_name].get("labels"),
                            "panoptic_segments": annotation["segments_info"],
                        },
                    )
                )

            def read_selected(payloads, metadata):
                return _read_pair(
                    payloads,
                    metadata,
                    labels=labels,
                    class_values=class_values,
                    thing_class_values=thing_class_values,
                    semantic_values=semantic_values,
                    max_segments=max_segments,
                )

            signature = (
                {
                    "image": tf.TensorSpec([None, None, 3], tf.uint8),
                    "mask": tf.TensorSpec([None, None], tf.uint8),
                }
                if labels == "semantic"
                else {
                    "image": tf.TensorSpec([None, None, 3], tf.uint8),
                    "panoptic_mask": tf.TensorSpec([None, None], tf.int64),
                    "segments": {
                        "segment_ids": tf.TensorSpec([max_segments], tf.int64),
                        "category_ids": tf.TensorSpec([max_segments], tf.int32),
                        "is_crowd": tf.TensorSpec([max_segments], tf.bool),
                        "valid_mask": tf.TensorSpec([max_segments], tf.bool),
                    },
                }
            )
            admitted = admit_inventory(
                {"lars": InventorySource({split: records}, read_selected)},
                {"lars": [split]},
                inventory_id=f"lars/v1.0.0/{split}/{inventory_digest}",
                snapshot_dir=output_dir / "snapshot",
                output_signature=signature,
            )
            sidecar = MetadataSidecar.from_inventory(admitted)
            sidecar.write_jsonl(str(output_dir / "metadata.jsonl"), policy="create")
            source_record = {
                "release": "LaRS v1.0.0",
                "split": split,
                "selected_count": len(records),
                "available_count": len(stems),
                "images_archive_sha256": images_digest,
                "annotations_archive_sha256": annotations_digest,
                "selected_ids_sha256": selected_digest,
                "categories": panoptic.get("categories"),
                "labels": labels,
                "max_segments": max_segments,
                "semantic_values": semantic_values,
            }
            (output_dir / "source.json").write_text(
                json.dumps(source_record, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            return admitted
        except Exception:
            shutil.rmtree(output_dir)
            raise


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("limit must be positive")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-archive", type=Path, required=True)
    parser.add_argument("--annotations-archive", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=_positive_int)
    parser.add_argument(
        "--labels", choices=("semantic", "panoptic"), default="semantic"
    )
    args = parser.parse_args()

    # Configure TensorFlow before constructing any dataset. The caller may
    # separately initialise JAX on the accelerator after this point.
    tf.config.set_visible_devices([], "GPU")
    admitted = build_inventory(
        args.images_archive,
        args.annotations_archive,
        args.split,
        args.output_dir,
        limit=args.limit,
        labels=args.labels,
    )
    source_info = json.loads((args.output_dir / "source.json").read_text())
    options = (
        {
            "geometry_kwargs": {
                "class_values": _CLASSES,
                "ignore_value": _IGNORE,
                "train_crop_size": 512,
                "train_resize_range": (512, 1024),
                "eval_long_side": 1024,
                "patch_size": 16,
            },
            "keep_original_mask": args.split != "train",
            "postproc_kwargs": {
                "normalize_image": True,
                "normalization_params": (
                    (0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225),
                ),
                "permute_image": True,
                "emit_semantic_targets": True,
            },
        }
        if args.labels == "semantic"
        else {
            "geometry_kwargs": {
                "train_crop_size": 512,
                "train_resize_range": (512, 1024),
                "eval_long_side": 1024,
                "patch_size": 16,
            },
            "panoptic_kwargs": {
                "class_values": tuple(
                    row["id"]
                    for row in sorted(
                        source_info["categories"], key=lambda row: row["id"]
                    )
                ),
                "thing_class_values": tuple(
                    row["id"]
                    for row in sorted(
                        source_info["categories"], key=lambda row: row["id"]
                    )
                    if row["isthing"]
                ),
                "max_segments": source_info["max_segments"],
            },
            "keep_original_annotations": args.split != "train",
            "postproc_kwargs": {
                "normalize_image": True,
                "normalization_params": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                "permute_image": True,
                "emit_panoptic_targets": True,
            },
        }
    )
    pipeline = get_pipeline(
        "vision/segmentation"
        if args.labels == "semantic"
        else "vision/panoptic_segmentation",
        apply_presets=False,
        overrides=options,
    )
    sidecar = MetadataSidecar.read_jsonl(str(args.output_dir / "metadata.jsonl"))
    batches, count, config = load_inventory(
        admitted,
        "train" if args.split == "train" else "validation",
        2 if args.split == "train" else 1,
        17,
        pipeline=pipeline,
        metadata_mode="numeric_only",
        metadata_sidecar=sidecar,
        map_parallel_calls=1,
        private_threadpool_size=1,
        max_intra_op_parallelism=1,
        prefetch=1,
        return_config=True,
    )
    encoded = config.to_bytes()
    (args.output_dir / "executed_config.json").write_bytes(encoded)
    first = next(iter(batches))
    real = first["padding_mask"].numpy()
    row_ids = first["metadata"]["row_id"].numpy()[real]
    print(
        json.dumps(
            {
                "retained": admitted.report["retained_count"],
                "batches": count,
                "first_batch_ids": [
                    sidecar.records[int(row_id)]["source_record"]["record_id"]
                    for row_id in row_ids
                ],
                "image_shape": first["image"].shape.as_list(),
                "labels": args.labels,
                "mask_shape": first[
                    "mask" if args.labels == "semantic" else "panoptic_mask"
                ].shape.as_list(),
                "target_shape": first["targets"]["masks"].shape.as_list(),
                "category_ids": first["targets"]["class_ids"]
                .numpy()[real][0][
                    first["targets"]["target_valid_mask"].numpy()[real][0]
                ]
                .tolist(),
                "present_targets": first["targets"]["num_targets"]
                .numpy()[real]
                .tolist(),
                "valid_pixels": tf.reduce_sum(
                    tf.cast(first["pixel_valid_mask"], tf.int32), axis=(1, 2)
                )
                .numpy()[real]
                .tolist(),
                "original_sizes": first["geometry"]["original_size"]
                .numpy()[real]
                .tolist(),
                "executed_config_sha256": hashlib.sha256(encoded).hexdigest(),
                "output_dir": str(args.output_dir.absolute()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

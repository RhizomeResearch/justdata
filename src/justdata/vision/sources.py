import hashlib
import json
import os
import re
import shutil
import stat
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Union
from urllib.parse import parse_qsl, quote, urlparse

import numpy as np
import tensorflow as tf
from loguru import logger

from justdata.core.sources import register_source_loader, source_cache_dir

_ZENODO_PREFIX = "zenodo:"
_ZENODO_QUERY_PARAMS = {"file"}
_ZENODO_ORIGIN = "zenodo.org"
_ZENODO_NETWORK_TIMEOUT_SECONDS = 60
_ZENODO_DOWNLOAD_TIMEOUT_SECONDS = 60 * 60
_ZENODO_DOWNLOAD_CHUNK_BYTES = 1024 * 1024
_ZENODO_MAX_ARCHIVE_BYTES = 20 * 1024**3
_ZENODO_MAX_EXTRACTED_BYTES = 100 * 1024**3
_ZENODO_MAX_ARCHIVE_MEMBERS = 100_000
_ZENODO_MAX_IMAGE_FILE_BYTES = 256 * 1024**2
_ZENODO_MAX_IMAGE_DIMENSION = 32_768
_ZENODO_MAX_IMAGE_PIXELS = 64_000_000
_IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".webp",
}
_WILDS_PREFIX = "wilds:"
_SUPPORTED_WILDS_IMAGE_CLASSIFICATION = {
    "camelyon17",
    "fmow",
    "iwildcam",
    "rxrx1",
}
_UNSUPPORTED_WILDS_DATASETS = {
    "amazon": "text",
    "civilcomments": "text",
    "globalwheat": "object detection",
    "ogb-molpcba": "graph",
    "poverty": "regression",
    "py150": "code/text",
}
_WILDS_QUERY_PARAMS = {
    "download",
    "source_metadata",
    "split_scheme",
    "unlabeled",
    "version",
}
_FMOW_SOURCE_METADATA_FIELDS = ("location_id", "timestamp")
_FMOW_TARGET_EQUIVALENT_FIELDS = {"category", "y"}


@dataclass(frozen=True)
class _ZenodoImageFolderSpec:
    record_id: str
    filename: str


@dataclass(frozen=True)
class _WILDSDatasetSpec:
    name: str
    split_scheme: str
    version: str | None
    download: bool
    unlabeled: bool
    source_metadata: tuple[str, ...]


def _strip_prefix(dataset_name: str, prefix: str) -> str:
    if not dataset_name.startswith(prefix):
        raise ValueError(
            f"Expected dataset name to start with {prefix!r}; got {dataset_name!r}."
        )
    return dataset_name[len(prefix) :]


def _import_datasets():
    try:
        import datasets
    except ImportError as e:
        raise ImportError(
            "The 'datasets' package is required for hf: vision source loading. "
            "Install justdata with the vision extra."
        ) from e
    return datasets


def _parse_zenodo_spec(dataset_name: str) -> _ZenodoImageFolderSpec:
    raw = _strip_prefix(dataset_name, _ZENODO_PREFIX)
    record_id, sep, query = raw.partition("?")
    if not re.fullmatch(r"[1-9][0-9]*", record_id):
        raise ValueError("Zenodo record id must be a positive integer.")

    values: dict[str, str] = {}
    for key, value in parse_qsl(query if sep else "", keep_blank_values=True):
        if key not in _ZENODO_QUERY_PARAMS:
            allowed = ", ".join(sorted(_ZENODO_QUERY_PARAMS))
            raise ValueError(
                f"Unsupported Zenodo option {key!r}. Supported options: {allowed}."
            )
        if key in values:
            raise ValueError(f"Duplicate Zenodo option {key!r}.")
        values[key] = value

    filename = values.get("file", "")
    if not filename:
        raise ValueError("Zenodo source requires ?file=<archive-name>.")
    _validate_zenodo_filename(filename)

    return _ZenodoImageFolderSpec(record_id=record_id, filename=filename)


def _validate_zenodo_filename(filename: str) -> None:
    posix_path = PurePosixPath(filename)
    windows_path = PureWindowsPath(filename)
    if (
        filename in {".", ".."}
        or "\x00" in filename
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or posix_path.name != filename
        or windows_path.name != filename
        or "/" in filename
        or "\\" in filename
    ):
        raise ValueError(
            f"Zenodo archive filename must be a basename without path segments: "
            f"{filename!r}."
        )


def _zenodo_record_dir(
    spec: _ZenodoImageFolderSpec,
    data_dir: Union[None, str, os.PathLike],
) -> Path:
    return source_cache_dir(data_dir, "zenodo", spec.record_id)


def _fetch_zenodo_record(record_id: str) -> dict[str, Any]:
    url = f"https://zenodo.org/api/records/{record_id}"
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def _select_zenodo_file(record: dict[str, Any], filename: str) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError("Zenodo record metadata must be an object.")
    files = record.get("files")
    if not isinstance(files, list):
        raise ValueError("Zenodo record metadata does not contain a files list.")

    for file_info in files:
        if not isinstance(file_info, dict):
            continue
        if file_info.get("key") == filename or file_info.get("filename") == filename:
            return file_info

    available = ", ".join(
        str(file_info.get("key") or file_info.get("filename")) for file_info in files
    )
    raise ValueError(
        f"Zenodo record does not contain file {filename!r}. "
        f"Available files: {available}."
    )


def _zenodo_download_url(
    spec: _ZenodoImageFolderSpec,
    file_info: dict[str, Any],
) -> str:
    links = file_info.get("links")
    if links is None:
        links = {}
    if not isinstance(links, dict):
        raise ValueError("Zenodo file links metadata must be an object.")
    for key in ("content", "download"):
        value = links.get(key)
        if value:
            return _validate_zenodo_download_url(str(value))

    value = links.get("self")
    if value and str(value).rstrip("/").endswith("/content"):
        return _validate_zenodo_download_url(str(value))

    quoted = quote(spec.filename)
    return _validate_zenodo_download_url(
        f"https://zenodo.org/records/{spec.record_id}/files/{quoted}?download=1"
    )


def _validate_zenodo_download_url(url: str) -> str:
    parsed = urlparse(url)
    try:
        port = parsed.port
    except ValueError as e:
        raise ValueError(
            "Zenodo download link must use the https://zenodo.org origin."
        ) from e
    if (
        parsed.scheme != "https"
        or parsed.hostname != _ZENODO_ORIGIN
        or port not in (None, 443)
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError("Zenodo download link must use the https://zenodo.org origin.")
    return url


def _zenodo_declared_size(file_info: dict[str, Any]) -> int | None:
    size = file_info.get("size")
    if size is None:
        return None
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError("Zenodo file size must be a non-negative integer.")
    if size > _ZENODO_MAX_ARCHIVE_BYTES:
        raise ValueError(
            f"Zenodo file size {size} exceeds the archive byte limit "
            f"({_ZENODO_MAX_ARCHIVE_BYTES})."
        )
    return size


def _checksum_parts(checksum: str | None) -> tuple[str, str] | None:
    if not checksum:
        return None
    if not isinstance(checksum, str):
        raise ValueError("Zenodo checksum must be a string.")

    algorithm, sep, expected = checksum.partition(":")
    if not sep:
        algorithm, expected = "md5", algorithm
    algorithm = algorithm.lower()
    if algorithm not in hashlib.algorithms_available:
        raise ValueError(f"Unsupported Zenodo checksum algorithm {algorithm!r}.")
    return algorithm, expected.lower()


def _file_matches_checksum(path: Path, checksum: str | None) -> bool:
    if not path.exists():
        return False

    parts = _checksum_parts(checksum)
    if parts is None:
        return True

    algorithm, expected = parts
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().lower() == expected


def _download_file(
    url: str,
    target: Path,
    *,
    expected_size: int | None,
    checksum: str | None,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    parts = _checksum_parts(checksum)
    digest = hashlib.new(parts[0]) if parts is not None else None
    started_at = time.monotonic()
    bytes_downloaded = 0
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
            with urllib.request.urlopen(
                url, timeout=_ZENODO_NETWORK_TIMEOUT_SECONDS
            ) as response:
                while True:
                    if time.monotonic() - started_at > _ZENODO_DOWNLOAD_TIMEOUT_SECONDS:
                        raise TimeoutError(
                            "Zenodo download exceeded the transfer time limit."
                        )
                    chunk = response.read(_ZENODO_DOWNLOAD_CHUNK_BYTES)
                    if not chunk:
                        break
                    bytes_downloaded += len(chunk)
                    byte_limit = (
                        expected_size
                        if expected_size is not None
                        else _ZENODO_MAX_ARCHIVE_BYTES
                    )
                    if bytes_downloaded > byte_limit:
                        raise ValueError(
                            "Zenodo download exceeded its declared or configured "
                            "archive byte limit."
                        )
                    tmp.write(chunk)
                    if digest is not None:
                        digest.update(chunk)

            if expected_size is not None and bytes_downloaded != expected_size:
                raise ValueError(
                    f"Zenodo download size mismatch: expected {expected_size} "
                    f"bytes, received {bytes_downloaded}."
                )
            if parts is not None and digest.hexdigest().lower() != parts[1]:
                raise ValueError(
                    f"Downloaded Zenodo file {target.name!r} failed checksum "
                    "validation."
                )
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_path, target)
        tmp_path = None
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def _archive_stem(filename: str) -> str:
    path = Path(filename)
    suffixes = path.suffixes
    if len(suffixes) >= 2 and suffixes[-2:] in (
        [".tar", ".gz"],
        [".tar", ".bz2"],
        [".tar", ".xz"],
    ):
        return path.name[: -len("".join(suffixes[-2:]))]
    return path.stem


def _assert_safe_extract_path(destination: Path, member_name: str) -> None:
    path = PurePosixPath(member_name)
    windows_path = PureWindowsPath(member_name)
    if (
        not member_name
        or "\\" in member_name
        or path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"Unsafe archive member path: {member_name!r}.")
    target = (destination / member_name).resolve()
    root = destination.resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"Archive member escapes extraction directory: {member_name}")


def _check_archive_budget(
    destination: Path,
    members: list[tuple[str, int]],
) -> None:
    if len(members) > _ZENODO_MAX_ARCHIVE_MEMBERS:
        raise ValueError(
            f"Archive member count {len(members)} exceeds the limit "
            f"({_ZENODO_MAX_ARCHIVE_MEMBERS})."
        )
    expanded_bytes = sum(size for _, size in members)
    if expanded_bytes > _ZENODO_MAX_EXTRACTED_BYTES:
        raise ValueError(
            f"Archive expanded size {expanded_bytes} exceeds the byte limit "
            f"({_ZENODO_MAX_EXTRACTED_BYTES})."
        )
    free_bytes = shutil.disk_usage(destination.parent).free
    if expanded_bytes > free_bytes:
        raise ValueError(
            f"Archive requires {expanded_bytes} extracted bytes but the cache has "
            f"only {free_bytes} bytes free."
        )


def _normalized_archive_member(destination: Path, member_name: str) -> str:
    _assert_safe_extract_path(destination, member_name)
    return PurePosixPath(member_name).as_posix().rstrip("/")


def _preflight_zip(archive: zipfile.ZipFile, destination: Path) -> None:
    members: list[tuple[str, int]] = []
    destinations: set[str] = set()
    for member in archive.infolist():
        normalized = _normalized_archive_member(destination, member.filename)
        if normalized in destinations:
            raise ValueError(
                f"Archive contains duplicate member path: {member.filename!r}."
            )
        destinations.add(normalized)

        mode = member.external_attr >> 16
        file_type = stat.S_IFMT(mode)
        is_directory = member.is_dir()
        if file_type == stat.S_IFLNK:
            raise ValueError(f"Archive member uses a link: {member.filename!r}.")
        if file_type not in (0, stat.S_IFREG, stat.S_IFDIR) or (
            file_type == stat.S_IFDIR and not is_directory
        ):
            raise ValueError(
                f"Archive member is not a regular file or directory: "
                f"{member.filename!r}."
            )
        members.append((member.filename, 0 if is_directory else member.file_size))
    _check_archive_budget(destination, members)


def _preflight_tar(archive: tarfile.TarFile, destination: Path) -> None:
    members: list[tuple[str, int]] = []
    destinations: set[str] = set()
    for member in archive.getmembers():
        normalized = _normalized_archive_member(destination, member.name)
        if normalized in destinations:
            raise ValueError(
                f"Archive contains duplicate member path: {member.name!r}."
            )
        destinations.add(normalized)
        if not (member.isfile() or member.isdir()):
            category = "link" if member.issym() or member.islnk() else "special file"
            raise ValueError(f"Archive member uses a {category}: {member.name!r}.")
        members.append((member.name, member.size if member.isfile() else 0))
    _check_archive_budget(destination, members)


def _extract_archive(archive_path: Path, extract_dir: Path) -> None:
    extract_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(
        tempfile.mkdtemp(prefix=f".{extract_dir.name}.", dir=extract_dir.parent)
    )

    try:
        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path) as archive:
                _preflight_zip(archive, tmp_dir)
                archive.extractall(tmp_dir)
        elif tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path) as archive:
                _preflight_tar(archive, tmp_dir)
                if hasattr(tarfile, "data_filter"):
                    archive.extractall(tmp_dir, filter="data")
                else:
                    archive.extractall(tmp_dir)
        else:
            raise ValueError(
                f"Zenodo file {archive_path.name!r} is not a supported ZIP/TAR archive."
            )

        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        os.replace(tmp_dir, extract_dir)
    except Exception:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        raise


def _prepare_zenodo_archive(
    spec: _ZenodoImageFolderSpec,
    data_dir: Union[None, str, os.PathLike],
) -> Path:
    record = _fetch_zenodo_record(spec.record_id)
    file_info = _select_zenodo_file(record, spec.filename)
    checksum = file_info.get("checksum")
    expected_size = _zenodo_declared_size(file_info)
    download_url = _zenodo_download_url(spec, file_info)

    record_dir = _zenodo_record_dir(spec, data_dir)
    archive_path = record_dir / "files" / spec.filename
    extract_dir = record_dir / "extracted" / _archive_stem(spec.filename)
    lock_path = record_dir / "locks" / f"{spec.filename}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from filelock import FileLock
    except ImportError as e:
        raise ImportError(
            "Zenodo source loading requires the 'filelock' package. Install "
            "justdata with the vision extra."
        ) from e

    with FileLock(lock_path):
        if not _file_matches_checksum(archive_path, checksum):
            archive_path.unlink(missing_ok=True)
            logger.info(
                f"Downloading Zenodo record {spec.record_id} file {spec.filename}."
            )
            _download_file(
                download_url,
                archive_path,
                expected_size=expected_size,
                checksum=checksum,
            )
            if extract_dir.exists():
                shutil.rmtree(extract_dir)

        if not extract_dir.exists():
            logger.info(f"Extracting Zenodo archive {archive_path}.")
            _extract_archive(archive_path, extract_dir)

    return extract_dir


def _resolve_imagefolder_root(extract_dir: Path, splits: list[str]) -> Path:
    if all((extract_dir / split).is_dir() for split in splits):
        return extract_dir

    candidates = [
        child
        for child in extract_dir.iterdir()
        if child.is_dir() and all((child / split).is_dir() for split in splits)
    ]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        names = ", ".join(sorted(candidate.name for candidate in candidates))
        raise ValueError(
            f"Multiple ImageFolder roots contain requested splits: {names}."
        )

    requested = ", ".join(splits)
    raise ValueError(
        f"Zenodo archive does not contain requested ImageFolder splits: {requested}."
    )


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS


def _validate_image_file(path: Path) -> None:
    size = path.stat().st_size
    if size > _ZENODO_MAX_IMAGE_FILE_BYTES:
        raise ValueError(
            f"Zenodo image {path} exceeds the encoded-file limit of "
            f"{_ZENODO_MAX_IMAGE_FILE_BYTES} bytes."
        )

    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError as e:
        raise ImportError(
            "Pillow is required for bounded Zenodo image validation. "
            "Install justdata with the vision extra."
        ) from e

    try:
        with Image.open(path) as image:
            width, height = image.size
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError) as e:
        raise ValueError(f"Zenodo image {path} has an invalid image header.") from e

    if width <= 0 or height <= 0:
        raise ValueError(
            f"Zenodo image {path} has invalid dimensions {width}x{height}."
        )
    if width > _ZENODO_MAX_IMAGE_DIMENSION or height > _ZENODO_MAX_IMAGE_DIMENSION:
        raise ValueError(
            f"Zenodo image {path} dimensions {width}x{height} exceed the "
            f"per-dimension limit of {_ZENODO_MAX_IMAGE_DIMENSION}."
        )
    pixels = width * height
    if pixels > _ZENODO_MAX_IMAGE_PIXELS:
        raise ValueError(
            f"Zenodo image {path} has {pixels} decoded pixels, exceeding the "
            f"limit of {_ZENODO_MAX_IMAGE_PIXELS}."
        )


def _split_class_dirs(split_dir: Path) -> list[Path]:
    return sorted(
        child
        for child in split_dir.iterdir()
        if child.is_dir() and not child.name.startswith(".")
    )


def _imagefolder_split_dirs(root: Path) -> list[Path]:
    return sorted(
        child
        for child in root.iterdir()
        if child.is_dir()
        and not child.name.startswith(".")
        and _split_class_dirs(child)
    )


def _imagefolder_class_to_label(root: Path) -> dict[str, np.int64]:
    class_names = sorted(
        {
            class_dir.name
            for split_dir in _imagefolder_split_dirs(root)
            for class_dir in _split_class_dirs(split_dir)
        }
    )
    if not class_names:
        raise ValueError("Zenodo ImageFolder archive does not contain class folders.")
    return {name: np.int64(index) for index, name in enumerate(class_names)}


def _imagefolder_records(
    root: Path,
    splits: list[str],
    *,
    dataset_name: str,
    spec: _ZenodoImageFolderSpec,
    class_to_label: dict[str, np.int64],
) -> list[dict[str, Any]]:
    split_dirs = {split: root / split for split in splits}
    records: list[dict[str, Any]] = []
    for split in splits:
        split_records = []
        split_dir = split_dirs[split]
        class_dirs = _split_class_dirs(split_dir)
        unknown_classes = sorted(
            class_dir.name
            for class_dir in class_dirs
            if class_dir.name not in class_to_label
        )
        if unknown_classes:
            names = ", ".join(unknown_classes)
            raise ValueError(
                f"Zenodo ImageFolder split {split!r} contains classes absent from "
                f"the archive-wide vocabulary: {names}."
            )
        for class_dir in class_dirs:
            label = class_to_label[class_dir.name]
            for path in sorted(class_dir.rglob("*")):
                if not _is_image_file(path):
                    continue
                _validate_image_file(path)
                relative_path = path.relative_to(root).as_posix()
                split_records.append(
                    {
                        "_path": os.fspath(path),
                        "label": label,
                        "metadata": {
                            "dataset": dataset_name,
                            "record_id": spec.record_id,
                            "archive": spec.filename,
                            "split": split,
                            "class_name": class_dir.name,
                            "class_index": label,
                            "filename": path.name,
                            "path": os.fspath(path),
                            "example_id": relative_path,
                        },
                    }
                )
        if not split_records:
            raise ValueError(f"Zenodo ImageFolder split {split!r} has no images.")
        records.extend(split_records)

    return records


def _records_to_vision_dataset(records: list[dict[str, Any]]) -> tf.data.Dataset:
    def gen():
        yield from records

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature={
            "_path": tf.TensorSpec(shape=(), dtype=tf.string),
            "label": tf.TensorSpec(shape=(), dtype=tf.int64),
            "metadata": {
                "dataset": tf.TensorSpec(shape=(), dtype=tf.string),
                "record_id": tf.TensorSpec(shape=(), dtype=tf.string),
                "archive": tf.TensorSpec(shape=(), dtype=tf.string),
                "split": tf.TensorSpec(shape=(), dtype=tf.string),
                "class_name": tf.TensorSpec(shape=(), dtype=tf.string),
                "class_index": tf.TensorSpec(shape=(), dtype=tf.int64),
                "filename": tf.TensorSpec(shape=(), dtype=tf.string),
                "path": tf.TensorSpec(shape=(), dtype=tf.string),
                "example_id": tf.TensorSpec(shape=(), dtype=tf.string),
            },
        },
    )

    def decode_image(sample):
        image = tf.io.decode_image(
            tf.io.read_file(sample["_path"]),
            channels=3,
            expand_animations=False,
        )
        image.set_shape([None, None, 3])
        return {
            "image": image,
            "label": sample["label"],
            "metadata": sample["metadata"],
        }

    ds = ds.map(decode_image, num_parallel_calls=tf.data.AUTOTUNE)
    try:
        ds = ds.apply(tf.data.experimental.assert_cardinality(len(records)))
    except Exception as e:
        logger.warning(f"Failed to assert cardinality for Zenodo ImageFolder: {e}")
    return ds


@register_source_loader(_ZENODO_PREFIX)
def load_zenodo_imagefolder_splits(
    dataset_name: str,
    splits: list[str],
    data_dir: Union[None, str, os.PathLike] = None,
) -> list[tf.data.Dataset]:
    spec = _parse_zenodo_spec(dataset_name)
    extract_dir = _prepare_zenodo_archive(spec, data_dir)
    root = _resolve_imagefolder_root(extract_dir, splits)
    class_to_label = _imagefolder_class_to_label(root)
    return [
        _records_to_vision_dataset(
            _imagefolder_records(
                root,
                [split],
                dataset_name=dataset_name,
                spec=spec,
                class_to_label=class_to_label,
            )
        )
        for split in splits
    ]


@register_source_loader("hf:")
def load_huggingface_vision_splits(
    dataset_name: str,
    splits: list[str],
    data_dir: Union[None, str, os.PathLike] = None,
    *,
    include_metadata: bool = False,
) -> list[tf.data.Dataset]:
    datasets = _import_datasets()
    hf_name = dataset_name[3:]
    loaded_splits = []

    for split in splits:
        ds_hf = datasets.load_dataset(
            hf_name,
            split=split,
            cache_dir=os.fspath(source_cache_dir(data_dir, "hf", "vision")),
        )

        if "image" not in ds_hf.column_names or "label" not in ds_hf.column_names:
            raise ValueError(
                f"Hugging Face vision dataset '{hf_name}' must contain 'image' and "
                f"'label' columns. Found columns: {ds_hf.column_names}."
            )

        def gen(_ds=ds_hf, _split=split):
            example_index = 0
            for batch in _ds.iter(batch_size=1024):
                for img, lbl in zip(batch["image"], batch["label"]):
                    item = {
                        "image": np.array(img),
                        "label": lbl,
                    }
                    if include_metadata:
                        filename = getattr(img, "filename", "") or ""
                        item["metadata"] = {
                            "dataset": hf_name,
                            "split": _split,
                            "example_id": str(example_index),
                            "filename": os.path.basename(filename) if filename else "",
                        }
                    example_index += 1
                    yield item

        output_signature = {
            "image": tf.TensorSpec(shape=None, dtype=tf.uint8),
            "label": tf.TensorSpec(shape=(), dtype=tf.int64),
        }
        if include_metadata:
            output_signature["metadata"] = {
                "dataset": tf.TensorSpec(shape=(), dtype=tf.string),
                "split": tf.TensorSpec(shape=(), dtype=tf.string),
                "example_id": tf.TensorSpec(shape=(), dtype=tf.string),
                "filename": tf.TensorSpec(shape=(), dtype=tf.string),
            }

        ds_tf = tf.data.Dataset.from_generator(
            gen,
            output_signature=output_signature,
        )

        try:
            ds_tf = ds_tf.apply(tf.data.experimental.assert_cardinality(len(ds_hf)))
        except Exception as e:
            logger.warning(
                f"Failed to assert cardinality for HF dataset {hf_name}: {e}"
            )

        loaded_splits.append(ds_tf)

    return loaded_splits


def _parse_bool(value: str, *, key: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"{key} must be a boolean value; got {value!r}.")


def _parse_wilds_source_metadata(
    value: str | None,
    *,
    dataset_name: str,
) -> tuple[str, ...]:
    if value is None:
        return ()
    if dataset_name != "fmow":
        raise ValueError(
            "WILDS source_metadata is supported only for the 'fmow' dataset."
        )

    requested = tuple(field.strip() for field in value.split(","))
    if not requested or any(not field for field in requested):
        raise ValueError(
            "WILDS FMoW source_metadata must be a comma-separated list of "
            "non-empty field names."
        )

    duplicates = sorted(field for field in set(requested) if requested.count(field) > 1)
    if duplicates:
        raise ValueError(
            f"Duplicate WILDS FMoW source metadata field(s): {', '.join(duplicates)}."
        )

    target_fields = sorted(set(requested) & _FMOW_TARGET_EQUIVALENT_FIELDS)
    if target_fields:
        raise ValueError(
            "Target-equivalent WILDS FMoW source metadata field(s) are not "
            f"allowed: {', '.join(target_fields)}."
        )

    unsupported = sorted(set(requested) - set(_FMOW_SOURCE_METADATA_FIELDS))
    if unsupported:
        supported = ", ".join(_FMOW_SOURCE_METADATA_FIELDS)
        raise ValueError(
            "Unsupported WILDS FMoW source metadata field(s): "
            f"{', '.join(unsupported)}. Supported fields: {supported}."
        )

    requested_set = set(requested)
    return tuple(
        field for field in _FMOW_SOURCE_METADATA_FIELDS if field in requested_set
    )


def _parse_wilds_spec(dataset_name: str) -> _WILDSDatasetSpec:
    if not dataset_name.startswith(_WILDS_PREFIX):
        raise ValueError(
            f"Expected WILDS dataset name to start with {_WILDS_PREFIX!r}; "
            f"got {dataset_name!r}."
        )

    raw = dataset_name[len(_WILDS_PREFIX) :]
    name, sep, query = raw.partition("?")
    if not name:
        raise ValueError("WILDS dataset name cannot be empty.")

    values: dict[str, str] = {}
    for key, value in parse_qsl(query if sep else "", keep_blank_values=True):
        if key not in _WILDS_QUERY_PARAMS:
            allowed = ", ".join(sorted(_WILDS_QUERY_PARAMS))
            raise ValueError(
                f"Unsupported WILDS option {key!r}. Supported options: {allowed}."
            )
        if key in values:
            raise ValueError(f"Duplicate WILDS option {key!r}.")
        values[key] = value

    if name not in _SUPPORTED_WILDS_IMAGE_CLASSIFICATION:
        reason = _UNSUPPORTED_WILDS_DATASETS.get(name)
        if reason is not None:
            raise ValueError(
                f"WILDS dataset {name!r} is a {reason} dataset and is not "
                "supported by justdata's vision classification loader."
            )
        supported = ", ".join(sorted(_SUPPORTED_WILDS_IMAGE_CLASSIFICATION))
        raise ValueError(
            f"Unsupported WILDS dataset {name!r}. Supported image classification "
            f"datasets: {supported}."
        )

    return _WILDSDatasetSpec(
        name=name,
        split_scheme=values.get("split_scheme", "official"),
        version=values.get("version") or None,
        download=_parse_bool(values.get("download", "false"), key="download"),
        unlabeled=_parse_bool(values.get("unlabeled", "false"), key="unlabeled"),
        source_metadata=_parse_wilds_source_metadata(
            values.get("source_metadata"),
            dataset_name=name,
        ),
    )


def _is_unlabeled_wilds_split(split: str) -> bool:
    return split.endswith("_unlabeled") or split == "extra_unlabeled"


def _validate_wilds_splits(splits: list[str], spec: _WILDSDatasetSpec) -> bool:
    has_unlabeled = any(_is_unlabeled_wilds_split(split) for split in splits)
    has_labeled = any(not _is_unlabeled_wilds_split(split) for split in splits)
    if has_labeled and has_unlabeled:
        raise ValueError(
            "WILDS labeled and unlabeled splits must be loaded in separate "
            "load_ds calls."
        )
    if spec.unlabeled and has_labeled:
        raise ValueError(
            "?unlabeled=true can only be used with WILDS unlabeled splits."
        )
    return has_unlabeled


def _import_wilds():
    try:
        import wilds
    except ImportError as e:
        raise ImportError(
            "The 'wilds' package is required for wilds: source loading. "
            "Install justdata with the wilds extra."
        ) from e
    return wilds


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _image_to_uint8_array(image: Any) -> np.ndarray:
    if hasattr(image, "convert"):
        image = image.convert("RGB")
    array = _as_numpy(image)
    if array.ndim == 3 and array.shape[0] in (1, 3) and array.shape[-1] > 4:
        array = np.transpose(array, (1, 2, 0))
    if array.dtype == np.uint8:
        return array
    if np.issubdtype(array.dtype, np.floating) and array.size:
        max_value = float(np.nanmax(array))
        if max_value <= 1.0:
            array = array * 255.0
    return np.clip(array, 0, 255).astype(np.uint8)


def _label_to_int64(label: Any) -> np.int64:
    label_array = _as_numpy(label)
    if label_array.size != 1:
        raise ValueError("WILDS vision classification labels must be scalar values.")
    return np.int64(label_array.reshape(()))


def _metadata_to_int64_dict(
    metadata: Any,
    metadata_fields: tuple[str, ...] | list[str],
) -> dict[str, np.int64]:
    if isinstance(metadata, dict):
        return {str(key): _label_to_int64(value) for key, value in metadata.items()}

    metadata_array = _as_numpy(metadata).reshape(-1)
    result = {}
    for i, field in enumerate(metadata_fields):
        if i >= metadata_array.shape[0]:
            break
        result[str(field)] = np.int64(metadata_array[i])
    return result


def _fmow_location_id(value: Any, *, public_index: int, raw_index: int) -> str:
    if not isinstance(value, (str, np.str_)) or not value:
        raise ValueError(
            "WILDS FMoW source metadata field 'location_id' is unavailable "
            f"for public index {public_index} (raw row {raw_index}): "
            "'img_path' must be a non-empty string."
        )

    text = str(value)
    posix_path = PurePosixPath(text)
    windows_path = PureWindowsPath(text)
    if len(posix_path.parts) >= 2 and posix_path.parent.name:
        return posix_path.parent.name
    if len(windows_path.parts) >= 2 and windows_path.parent.name:
        return windows_path.parent.name
    raise ValueError(
        "WILDS FMoW source metadata field 'location_id' is unavailable "
        f"for public index {public_index} (raw row {raw_index}): "
        "'img_path' does not contain a sequence directory."
    )


def _fmow_timestamp(value: Any, *, public_index: int, raw_index: int) -> str:
    if not isinstance(value, (str, np.str_)) or not value:
        raise ValueError(
            "WILDS FMoW source metadata field 'timestamp' is unavailable "
            f"for public index {public_index} (raw row {raw_index}): "
            "the source value must be a non-empty string."
        )

    text = str(value)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as e:
        raise ValueError(
            "WILDS FMoW source metadata field 'timestamp' is invalid "
            f"for public index {public_index} (raw row {raw_index}): "
            "expected an ISO-8601 timestamp."
        ) from e
    if parsed.utcoffset() is None:
        raise ValueError(
            "WILDS FMoW source metadata field 'timestamp' is invalid "
            f"for public index {public_index} (raw row {raw_index}): "
            "an explicit UTC designator or timezone offset is required."
        )
    return text


def _prepare_fmow_source_metadata(
    dataset,
    requested_fields: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    if not requested_fields:
        return {}

    metadata = getattr(dataset, "metadata", None)
    if (
        metadata is None
        or not hasattr(metadata, "iloc")
        or not hasattr(metadata, "columns")
    ):
        raise ValueError(
            "WILDS FMoW source metadata requires the authoritative raw "
            "metadata table exposed as dataset.metadata."
        )

    full_idxs = getattr(dataset, "full_idxs", None)
    if full_idxs is None:
        raise ValueError(
            "WILDS FMoW source metadata requires the authoritative public-to-raw "
            "dataset.full_idxs mapping."
        )
    public_to_raw = _as_numpy(full_idxs)
    if public_to_raw.ndim != 1 or not np.issubdtype(public_to_raw.dtype, np.integer):
        raise ValueError(
            "WILDS FMoW dataset.full_idxs must be a one-dimensional integer "
            "public-to-raw index mapping."
        )

    try:
        public_size = len(dataset)
        raw_size = len(metadata)
    except TypeError as e:
        raise ValueError(
            "WILDS FMoW source metadata requires sized public and raw datasets."
        ) from e
    if len(public_to_raw) != public_size:
        raise ValueError(
            "WILDS FMoW dataset.full_idxs length does not match the public "
            f"dataset size ({len(public_to_raw)} != {public_size})."
        )
    if len(np.unique(public_to_raw)) != len(public_to_raw):
        raise ValueError(
            "WILDS FMoW dataset.full_idxs must map each public index to a "
            "unique raw metadata row."
        )
    if public_to_raw.size and (
        int(public_to_raw.min()) < 0 or int(public_to_raw.max()) >= raw_size
    ):
        raise ValueError(
            "WILDS FMoW dataset.full_idxs contains a raw metadata row outside "
            f"the valid range [0, {raw_size})."
        )

    required_columns = {
        "location_id": "img_path",
        "timestamp": "timestamp",
    }
    columns = set(metadata.columns)
    missing_columns = sorted(
        required_columns[field]
        for field in requested_fields
        if required_columns[field] not in columns
    )
    if missing_columns:
        raise ValueError(
            "WILDS FMoW source metadata is unavailable because the raw metadata "
            f"table is missing column(s): {', '.join(missing_columns)}."
        )

    raw_indices = public_to_raw.astype(np.int64, copy=False)
    rows = metadata.iloc[raw_indices]
    prepared: dict[str, tuple[str, ...]] = {}
    if "location_id" in requested_fields:
        prepared["location_id"] = tuple(
            _fmow_location_id(
                value,
                public_index=public_index,
                raw_index=int(raw_indices[public_index]),
            )
            for public_index, value in enumerate(rows["img_path"].tolist())
        )
    if "timestamp" in requested_fields:
        prepared["timestamp"] = tuple(
            _fmow_timestamp(
                value,
                public_index=public_index,
                raw_index=int(raw_indices[public_index]),
            )
            for public_index, value in enumerate(rows["timestamp"].tolist())
        )
    return prepared


def _is_fmow_timestamp_arg(value: Any) -> bool:
    return getattr(value, "name", None) == "timestamp"


@contextmanager
def _wilds_fmow_datetime_compat(enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return

    try:
        import pandas as pd
    except ImportError:
        yield
        return

    original_to_datetime = pd.to_datetime

    def to_datetime_compat(arg, *args, **kwargs):
        if kwargs.get("format") is not None or not _is_fmow_timestamp_arg(arg):
            return original_to_datetime(arg, *args, **kwargs)

        try:
            return original_to_datetime(arg, *args, **kwargs)
        except ValueError as e:
            if "match format" not in str(e):
                raise
            compat_kwargs = dict(kwargs)
            compat_kwargs["format"] = "ISO8601"
            return original_to_datetime(arg, *args, **compat_kwargs)

    pd.to_datetime = to_datetime_compat
    try:
        yield
    finally:
        pd.to_datetime = original_to_datetime


def _split_to_dataset(
    dataset,
    split: str,
    *,
    spec: _WILDSDatasetSpec,
    include_label: bool,
    source_metadata: dict[str, tuple[str, ...]],
) -> tf.data.Dataset:
    subset = dataset.get_subset(split, transform=None)
    metadata_fields = tuple(getattr(subset, "metadata_fields", ()))
    version = str(getattr(dataset, "version", spec.version or ""))
    split_scheme = str(getattr(dataset, "split_scheme", spec.split_scheme))
    indices = getattr(subset, "indices", None)

    def gen():
        for subset_index, item in enumerate(subset):
            if include_label:
                image, label, metadata = item
            else:
                if len(item) == 2:
                    image, metadata = item
                elif len(item) == 3:
                    image, _label, metadata = item
                else:
                    raise ValueError("WILDS unlabeled samples must be 2- or 3-tuples.")

            if indices is None:
                wilds_index = subset_index
            else:
                wilds_index = int(_as_numpy(indices[subset_index]).reshape(()))

            sample = {
                "image": _image_to_uint8_array(image),
                "metadata": {
                    "dataset": spec.name,
                    "split": split,
                    "example_id": str(wilds_index),
                    "wilds_index": np.int64(wilds_index),
                    "split_scheme": split_scheme,
                    "version": version,
                    "wilds": _metadata_to_int64_dict(metadata, metadata_fields),
                },
            }
            if source_metadata:
                try:
                    sample["metadata"]["wilds_source"] = {
                        field: source_metadata[field][wilds_index]
                        for field in spec.source_metadata
                    }
                except IndexError as e:
                    raise ValueError(
                        f"WILDS subset index {wilds_index} is outside the "
                        "authoritative FMoW source metadata mapping."
                    ) from e
            if include_label:
                sample["label"] = _label_to_int64(label)
            yield sample

    output_signature = {
        "image": tf.TensorSpec(shape=None, dtype=tf.uint8),
        "metadata": {
            "dataset": tf.TensorSpec(shape=(), dtype=tf.string),
            "split": tf.TensorSpec(shape=(), dtype=tf.string),
            "example_id": tf.TensorSpec(shape=(), dtype=tf.string),
            "wilds_index": tf.TensorSpec(shape=(), dtype=tf.int64),
            "split_scheme": tf.TensorSpec(shape=(), dtype=tf.string),
            "version": tf.TensorSpec(shape=(), dtype=tf.string),
            "wilds": {
                field: tf.TensorSpec(shape=(), dtype=tf.int64)
                for field in metadata_fields
            },
        },
    }
    if source_metadata:
        output_signature["metadata"]["wilds_source"] = {
            field: tf.TensorSpec(shape=(), dtype=tf.string)
            for field in spec.source_metadata
        }
    if include_label:
        output_signature["label"] = tf.TensorSpec(shape=(), dtype=tf.int64)

    ds_tf = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    try:
        ds_tf = ds_tf.apply(tf.data.experimental.assert_cardinality(len(subset)))
    except Exception as e:
        logger.warning(
            f"Failed to assert cardinality for WILDS dataset {spec.name}: {e}"
        )
    return ds_tf


@register_source_loader(_WILDS_PREFIX)
def load_wilds_vision_splits(
    dataset_name: str,
    splits: list[str],
    data_dir: Union[None, str, os.PathLike] = None,
) -> list[tf.data.Dataset]:
    spec = _parse_wilds_spec(dataset_name)
    uses_unlabeled = _validate_wilds_splits(splits, spec)
    wilds = _import_wilds()

    kwargs = {
        "dataset": spec.name,
        "download": spec.download,
        "split_scheme": spec.split_scheme,
    }
    kwargs["root_dir"] = os.fspath(source_cache_dir(data_dir, "wilds"))
    if spec.version is not None:
        kwargs["version"] = spec.version
    if spec.unlabeled or uses_unlabeled:
        kwargs["unlabeled"] = True

    with _wilds_fmow_datetime_compat(spec.name == "fmow"):
        dataset = wilds.get_dataset(**kwargs)
    include_label = not (spec.unlabeled or uses_unlabeled)
    source_metadata = _prepare_fmow_source_metadata(
        dataset,
        spec.source_metadata,
    )
    return [
        _split_to_dataset(
            dataset,
            split,
            spec=spec,
            include_label=include_label,
            source_metadata=source_metadata,
        )
        for split in splits
    ]

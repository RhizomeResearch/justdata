import concurrent.futures
import hashlib
import io
import shutil
import stat
import tarfile
import time
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
        split_classes = {
            "train": ("class_a", "class_b", "class_c"),
            "val": ("class_b", "class_c"),
        }
        for split, class_names in split_classes.items():
            for index, class_name in enumerate(class_names):
                archive.writestr(
                    f"tiny-root/{split}/{class_name}/{split}_{class_name}.png",
                    _png_bytes(index + 1),
                )
        archive.writestr(
            "tiny-root/train/.hidden/ignored.png",
            _png_bytes(0),
        )
        archive.writestr(
            "tiny-root/.hidden_split/class_0/ignored.png",
            _png_bytes(0),
        )
        archive.writestr("tiny-root/empty/", b"")

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
                    "size": archive_path.stat().st_size,
                    "links": {"content": "https://zenodo.org/api/files/unit/tiny.zip"},
                }
            ]
        }

    monkeypatch.setattr(vision_sources, "_fetch_zenodo_record", fetch_record)

    def download_file(url, target, *, expected_size, checksum):
        assert url.startswith("https://zenodo.org/")
        assert expected_size == archive_path.stat().st_size
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(archive_path, target)

    monkeypatch.setattr(vision_sources, "_download_file", download_file)


def test_zenodo_source_parses_generic_imagefolder_spec():
    spec = vision_sources._parse_zenodo_spec(DATASET)

    assert spec.record_id == "123"
    assert spec.filename == "tiny.zip"

    with pytest.raises(ValueError, match="requires"):
        vision_sources._parse_zenodo_spec("zenodo:123")

    with pytest.raises(ValueError, match="Unsupported Zenodo option"):
        vision_sources._parse_zenodo_spec("zenodo:123?file=tiny.zip&root=data")


@pytest.mark.parametrize(
    "dataset_name",
    [
        "zenodo:abc?file=tiny.zip",
        "zenodo:0?file=tiny.zip",
        "zenodo:123/4?file=tiny.zip",
        "zenodo:123?file=../tiny.zip",
        "zenodo:123?file=/tmp/tiny.zip",
        "zenodo:123?file=dir/tiny.zip",
        "zenodo:123?file=dir%5Ctiny.zip",
        "zenodo:123?file=C%3A%5Ctiny.zip",
        "zenodo:123?file=..",
    ],
)
def test_zenodo_source_rejects_unsafe_record_ids_and_filenames(dataset_name):
    with pytest.raises(ValueError, match="record id|filename"):
        vision_sources._parse_zenodo_spec(dataset_name)


@pytest.mark.parametrize(
    "url",
    [
        "http://zenodo.org/file.zip",
        "file:///tmp/file.zip",
        "https://example.com/file.zip",
        "https://zenodo.org.evil.example/file.zip",
        "https://user@zenodo.org/file.zip",
        "https://zenodo.org:444/file.zip",
        "https://zenodo.org:not-a-port/file.zip",
    ],
)
def test_zenodo_download_link_requires_https_zenodo_origin(url):
    spec = vision_sources._ZenodoImageFolderSpec("123", "tiny.zip")

    with pytest.raises(ValueError, match="https://zenodo.org origin"):
        vision_sources._zenodo_download_url(spec, {"links": {"content": url}})


@pytest.mark.parametrize("size", [-1, True, "100", 21 * 1024**3])
def test_zenodo_metadata_is_validated_before_cache_writes(monkeypatch, tmp_path, size):
    monkeypatch.setattr(
        vision_sources,
        "_fetch_zenodo_record",
        lambda record_id: {
            "files": [
                {
                    "key": "tiny.zip",
                    "size": size,
                    "links": {"content": "https://zenodo.org/unit/tiny.zip"},
                }
            ]
        },
    )

    with pytest.raises(ValueError, match="size|byte limit"):
        vision_sources._prepare_zenodo_archive(
            vision_sources._ZenodoImageFolderSpec("123", "tiny.zip"),
            tmp_path / "cache",
        )

    assert not (tmp_path / "cache").exists()


def test_zenodo_rejects_malformed_record_and_link_metadata():
    with pytest.raises(ValueError, match="record metadata"):
        vision_sources._select_zenodo_file([], "tiny.zip")

    with pytest.raises(ValueError, match="links metadata"):
        vision_sources._zenodo_download_url(
            vision_sources._ZenodoImageFolderSpec("123", "tiny.zip"),
            {"links": ["https://zenodo.org/file"]},
        )


class _DownloadResponse(io.BytesIO):
    pass


def _download_staging_files(target):
    return list(target.parent.glob(f".{target.name}.*.tmp"))


def test_zenodo_download_streams_and_atomically_replaces(monkeypatch, tmp_path):
    payload = b"bounded payload"
    checksum = hashlib.md5(payload).hexdigest()
    target = tmp_path / "files" / "tiny.zip"
    monkeypatch.setattr(
        vision_sources.urllib.request,
        "urlopen",
        lambda url, timeout: _DownloadResponse(payload),
    )

    vision_sources._download_file(
        "https://zenodo.org/file",
        target,
        expected_size=len(payload),
        checksum=f"md5:{checksum}",
    )

    assert target.read_bytes() == payload
    assert not _download_staging_files(target)


@pytest.mark.parametrize(
    ("payload", "expected_size", "checksum", "message"),
    [
        (b"short", 6, None, "size mismatch"),
        (b"oversized", 4, None, "byte limit"),
        (b"payload", 7, "md5:00000000000000000000000000000000", "checksum"),
    ],
)
def test_zenodo_download_failures_leave_no_partial_files(
    monkeypatch, tmp_path, payload, expected_size, checksum, message
):
    target = tmp_path / "tiny.zip"
    monkeypatch.setattr(
        vision_sources.urllib.request,
        "urlopen",
        lambda url, timeout: _DownloadResponse(payload),
    )

    with pytest.raises(ValueError, match=message):
        vision_sources._download_file(
            "https://zenodo.org/file",
            target,
            expected_size=expected_size,
            checksum=checksum,
        )

    assert not target.exists()
    assert not _download_staging_files(target)


def test_zenodo_download_timeout_leaves_no_partial_files(monkeypatch, tmp_path):
    class TimeoutResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, size):
            raise TimeoutError("timed out")

    target = tmp_path / "tiny.zip"
    monkeypatch.setattr(
        vision_sources.urllib.request,
        "urlopen",
        lambda url, timeout: TimeoutResponse(),
    )

    with pytest.raises(TimeoutError, match="timed out"):
        vision_sources._download_file(
            "https://zenodo.org/file",
            target,
            expected_size=1,
            checksum=None,
        )

    assert not target.exists()
    assert not _download_staging_files(target)


def test_zenodo_download_enforces_overall_deadline(monkeypatch, tmp_path):
    target = tmp_path / "tiny.zip"
    monkeypatch.setattr(
        vision_sources.urllib.request,
        "urlopen",
        lambda url, timeout: _DownloadResponse(b"payload"),
    )
    times = iter([0, vision_sources._ZENODO_DOWNLOAD_TIMEOUT_SECONDS + 1])
    monkeypatch.setattr(vision_sources.time, "monotonic", lambda: next(times))

    with pytest.raises(TimeoutError, match="transfer time limit"):
        vision_sources._download_file(
            "https://zenodo.org/file",
            target,
            expected_size=None,
            checksum=None,
        )

    assert not target.exists()
    assert not _download_staging_files(target)


def _write_zip(path, members):
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members:
            archive.writestr(name, payload)


@pytest.mark.parametrize("member_name", ["../escape.txt", "/absolute.txt"])
def test_zip_extraction_rejects_unsafe_paths(tmp_path, member_name):
    archive_path = tmp_path / "hostile.zip"
    _write_zip(archive_path, [(member_name, b"payload")])

    with pytest.raises(ValueError, match="member path|escapes"):
        vision_sources._extract_archive(archive_path, tmp_path / "output")


@pytest.mark.parametrize(
    ("file_type", "message"),
    [
        (stat.S_IFLNK, "link"),
        (stat.S_IFIFO, "regular file or directory"),
        (stat.S_IFCHR, "regular file or directory"),
        (stat.S_IFBLK, "regular file or directory"),
    ],
)
def test_zip_extraction_rejects_links_and_special_files(tmp_path, file_type, message):
    archive_path = tmp_path / "special.zip"
    member = zipfile.ZipInfo("hostile-member")
    member.create_system = 3
    member.external_attr = (file_type | 0o777) << 16
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(member, "target")

    with pytest.raises(ValueError, match=message + ".*hostile-member"):
        vision_sources._extract_archive(archive_path, tmp_path / "output")


def test_zip_extraction_rejects_duplicate_paths(tmp_path):
    archive_path = tmp_path / "duplicate.zip"
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("same.txt", b"first")
            archive.writestr("same.txt", b"second")

    with pytest.raises(ValueError, match="duplicate.*same.txt"):
        vision_sources._extract_archive(archive_path, tmp_path / "output")


def test_zip_extraction_enforces_expanded_size_and_member_limits(monkeypatch, tmp_path):
    archive_path = tmp_path / "budget.zip"
    _write_zip(archive_path, [("one.txt", b"12"), ("two.txt", b"34")])

    monkeypatch.setattr(vision_sources, "_ZENODO_MAX_EXTRACTED_BYTES", 3)
    with pytest.raises(ValueError, match="expanded size"):
        vision_sources._extract_archive(archive_path, tmp_path / "size-output")

    monkeypatch.setattr(vision_sources, "_ZENODO_MAX_EXTRACTED_BYTES", 100)
    monkeypatch.setattr(vision_sources, "_ZENODO_MAX_ARCHIVE_MEMBERS", 1)
    with pytest.raises(ValueError, match="member count"):
        vision_sources._extract_archive(archive_path, tmp_path / "count-output")


def test_archive_preflight_checks_available_cache_space(monkeypatch, tmp_path):
    archive_path = tmp_path / "disk.zip"
    _write_zip(archive_path, [("payload.txt", b"1234")])
    disk_usage = shutil.disk_usage(tmp_path)
    monkeypatch.setattr(
        vision_sources.shutil,
        "disk_usage",
        lambda path: disk_usage._replace(free=3),
    )

    with pytest.raises(ValueError, match="only 3 bytes free"):
        vision_sources._extract_archive(archive_path, tmp_path / "output")


def _write_tar(path, members):
    with tarfile.open(path, "w") as archive:
        for member, payload in members:
            archive.addfile(
                member, io.BytesIO(payload) if payload is not None else None
            )


@pytest.mark.parametrize(
    ("member_type", "message"),
    [
        (tarfile.SYMTYPE, "link"),
        (tarfile.LNKTYPE, "link"),
        (tarfile.FIFOTYPE, "special file"),
        (tarfile.CHRTYPE, "special file"),
        (tarfile.BLKTYPE, "special file"),
    ],
)
def test_tar_extraction_rejects_links_and_special_files(tmp_path, member_type, message):
    archive_path = tmp_path / "hostile.tar"
    member = tarfile.TarInfo("hostile-member")
    member.type = member_type
    member.linkname = "target"
    _write_tar(archive_path, [(member, None)])

    with pytest.raises(ValueError, match=message + ".*hostile-member"):
        vision_sources._extract_archive(archive_path, tmp_path / "output")


def test_tar_extraction_rejects_unsafe_and_duplicate_paths(tmp_path):
    for name, member_names, message in (
        ("escape.tar", ["../escape"], "member path|escapes"),
        ("duplicate.tar", ["same", "same"], "duplicate.*same"),
    ):
        archive_path = tmp_path / name
        members = []
        for member_name in member_names:
            member = tarfile.TarInfo(member_name)
            member.size = 1
            members.append((member, b"x"))
        _write_tar(archive_path, members)

        with pytest.raises(ValueError, match=message):
            vision_sources._extract_archive(archive_path, tmp_path / f"{name}-output")


def test_tar_extraction_rejects_large_declared_size_without_payload(
    monkeypatch, tmp_path
):
    archive_path = tmp_path / "declared-size.tar"
    member = tarfile.TarInfo("huge.bin")
    member.size = 101
    _write_tar(archive_path, [(member, b"\0" * 101)])
    monkeypatch.setattr(vision_sources, "_ZENODO_MAX_EXTRACTED_BYTES", 100)

    with pytest.raises(ValueError, match="expanded size.*101"):
        vision_sources._extract_archive(archive_path, tmp_path / "output")


def test_zip_and_tar_happy_paths_still_extract(tmp_path):
    zip_path = tmp_path / "normal.zip"
    _write_zip(zip_path, [("root/class/image.txt", b"zip")])
    zip_output = tmp_path / "zip-output"
    vision_sources._extract_archive(zip_path, zip_output)
    assert (zip_output / "root" / "class" / "image.txt").read_bytes() == b"zip"

    tar_path = tmp_path / "normal.tar"
    member = tarfile.TarInfo("root/class/image.txt")
    member.size = 3
    _write_tar(tar_path, [(member, b"tar")])
    tar_output = tmp_path / "tar-output"
    vision_sources._extract_archive(tar_path, tar_output)
    assert (tar_output / "root" / "class" / "image.txt").read_bytes() == b"tar"


def test_zenodo_cache_preparation_is_serialized(monkeypatch, tmp_path):
    archive_path, checksum = _write_tiny_imagefolder_zip(tmp_path)
    calls = 0

    monkeypatch.setattr(
        vision_sources,
        "_fetch_zenodo_record",
        lambda record_id: {
            "files": [
                {
                    "key": "tiny.zip",
                    "checksum": f"md5:{checksum}",
                    "size": archive_path.stat().st_size,
                    "links": {"content": "https://zenodo.org/unit/tiny.zip"},
                }
            ]
        },
    )

    def download(url, target, *, expected_size, checksum):
        nonlocal calls
        calls += 1
        time.sleep(0.05)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(archive_path, target)

    monkeypatch.setattr(vision_sources, "_download_file", download)
    spec = vision_sources._ZenodoImageFolderSpec("123", "tiny.zip")
    data_dir = tmp_path / "cache"

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda _: vision_sources._prepare_zenodo_archive(spec, data_dir),
                range(2),
            )
        )

    assert calls == 1
    assert results[0] == results[1]
    assert (results[0] / "tiny-root" / "train").is_dir()
    assert not list(results[0].parent.glob(".tiny.*"))


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
        tmp_path / "cache" / "zenodo" / "123" / "extracted" / "tiny" / "tiny-root"
    ).is_dir()
    assert int(tf.data.Dataset.cardinality(train_ds).numpy()) == 3
    assert int(tf.data.Dataset.cardinality(val_ds).numpy()) == 2

    samples = list(train_ds.as_numpy_iterator())

    assert [int(sample["label"]) for sample in samples] == [0, 1, 2]
    assert samples[0]["image"].shape == (4, 4, 3)
    assert samples[0]["image"].dtype == np.uint8
    assert samples[0]["metadata"]["dataset"] == DATASET.encode()
    assert samples[0]["metadata"]["record_id"] == b"123"
    assert samples[0]["metadata"]["archive"] == b"tiny.zip"
    assert samples[0]["metadata"]["split"] == b"train"
    assert samples[0]["metadata"]["class_name"] == b"class_a"
    assert samples[0]["metadata"]["class_index"] == 0
    assert samples[0]["metadata"]["example_id"] == (b"train/class_a/train_class_a.png")


def test_zenodo_imagefolder_labels_are_stable_across_separate_split_loads(
    monkeypatch, tmp_path
):
    archive_path, checksum = _write_tiny_imagefolder_zip(tmp_path)
    _install_tiny_zenodo_record(monkeypatch, archive_path, checksum)
    data_dir = tmp_path / "cache"

    (train_ds,) = vision_sources.load_zenodo_imagefolder_splits(
        DATASET,
        ["train"],
        data_dir=data_dir,
    )
    (val_ds,) = vision_sources.load_zenodo_imagefolder_splits(
        DATASET,
        ["val"],
        data_dir=data_dir,
    )
    combined_train_ds, combined_val_ds = vision_sources.load_zenodo_imagefolder_splits(
        DATASET,
        ["train", "val"],
        data_dir=data_dir,
    )

    train_samples = list(train_ds.as_numpy_iterator())
    val_samples = list(val_ds.as_numpy_iterator())

    assert [int(sample["label"]) for sample in train_samples] == [0, 1, 2]
    assert [int(sample["label"]) for sample in val_samples] == [1, 2]
    assert [int(sample["label"]) for sample in train_samples] == [
        int(sample["label"]) for sample in combined_train_ds.as_numpy_iterator()
    ]
    assert [int(sample["label"]) for sample in val_samples] == [
        int(sample["label"]) for sample in combined_val_ds.as_numpy_iterator()
    ]
    expected_labels = {b"class_a": 0, b"class_b": 1, b"class_c": 2}
    for sample in train_samples + val_samples:
        assert sample["metadata"]["class_index"] == sample["label"]
        assert expected_labels[sample["metadata"]["class_name"]] == sample["label"]


def test_zenodo_imagefolder_empty_split_still_errors(monkeypatch, tmp_path):
    archive_path, checksum = _write_tiny_imagefolder_zip(tmp_path)
    _install_tiny_zenodo_record(monkeypatch, archive_path, checksum)

    with pytest.raises(ValueError, match="split 'empty' has no images"):
        vision_sources.load_zenodo_imagefolder_splits(
            DATASET,
            ["empty"],
            data_dir=tmp_path / "cache",
        )


def test_zenodo_imagefolder_unknown_class_errors_descriptively(tmp_path):
    (tmp_path / "val" / "class_a").mkdir(parents=True)

    with pytest.raises(ValueError, match="absent from the archive-wide vocabulary"):
        vision_sources._imagefolder_records(
            tmp_path,
            ["val"],
            dataset_name=DATASET,
            spec=vision_sources._ZenodoImageFolderSpec("123", "tiny.zip"),
            class_to_label={},
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
        num_classes=3,
        cache_dataset=False,
        metadata_mode="numeric_only",
        data_dir=tmp_path / "cache",
    )
    batch = next(iter(ds))

    assert int(n.numpy()) == 1
    assert batch["image"].shape == (2, 3, 8, 8)
    np.testing.assert_array_equal(batch["label"].numpy(), [1, 2])
    np.testing.assert_array_equal(batch["metadata"]["class_index"].numpy(), [1, 2])
    np.testing.assert_array_equal(batch["metadata"]["hard_label"].numpy(), [1, 2])
    np.testing.assert_array_equal(batch["padding_mask"].numpy(), [True, True])

import csv

import numpy as np
import tensorflow as tf

from justdata.acoustic.sources import load_local_audio_manifest_splits


def _write_manifest(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "path",
                "label",
                "split",
                "clip_id",
                "source_id",
                "start_time",
                "end_time",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_local_manifest_loader_cardinality(tmp_path, write_wav_file):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    write_wav_file(audio_dir / "train.wav", np.arange(8, dtype=np.int16))
    write_wav_file(audio_dir / "val.wav", np.arange(8, dtype=np.int16))
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [
            {
                "path": "audio/train.wav",
                "label": "1",
                "split": "train",
                "clip_id": "clip-train",
                "source_id": "source-a",
                "start_time": "0.0",
                "end_time": "0.0",
            },
            {
                "path": "audio/val.wav",
                "label": "2",
                "split": "validation",
                "clip_id": "clip-val",
                "source_id": "source-b",
                "start_time": "0.0",
                "end_time": "0.0",
            },
        ],
    )

    ds = load_local_audio_manifest_splits(f"local_audio:{manifest}", ["train"])[0]

    assert tf.data.Dataset.cardinality(ds).numpy() == 1


def test_local_manifest_preserves_filename_and_clip_id(tmp_path, write_wav_file):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    write_wav_file(audio_dir / "train.wav", np.arange(8, dtype=np.int16))
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [
            {
                "path": "audio/train.wav",
                "label": "1",
                "split": "train",
                "clip_id": "clip-train",
                "source_id": "source-a",
                "start_time": "0.0",
                "end_time": "0.0",
            },
        ],
    )

    ds = load_local_audio_manifest_splits(f"local_audio:{manifest}", ["train"])[0]
    sample = next(iter(ds))

    assert sample["filename"].numpy() == b"train.wav"
    assert sample["clip_id"].numpy() == b"clip-train"

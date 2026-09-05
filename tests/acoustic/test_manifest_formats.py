import csv
import json

import pytest
import tensorflow as tf

from justdata.acoustic.dcase2025 import load_dcase2025_task1_splits
from justdata.acoustic.sources import load_local_audio_manifest_splits


def _write_manifest(path, *, clip_id="clip-a", label=1):
    row = {
        "path": "airport-london-0001-0001-a.wav",
        "label": label,
        "split": "eval",
        "clip_id": clip_id,
        "source_id": None,
        "start_time": None,
        "end_time": 1,
    }
    if path.suffix == ".jsonl":
        path.write_text("\n" + json.dumps(row) + "\n\n", encoding="utf-8")
    else:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=row,
                delimiter="\t" if path.suffix in {".tsv", ".tab"} else ",",
            )
            writer.writeheader()
            writer.writerow(row)


@pytest.mark.parametrize("suffix", [".csv", ".tsv", ".tab", ".jsonl"])
@pytest.mark.parametrize(
    "prefix,loader",
    [
        ("local_audio:", load_local_audio_manifest_splits),
        ("dcase2025:", load_dcase2025_task1_splits),
    ],
)
def test_manifest_formats_preserve_records_and_empty_splits(
    tmp_path, suffix, prefix, loader
):
    manifest = tmp_path / f"manifest{suffix}"
    _write_manifest(manifest)

    dataset, empty = loader(f"{prefix}{manifest}", ["eval", "absent"])
    sample = next(iter(dataset))

    assert tf.data.Dataset.cardinality(dataset).numpy() == 1
    assert tf.data.Dataset.cardinality(empty).numpy() == 0
    assert sample["clip_id"].numpy() == b"clip-a"
    assert (
        sample["path"].numpy()
        == str(tmp_path / "airport-london-0001-0001-a.wav").encode()
    )
    assert sample["start_time"].numpy() == 0.0
    assert sample["end_time"].numpy() == 1.0


def test_manifest_discovery_keeps_source_specific_precedence(tmp_path):
    _write_manifest(tmp_path / "manifest.jsonl", clip_id="jsonl")
    _write_manifest(tmp_path / "metadata.csv", clip_id="csv")

    local = load_local_audio_manifest_splits("local_audio:", ["eval"], tmp_path)[0]
    dcase = load_dcase2025_task1_splits("dcase2025:", ["eval"], tmp_path)[0]

    assert next(iter(local))["clip_id"].numpy() == b"jsonl"
    assert next(iter(dcase))["clip_id"].numpy() == b"csv"


def test_local_manifest_requires_columns_and_row_labels(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"path": "missing.wav"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="missing required columns"):
        load_local_audio_manifest_splits(f"local_audio:{manifest}", ["eval"])

    _write_manifest(manifest, label=None)
    with pytest.raises(ValueError, match="rows must contain a label"):
        load_local_audio_manifest_splits(f"local_audio:{manifest}", ["eval"])

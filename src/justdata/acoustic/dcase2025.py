from __future__ import annotations

import csv
import json
import os
import re
import warnings
from pathlib import Path
from typing import Any, Literal, Union

import numpy as np
import tensorflow as tf
from loguru import logger

from justdata.acoustic.adapters import adapt_acoustic_sample
from justdata.acoustic.presets import register_preset
from justdata.acoustic.schema import (
    CLIP_ID,
    DATASET,
    END_TIME,
    EXAMPLE_ID,
    FILENAME,
    LABEL,
    METADATA,
    PATH,
    SOURCE_ID,
    SPLIT,
    START_TIME,
)
from justdata.core.adapters import register_adapter
from justdata.core.filters import metadata_filter_predicate
from justdata.core.loader import load_ds
from justdata.core.registry import get_pipeline
from justdata.core.sources import register_source_loader


class SplitLeakageError(ValueError):
    pass


known_train_devices = {"A", "B", "C", "S1", "S2", "S3"}
real_devices = {"A", "B", "C", "D"}
simulated_devices = {"S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10"}

allow_stats_on_split = {
    "dev_train_25": True,
    "dev_test": False,
    "eval": False,
}

_DEVICE_RE = re.compile(
    r"(?:^|[._\-/])(?P<device>unknown|s10|s[1-9]|[abcd])(?:$|[._\-/])",
    re.IGNORECASE,
)
_MISSING_VALUES = {"", "none", "null", "nan", "n/a", "na"}


def _text(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "numpy"):
        try:
            value = value.numpy()
        except Exception:
            return None
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    value = str(value).strip()
    if value.lower() in _MISSING_VALUES:
        return None
    return value


def _row_value(row: dict, *keys: str) -> str | None:
    lowered = {str(key).lower(): value for key, value in row.items()}
    for key in keys:
        value = row.get(key)
        if value is None:
            value = lowered.get(key.lower())
        value = _text(value)
        if value is not None:
            return value
    return None


def _filename_from_row(row: dict) -> str | None:
    value = _row_value(row, FILENAME, "file", "filename", "audio_filename")
    if value is not None:
        return Path(value).name
    path = _row_value(row, PATH, "audio_path", "filepath", "file_path")
    if path is None:
        return None
    return Path(path).name


def _stem_tokens(row: dict) -> list[str]:
    filename = _filename_from_row(row)
    if filename is None:
        return []
    return [token for token in re.split(r"[._\-/]+", Path(filename).stem) if token]


def _normalize_device(value: str | None) -> str | None:
    if value is None:
        return None
    value = Path(value).stem.strip()
    if value.lower() == "unknown":
        return "unknown"
    value = value.upper()
    if value in real_devices or value in simulated_devices:
        return value
    return None


def _scene_from_filename(row: dict) -> str | None:
    filename = _filename_from_row(row)
    if filename is None:
        return None
    stem = Path(filename).stem.lower()
    for name in DCASE2025Task1Adapter.class_names:
        name_pattern = re.escape(name).replace("_", r"[._\-/]")
        pattern = re.compile(rf"(?:^|[._\-/]){name_pattern}(?:$|[._\-/])")
        if pattern.search(stem):
            return name
    return None


def parse_scene_from_row(row) -> str | None:
    value = _row_value(row, "scene_label", "scene", "scene_name", "class_name")
    if value is None:
        value = _row_value(row, LABEL, "target")
    if value is not None:
        normalized = value.lower().replace("-", "_").replace(" ", "_")
        if normalized in DCASE2025Task1Adapter.class_names:
            return normalized
        if normalized.isdigit():
            index = int(normalized)
            if 0 <= index < len(DCASE2025Task1Adapter.class_names):
                return DCASE2025Task1Adapter.class_names[index]
    return _scene_from_filename(row)


def parse_device_from_row(row) -> str:
    value = _normalize_device(
        _row_value(
            row,
            "device",
            "device_id",
            "recording_device",
            "source_device",
            "domain",
        )
    )
    if value is not None:
        return value

    filename = _filename_from_row(row)
    if filename is not None:
        matches = list(_DEVICE_RE.finditer(filename))
        if matches:
            device = _normalize_device(matches[-1].group("device"))
            if device is not None:
                return device

    split = _row_value(row, SPLIT, "subset", "partition")
    if split == "eval":
        return "unknown"
    raise ValueError("Could not parse DCASE device from row.")


def parse_city_from_row(row) -> str | None:
    value = _row_value(row, "city", "recording_city", "location_city")
    if value is not None:
        return value

    scene = parse_scene_from_row(row)
    if scene is None:
        return None
    tokens = _stem_tokens(row)
    scene_tokens = scene.split("_") if scene else []
    if scene_tokens and [token.lower() for token in tokens[: len(scene_tokens)]] == scene_tokens:
        tokens = tokens[len(scene_tokens) :]
    if tokens:
        candidate = tokens[0]
        if _normalize_device(candidate) is None and not candidate.isdigit():
            return candidate
    return None


def _parse_location_id(row) -> str | None:
    value = _row_value(row, "location_id", "location", "recording_location")
    if value is not None:
        return value
    tokens = _stem_tokens(row)
    scene = parse_scene_from_row(row)
    if scene is None:
        return None
    offset = len(scene.split("_")) if scene else 0
    if len(tokens) > offset + 1:
        candidate = tokens[offset + 1]
        if _normalize_device(candidate) is None:
            return candidate
    return None


def _parse_segment_id(row) -> str | None:
    value = _row_value(row, "segment_id", "segment", CLIP_ID)
    if value is not None:
        return value
    tokens = _stem_tokens(row)
    if tokens and _normalize_device(tokens[-1]) is not None:
        tokens = tokens[:-1]
    if tokens and tokens[-1].isdigit():
        return tokens[-1]
    return None


def parse_source_recording_id(row) -> str | None:
    value = _row_value(
        row,
        "source_recording_id",
        "original_recording_id",
        "recording_id",
        SOURCE_ID,
    )
    if value is not None:
        return value

    tokens = _stem_tokens(row)
    if not tokens:
        return None
    if _normalize_device(tokens[-1]) is not None:
        tokens = tokens[:-1]
    segment_id = _parse_segment_id(row)
    if segment_id is not None and tokens and tokens[-1] == segment_id:
        tokens = tokens[:-1]
    elif tokens and tokens[-1].isdigit():
        tokens = tokens[:-1]
    return "-".join(tokens) if tokens else None


def _device_type(device: str) -> Literal["real", "simulated", "unknown"]:
    if device in real_devices:
        return "real"
    if device in simulated_devices:
        return "simulated"
    return "unknown"


def _scene_id(scene_label: str | None) -> int | None:
    if scene_label is None:
        return None
    try:
        return DCASE2025Task1Adapter.class_names.index(scene_label)
    except ValueError:
        return None


def _tensor_scene_id(scene_label: tf.Tensor) -> tf.Tensor:
    labels = tf.constant(DCASE2025Task1Adapter.class_names, dtype=tf.string)
    matches = tf.equal(labels, scene_label)
    return tf.cond(
        tf.reduce_any(matches),
        lambda: tf.cast(tf.argmax(tf.cast(matches, tf.int32)), tf.int64),
        lambda: tf.constant(-1, dtype=tf.int64),
    )


def _tensor_device_type(device: tf.Tensor) -> tf.Tensor:
    real = tf.constant(sorted(real_devices), dtype=tf.string)
    simulated = tf.constant(sorted(simulated_devices), dtype=tf.string)
    return tf.case(
        (
            (tf.reduce_any(tf.equal(real, device)), lambda: tf.constant("real")),
            (tf.reduce_any(tf.equal(simulated, device)), lambda: tf.constant("simulated")),
        ),
        default=lambda: tf.constant("unknown"),
        exclusive=False,
    )


def _tensor_known_device(device: tf.Tensor) -> tf.Tensor:
    known = tf.constant(sorted(known_train_devices), dtype=tf.string)
    return tf.reduce_any(tf.equal(known, device))


def _optional_text(value: Any) -> Any:
    if hasattr(value, "dtype"):
        return value
    return _text(value)


def _optional_int(value: Any) -> Any:
    if hasattr(value, "dtype"):
        return value
    if value is None:
        return None
    value = int(value)
    return value if value >= 0 else None


def _field_or_parse(sample: dict, key: str, parser) -> Any:
    if key in sample:
        return sample[key]
    return parser(sample)


class DCASE2025Task1Adapter:
    dataset_name = "dcase2025_task1"
    class_names = (
        "airport",
        "shopping_mall",
        "metro_station",
        "street_pedestrian",
        "public_square",
        "street_traffic",
        "tram",
        "bus",
        "metro",
        "park",
    )

    allow_stats_on_split = allow_stats_on_split

    def __call__(self, sample: dict) -> dict:
        has_audio = any(key in sample for key in ("waveform", "audio", PATH))
        result = (
            adapt_acoustic_sample(sample, dataset=self.dataset_name, label_key=None)
            if has_audio
            else dict(sample)
        )

        metadata = dict(result.get(METADATA, {}))
        split = sample.get(SPLIT)
        if split is None:
            split = _row_value(sample, SPLIT, "subset", "partition")
        filename = sample.get(FILENAME)
        if filename is None:
            filename = _filename_from_row(sample)

        scene_label = sample.get("scene_label")
        if scene_label is None:
            scene_label = parse_scene_from_row(sample)
        scene_id = sample.get("scene_id")
        if scene_id is None:
            if hasattr(scene_label, "dtype"):
                scene_id = _tensor_scene_id(scene_label)
            else:
                scene_id = _scene_id(scene_label)

        device = sample.get("device")
        if device is None:
            device = parse_device_from_row(sample)
        if not hasattr(device, "dtype"):
            device = parse_device_from_row({**sample, "device": device})

        if hasattr(device, "dtype"):
            device_type = _tensor_device_type(device)
            is_known_device = _tensor_known_device(device)
        else:
            device_type = _device_type(device)
            is_known_device = device in known_train_devices

        if LABEL in sample:
            result[LABEL] = tf.cast(sample[LABEL], tf.int64)
        elif not hasattr(scene_id, "dtype") and scene_id is not None:
            result[LABEL] = scene_id

        metadata.update(
            {
                "dataset": tf.constant(self.dataset_name) if hasattr(device, "dtype") else self.dataset_name,
                "split": _optional_text(split),
                "filename": _optional_text(filename),
                "scene_label": _optional_text(scene_label),
                "scene_id": _optional_int(scene_id),
                "device": device,
                "device_type": device_type,
                "is_known_device": is_known_device,
                "city": _optional_text(_field_or_parse(sample, "city", parse_city_from_row)),
                "location_id": _optional_text(_field_or_parse(sample, "location_id", _parse_location_id)),
                "segment_id": _optional_text(_field_or_parse(sample, "segment_id", _parse_segment_id)),
                "source_recording_id": _optional_text(
                    _field_or_parse(sample, "source_recording_id", parse_source_recording_id)
                ),
                "fold": _optional_text(sample.get("fold")),
            }
        )
        result[METADATA] = metadata
        return result


_ADAPTER = DCASE2025Task1Adapter()


@register_adapter("dcase2025_task1")
@register_adapter("dcase2025:")
def dcase2025_task1_adapter(sample: dict) -> dict:
    return _ADAPTER(sample)


def assert_stats_allowed(split: str, *, allow_override: bool = False) -> None:
    if split not in allow_stats_on_split:
        raise ValueError(f"Unknown DCASE 2025 split: {split!r}.")
    if allow_stats_on_split[split]:
        return
    message = (
        f"Statistics on split {split!r} are not DCASE 2025 Task 1 compliant. "
        "Use dev_train_25 for training, normalization statistics, and adaptation."
    )
    if allow_override:
        logger.warning(f"{message} Proceeding because allow_override=True.")
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        return
    raise SplitLeakageError(message)


def _register_presets() -> None:
    base = {
        "name": "dcase2025_task1_efficientat_32k_1s",
        "input_duration": 1.0,
        "target_sample_rate": 32000,
        "preprocess": {
            "target_sample_rate": 32000,
            "resampler": "tensorflow",
            "channel_strategy": "mono_mean",
        },
        "segment": {
            "clip_duration": 1.0,
            "train_mode": "random_crop",
            "eval_mode": "center_crop",
            "pad_mode": "zero",
            "pad_position": "right",
        },
        "frontend": {
            "name": "logmel",
            "stft": {
                "sample_rate": 32000,
                "n_fft": 1024,
                "win_length": 1024,
                "hop_length": 320,
                "center": True,
            },
            "mel": {"n_mels": 128, "f_min": 0.0, "f_max": 14000.0},
            "log": {"kind": "log"},
        },
        "label_transform": {
            "mode": "index",
            "num_classes": len(DCASE2025Task1Adapter.class_names),
            "class_names": DCASE2025Task1Adapter.class_names,
        },
        "layout": "btf",
        "output_key": "inputs",
        "dtype": "float32",
        "metadata_mode": "numeric_only",
    }
    register_preset("dcase2025_task1_efficientat_32k_1s", base)

    native = dict(base)
    native["name"] = "dcase2025_task1_native_44k_1s"
    native["target_sample_rate"] = 44100
    native["preprocess"] = dict(base["preprocess"], target_sample_rate=44100)
    native["frontend"] = dict(base["frontend"])
    native["frontend"]["stft"] = dict(base["frontend"]["stft"], sample_rate=44100)
    register_preset("dcase2025_task1_native_44k_1s", native)


_register_presets()


def _read_manifest(path: Path) -> list[dict[str, str]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(
                        {str(key): "" if value is None else str(value) for key, value in json.loads(line).items()}
                    )
        return rows

    dialect = csv.excel_tab if suffix in {".tsv", ".tab"} else csv.excel
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, dialect=dialect)
        return [
            {str(key): "" if value is None else str(value) for key, value in row.items()}
            for row in reader
        ]


def _resolve_manifest_path(
    dataset_name: str,
    data_dir: Union[None, str, os.PathLike],
) -> Path:
    if dataset_name.startswith("dcase2025:"):
        spec = dataset_name[len("dcase2025:") :]
        path = Path(spec).expanduser() if spec else Path(data_dir or "")
        if not path.is_absolute() and data_dir is not None:
            candidate = Path(data_dir).expanduser() / path
            if candidate.exists() or not path.exists():
                path = candidate
    elif data_dir is not None:
        path = Path(data_dir).expanduser()
    else:
        raise ValueError("dcase2025_task1 loading requires data_dir or a dcase2025: manifest path.")

    if path.is_dir():
        for name in (
            "manifest.csv",
            "metadata.csv",
            "dcase2025_task1.csv",
            "manifest.tsv",
            "metadata.tsv",
            "manifest.jsonl",
            "metadata.jsonl",
        ):
            candidate = path / name
            if candidate.exists():
                return candidate
        raise ValueError(f"No DCASE manifest found in {path}.")
    return path


def _resolve_audio_path(row: dict, manifest_path: Path, data_dir: Union[None, str, os.PathLike]) -> str:
    value = _row_value(row, PATH, "audio_path", "filepath", "file_path", FILENAME)
    if value is None:
        raise ValueError("DCASE manifest rows must contain a path, audio_path, or filename.")
    path = Path(value).expanduser()
    if path.is_absolute():
        return os.fspath(path)
    manifest_relative = manifest_path.parent / path
    if manifest_relative.exists() or data_dir is None:
        return os.fspath(manifest_relative)
    return os.fspath(Path(data_dir).expanduser() / path)


def _float_value(value: str | None) -> float:
    return 0.0 if value is None else float(value)


def _record(row: dict[str, str], manifest_path: Path, split: str, data_dir) -> dict:
    audio_path = _resolve_audio_path(row, manifest_path, data_dir)
    filename = _row_value(row, FILENAME, "filename") or Path(audio_path).name
    scene_label = parse_scene_from_row(row)
    scene_id = _scene_id(scene_label)
    device = parse_device_from_row({**row, SPLIT: split})
    segment_id = _parse_segment_id(row) or Path(filename).stem
    source_recording_id = parse_source_recording_id(row) or segment_id

    record = {
        PATH: audio_path,
        SPLIT: split,
        DATASET: "dcase2025_task1",
        EXAMPLE_ID: _row_value(row, EXAMPLE_ID, "example_id") or f"{split}:{Path(filename).stem}",
        CLIP_ID: _row_value(row, CLIP_ID, "clip_id") or segment_id,
        SOURCE_ID: _row_value(row, SOURCE_ID, "source_id") or source_recording_id,
        START_TIME: _float_value(_row_value(row, START_TIME, "start")),
        END_TIME: _float_value(_row_value(row, END_TIME, "end")),
        FILENAME: filename,
        "scene_label": scene_label or "",
        "scene_id": -1 if scene_id is None else scene_id,
        "device": device,
        "device_type": _device_type(device),
        "is_known_device": device in known_train_devices,
        "city": parse_city_from_row(row) or "",
        "location_id": _parse_location_id(row) or "",
        "segment_id": segment_id,
        "source_recording_id": source_recording_id,
        "fold": _row_value(row, "fold") or "",
    }
    if scene_id is not None:
        record[LABEL] = np.int64(scene_id)
    return record


def _records_to_dataset(records: list[dict]) -> tf.data.Dataset:
    has_label = any(LABEL in record for record in records)
    keys = [
        PATH,
        SPLIT,
        DATASET,
        EXAMPLE_ID,
        CLIP_ID,
        SOURCE_ID,
        START_TIME,
        END_TIME,
        FILENAME,
        "scene_label",
        "scene_id",
        "device",
        "device_type",
        "is_known_device",
        "city",
        "location_id",
        "segment_id",
        "source_recording_id",
        "fold",
    ]
    if has_label:
        keys.append(LABEL)

    def gen():
        for record in records:
            item = {}
            for key in keys:
                if key == LABEL:
                    item[key] = np.int64(record.get(key, -1))
                else:
                    item[key] = record[key]
            yield item

    signature = {
        PATH: tf.TensorSpec(shape=(), dtype=tf.string),
        SPLIT: tf.TensorSpec(shape=(), dtype=tf.string),
        DATASET: tf.TensorSpec(shape=(), dtype=tf.string),
        EXAMPLE_ID: tf.TensorSpec(shape=(), dtype=tf.string),
        CLIP_ID: tf.TensorSpec(shape=(), dtype=tf.string),
        SOURCE_ID: tf.TensorSpec(shape=(), dtype=tf.string),
        START_TIME: tf.TensorSpec(shape=(), dtype=tf.float32),
        END_TIME: tf.TensorSpec(shape=(), dtype=tf.float32),
        FILENAME: tf.TensorSpec(shape=(), dtype=tf.string),
        "scene_label": tf.TensorSpec(shape=(), dtype=tf.string),
        "scene_id": tf.TensorSpec(shape=(), dtype=tf.int64),
        "device": tf.TensorSpec(shape=(), dtype=tf.string),
        "device_type": tf.TensorSpec(shape=(), dtype=tf.string),
        "is_known_device": tf.TensorSpec(shape=(), dtype=tf.bool),
        "city": tf.TensorSpec(shape=(), dtype=tf.string),
        "location_id": tf.TensorSpec(shape=(), dtype=tf.string),
        "segment_id": tf.TensorSpec(shape=(), dtype=tf.string),
        "source_recording_id": tf.TensorSpec(shape=(), dtype=tf.string),
        "fold": tf.TensorSpec(shape=(), dtype=tf.string),
    }
    if has_label:
        signature[LABEL] = tf.TensorSpec(shape=(), dtype=tf.int64)

    ds = tf.data.Dataset.from_generator(gen, output_signature=signature)
    return ds.apply(tf.data.experimental.assert_cardinality(len(records)))


def load_dcase2025_task1_splits(
    dataset_name: str,
    splits: list[str],
    data_dir: Union[None, str, os.PathLike] = None,
) -> list[tf.data.Dataset]:
    manifest_path = _resolve_manifest_path(dataset_name, data_dir)
    rows = _read_manifest(manifest_path)
    datasets = []
    for split in splits:
        split_rows = [
            row
            for row in rows
            if (_row_value(row, SPLIT, "subset", "partition") or split) == split
        ]
        records = [_record(row, manifest_path, split, data_dir) for row in split_rows]
        datasets.append(_records_to_dataset(records))
    return datasets


@register_source_loader("dcase2025_task1")
def _load_dcase2025_task1_splits(
    dataset_name: str,
    splits: list[str],
    data_dir: Union[None, str, os.PathLike] = None,
) -> list[tf.data.Dataset]:
    return load_dcase2025_task1_splits(dataset_name, splits, data_dir)


def _dataset_type_for_split(split: str, source: bool) -> str:
    if source and split == "dev_train_25":
        return "train"
    return "validation"


def _make_dataset(
    *,
    dataset: str,
    split: str,
    domain: dict[str, Any] | None,
    preset: str,
    allow_stats: bool,
    allow_override: bool,
    source: bool,
    load_kwargs: dict[str, Any],
):
    if allow_stats:
        assert_stats_allowed(split, allow_override=allow_override)

    filter_fn = load_kwargs.pop("filter_fn", None)
    if domain:
        domain_filter = metadata_filter_predicate(**domain)
        if filter_fn is None:
            filter_fn = domain_filter
        else:
            previous_filter = filter_fn
            filter_fn = lambda sample: tf.logical_and(
                previous_filter(sample),
                domain_filter(sample),
            )

    pipeline = load_kwargs.pop("pipeline", None)
    if pipeline is None:
        pipeline = get_pipeline(dataset=dataset, preset=preset)

    return load_ds(
        dataset,
        split,
        load_kwargs.pop("dataset_type", _dataset_type_for_split(split, source=source)),
        load_kwargs.pop("batch_size", 32),
        load_kwargs.pop("seed", 0),
        pipeline=pipeline,
        deterministic=load_kwargs.pop("deterministic", True),
        cache_dataset=load_kwargs.pop("cache_dataset", False),
        filter_fn=filter_fn,
        **load_kwargs,
    )


def make_source_dataset(
    dataset: str = "dcase2025_task1",
    split: str = "dev_train_25",
    source_domain: dict[str, Any] | None = None,
    preset: str = "dcase2025_task1_efficientat_32k_1s",
    *,
    allow_stats: bool = True,
    allow_override: bool = False,
    **load_kwargs,
):
    return _make_dataset(
        dataset=dataset,
        split=split,
        domain=source_domain,
        preset=preset,
        allow_stats=allow_stats,
        allow_override=allow_override,
        source=True,
        load_kwargs=dict(load_kwargs),
    )


def make_target_dataset(
    dataset: str = "dcase2025_task1",
    split: str = "dev_test",
    target_domain: dict[str, Any] | None = None,
    preset: str = "dcase2025_task1_efficientat_32k_1s",
    *,
    allow_stats: bool = False,
    allow_override: bool = False,
    **load_kwargs,
):
    return _make_dataset(
        dataset=dataset,
        split=split,
        domain=target_domain,
        preset=preset,
        allow_stats=allow_stats,
        allow_override=allow_override,
        source=False,
        load_kwargs=dict(load_kwargs),
    )


__all__ = [
    "DCASE2025Task1Adapter",
    "SplitLeakageError",
    "allow_stats_on_split",
    "assert_stats_allowed",
    "dcase2025_task1_adapter",
    "known_train_devices",
    "load_dcase2025_task1_splits",
    "make_source_dataset",
    "make_target_dataset",
    "parse_city_from_row",
    "parse_device_from_row",
    "parse_scene_from_row",
    "parse_source_recording_id",
    "real_devices",
    "simulated_devices",
]

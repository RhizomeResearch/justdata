from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Literal, Union

import numpy as np
import tensorflow as tf
from loguru import logger

from justdata.acoustic.adapters import (
    standardize_waveform_layout,
    to_float32_waveform,
)
from justdata.acoustic.decoding import decode_audio_file
from justdata.acoustic.schema import (
    CLIP_ID,
    DATASET,
    END_TIME,
    EXAMPLE_ID,
    FILENAME,
    LABEL,
    PATH,
    SAMPLE_RATE,
    SOURCE_ID,
    SPLIT,
    START_TIME,
    WAVEFORM,
)
from justdata.core.sources import register_source_loader


_LOCAL_REQUIRED_COLUMNS = {
    PATH,
    LABEL,
    SPLIT,
    CLIP_ID,
    SOURCE_ID,
    START_TIME,
    END_TIME,
}


def _strip_prefix(dataset_name: str, prefix: str) -> str:
    if not dataset_name.startswith(prefix):
        raise ValueError(f"Expected dataset name to start with {prefix!r}; got {dataset_name!r}.")
    return dataset_name[len(prefix) :]


def _resolve_manifest_path(
    dataset_name: str,
    prefix: str,
    data_dir: Union[None, str, os.PathLike],
) -> Path:
    spec = _strip_prefix(dataset_name, prefix)
    if not spec:
        if data_dir is None:
            raise ValueError(f"{prefix} requires a manifest path or data_dir.")
        path = Path(data_dir)
    else:
        path = Path(spec).expanduser()
        if not path.is_absolute() and data_dir is not None:
            candidate = Path(data_dir).expanduser() / path
            if candidate.exists() or not path.exists():
                path = candidate

    if path.is_dir():
        for name in ("manifest.csv", "manifest.jsonl", "metadata.csv", "metadata.jsonl"):
            candidate = path / name
            if candidate.exists():
                return candidate
        raise ValueError(f"No manifest.csv/jsonl or metadata.csv/jsonl found in {path}.")

    return path


def _read_manifest(path: Path) -> tuple[list[dict[str, str]], set[str]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = []
        columns = set()
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row = {str(k): "" if v is None else str(v) for k, v in row.items()}
                rows.append(row)
                columns.update(row)
        return rows, columns

    with path.open("r", encoding="utf-8", newline="") as f:
        dialect = csv.excel_tab if suffix in {".tsv", ".tab"} else csv.excel
        reader = csv.DictReader(f, dialect=dialect)
        rows = [
            {str(k): "" if v is None else str(v) for k, v in row.items()}
            for row in reader
        ]
        return rows, set(reader.fieldnames or [])


def _coalesce(row: dict[str, str], *keys: str, default: str = "") -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return default


def _float_value(value: str, default: float = 0.0) -> float:
    if value == "":
        return default
    return float(value)


def _resolve_audio_path(value: str, manifest_path: Path, data_dir: Union[None, str, os.PathLike]) -> str:
    path = Path(value).expanduser()
    if path.is_absolute():
        return os.fspath(path)

    manifest_relative = manifest_path.parent / path
    if manifest_relative.exists() or data_dir is None:
        return os.fspath(manifest_relative)

    data_relative = Path(data_dir).expanduser() / path
    return os.fspath(data_relative)


def _label_dtype(records: list[dict]) -> tf.DType:
    labels = [record[LABEL] for record in records if LABEL in record]
    if labels and all(isinstance(label, (int, np.integer)) for label in labels):
        return tf.int64
    return tf.string


def _looks_int(value: str) -> bool:
    try:
        int(value)
    except (TypeError, ValueError):
        return False
    return True


def _manifest_records(
    rows: list[dict[str, str]],
    manifest_path: Path,
    requested_split: str,
    *,
    data_dir: Union[None, str, os.PathLike],
    dataset: str,
    require_local_columns: bool,
) -> list[dict]:
    records = []
    for row in rows:
        split = _coalesce(row, SPLIT, "subset", "partition", default=requested_split)
        if split != requested_split:
            continue

        audio_path_value = _coalesce(row, PATH, "audio_path", "filename", "file")
        if not audio_path_value:
            raise ValueError("Acoustic manifest rows must contain a path or filename.")

        label_value = _coalesce(row, LABEL, "scene_label", "event_label", "target")
        if require_local_columns and label_value == "":
            raise ValueError("Local acoustic manifest rows must contain a label.")

        audio_path = _resolve_audio_path(audio_path_value, manifest_path, data_dir)
        filename = _coalesce(row, FILENAME, "filename", default=Path(audio_path).name)
        clip_id = _coalesce(row, CLIP_ID, "segment_id", default=Path(filename).stem)

        label = int(label_value) if _looks_int(label_value) else label_value
        records.append(
            {
                PATH: audio_path,
                LABEL: label,
                SPLIT: split,
                CLIP_ID: clip_id,
                SOURCE_ID: _coalesce(row, SOURCE_ID, "recording_id", "device"),
                START_TIME: _float_value(_coalesce(row, START_TIME)),
                END_TIME: _float_value(_coalesce(row, END_TIME)),
                DATASET: dataset,
                EXAMPLE_ID: _coalesce(row, EXAMPLE_ID, default=clip_id),
                FILENAME: filename,
            }
        )
    return records


def _records_to_dataset(records: list[dict]) -> tf.data.Dataset:
    label_dtype = _label_dtype(records)

    def gen():
        for record in records:
            item = dict(record)
            if label_dtype == tf.string:
                item[LABEL] = str(item[LABEL])
            else:
                item[LABEL] = np.int64(item[LABEL])
            yield item

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature={
            PATH: tf.TensorSpec(shape=(), dtype=tf.string),
            LABEL: tf.TensorSpec(shape=(), dtype=label_dtype),
            SPLIT: tf.TensorSpec(shape=(), dtype=tf.string),
            CLIP_ID: tf.TensorSpec(shape=(), dtype=tf.string),
            SOURCE_ID: tf.TensorSpec(shape=(), dtype=tf.string),
            START_TIME: tf.TensorSpec(shape=(), dtype=tf.float32),
            END_TIME: tf.TensorSpec(shape=(), dtype=tf.float32),
            DATASET: tf.TensorSpec(shape=(), dtype=tf.string),
            EXAMPLE_ID: tf.TensorSpec(shape=(), dtype=tf.string),
            FILENAME: tf.TensorSpec(shape=(), dtype=tf.string),
        },
    )

    try:
        ds = ds.apply(tf.data.experimental.assert_cardinality(len(records)))
    except Exception as e:
        logger.warning(f"Failed to assert cardinality for acoustic manifest: {e}")

    return ds


def _load_manifest_splits(
    dataset_name: str,
    splits: list[str],
    data_dir: Union[None, str, os.PathLike],
    *,
    prefix: str,
    require_local_columns: bool,
) -> list[tf.data.Dataset]:
    manifest_path = _resolve_manifest_path(dataset_name, prefix, data_dir)
    rows, columns = _read_manifest(manifest_path)

    if require_local_columns:
        missing = _LOCAL_REQUIRED_COLUMNS - columns
        if missing:
            missing_cols = ", ".join(sorted(missing))
            raise ValueError(f"Local acoustic manifest is missing required columns: {missing_cols}.")

    dataset = manifest_path.stem
    return [
        _records_to_dataset(
            _manifest_records(
                rows,
                manifest_path,
                split,
                data_dir=data_dir,
                dataset=dataset,
                require_local_columns=require_local_columns,
            )
        )
        for split in splits
    ]


@register_source_loader("local_audio:")
def load_local_audio_manifest_splits(
    dataset_name: str,
    splits: list[str],
    data_dir: Union[None, str, os.PathLike] = None,
) -> list[tf.data.Dataset]:
    return _load_manifest_splits(
        dataset_name,
        splits,
        data_dir,
        prefix="local_audio:",
        require_local_columns=True,
    )


@register_source_loader("dcase2025:")
def load_dcase2025_splits(
    dataset_name: str,
    splits: list[str],
    data_dir: Union[None, str, os.PathLike] = None,
) -> list[tf.data.Dataset]:
    return _load_manifest_splits(
        dataset_name,
        splits,
        data_dir,
        prefix="dcase2025:",
        require_local_columns=False,
    )


def _import_datasets():
    try:
        import datasets
    except ImportError as e:
        raise ImportError(
            "The 'datasets' package is required for hf_audio: source loading. "
            "Install justdata with the acoustic extra."
        ) from e
    return datasets


def _hf_label_signature(value) -> tuple[tf.DType, tuple | None]:
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value)
        if np.issubdtype(arr.dtype, np.integer):
            return tf.int64, (None,)
        if np.issubdtype(arr.dtype, np.floating):
            return tf.float32, (None,)
        return tf.string, (None,)
    if isinstance(value, (int, np.integer, bool)):
        return tf.int64, ()
    if isinstance(value, (float, np.floating)):
        return tf.float32, ()
    return tf.string, ()


def _normalize_label(value, dtype: tf.DType):
    if dtype == tf.int64:
        return np.asarray(value, dtype=np.int64)
    if dtype == tf.float32:
        return np.asarray(value, dtype=np.float32)
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=str)
    return str(value)


def _as_waveform_np(audio_value, *, decode_mode: str, fallback_sample_rate: int | None):
    sample_rate = fallback_sample_rate
    path = None

    if isinstance(audio_value, dict):
        path = audio_value.get("path")
        sample_rate = audio_value.get("sampling_rate", audio_value.get(SAMPLE_RATE, sample_rate))
        array = audio_value.get("array")
        if decode_mode == "justdata" and path:
            waveform, decoded_sample_rate = decode_audio_file(path)
            sample_rate = int(decoded_sample_rate.numpy())
            array = waveform.numpy()
    else:
        array = None
        if isinstance(audio_value, (str, bytes, os.PathLike)):
            path = os.fspath(audio_value)

    if array is None and path:
        waveform, decoded_sample_rate = decode_audio_file(path)
        sample_rate = int(decoded_sample_rate.numpy())
        array = waveform.numpy()

    if array is None:
        array = audio_value

    if sample_rate is None:
        raise ValueError("Hugging Face audio records must provide a sampling_rate.")

    tensor = tf.convert_to_tensor(array)
    tensor = to_float32_waveform(tensor, input_dtype=tensor.dtype)
    tensor = standardize_waveform_layout(tensor)
    return tensor.numpy().astype(np.float32), np.int32(sample_rate), path


@register_source_loader("hf_audio:")
def load_huggingface_audio_splits(
    dataset_name: str,
    splits: list[str],
    data_dir: Union[None, str, os.PathLike] = None,
    *,
    decode_mode: Literal["hf_native", "justdata"] = "hf_native",
    hf_audio_sampling_rate: int | None = None,
    audio_column: str = "audio",
    label_column: str | None = "label",
) -> list[tf.data.Dataset]:
    if decode_mode not in {"hf_native", "justdata"}:
        raise ValueError("decode_mode must be 'hf_native' or 'justdata'.")

    datasets = _import_datasets()
    hf_name = _strip_prefix(dataset_name, "hf_audio:")
    loaded_splits = []

    for split in splits:
        ds_hf = datasets.load_dataset(
            hf_name,
            split=split,
            cache_dir=str(data_dir) if data_dir else None,
        )

        if audio_column not in ds_hf.column_names:
            raise ValueError(
                f"Hugging Face audio dataset '{hf_name}' must contain audio column "
                f"{audio_column!r}. Found columns: {ds_hf.column_names}."
            )
        if label_column is not None and label_column not in ds_hf.column_names:
            raise ValueError(
                f"Hugging Face audio dataset '{hf_name}' must contain label column "
                f"{label_column!r}. Found columns: {ds_hf.column_names}."
            )

        if decode_mode == "hf_native" and hf_audio_sampling_rate is not None:
            ds_hf = ds_hf.cast_column(
                audio_column,
                datasets.Audio(sampling_rate=hf_audio_sampling_rate),
            )

        first = next(iter(ds_hf), None)
        label_dtype = None
        label_shape = None
        if label_column is not None:
            if first is None:
                label_dtype, label_shape = tf.int64, ()
            else:
                label_dtype, label_shape = _hf_label_signature(first[label_column])

        def gen(_ds=ds_hf, _split=split, _label_dtype=label_dtype):
            for idx, sample in enumerate(_ds):
                waveform, sample_rate, path = _as_waveform_np(
                    sample[audio_column],
                    decode_mode=decode_mode,
                    fallback_sample_rate=hf_audio_sampling_rate,
                )
                filename = Path(path).name if path else ""
                item = {
                    WAVEFORM: waveform,
                    SAMPLE_RATE: sample_rate,
                    DATASET: hf_name,
                    SPLIT: _split,
                    EXAMPLE_ID: str(sample.get("id", idx)),
                    FILENAME: filename,
                }
                if label_column is not None:
                    item[LABEL] = _normalize_label(sample[label_column], _label_dtype)
                yield item

        output_signature = {
            WAVEFORM: tf.TensorSpec(shape=(None, None), dtype=tf.float32),
            SAMPLE_RATE: tf.TensorSpec(shape=(), dtype=tf.int32),
            DATASET: tf.TensorSpec(shape=(), dtype=tf.string),
            SPLIT: tf.TensorSpec(shape=(), dtype=tf.string),
            EXAMPLE_ID: tf.TensorSpec(shape=(), dtype=tf.string),
            FILENAME: tf.TensorSpec(shape=(), dtype=tf.string),
        }
        if label_column is not None:
            output_signature[LABEL] = tf.TensorSpec(shape=label_shape, dtype=label_dtype)

        ds_tf = tf.data.Dataset.from_generator(gen, output_signature=output_signature)

        try:
            ds_tf = ds_tf.apply(tf.data.experimental.assert_cardinality(len(ds_hf)))
        except Exception as e:
            logger.warning(f"Failed to assert cardinality for HF audio dataset {hf_name}: {e}")

        loaded_splits.append(ds_tf)

    return loaded_splits


__all__ = [
    "load_dcase2025_splits",
    "load_huggingface_audio_splits",
    "load_local_audio_manifest_splits",
]

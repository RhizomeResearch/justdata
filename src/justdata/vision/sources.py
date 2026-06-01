import os
from dataclasses import dataclass
from typing import Any, Union
from urllib.parse import parse_qsl

import numpy as np
import tensorflow as tf
from loguru import logger

from justdata.core.sources import register_source_loader

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
_WILDS_QUERY_PARAMS = {"download", "split_scheme", "unlabeled", "version"}


@dataclass(frozen=True)
class _WILDSDatasetSpec:
    name: str
    split_scheme: str
    version: str | None
    download: bool
    unlabeled: bool


def _import_datasets():
    try:
        import datasets
    except ImportError as e:
        raise ImportError(
            "The 'datasets' package is required for hf: vision source loading. "
            "Install justdata with the vision extra."
        ) from e
    return datasets


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
            cache_dir=str(data_dir) if data_dir else None,
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


def _split_to_dataset(
    dataset,
    split: str,
    *,
    spec: _WILDSDatasetSpec,
    include_label: bool,
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
    if data_dir is not None:
        kwargs["root_dir"] = str(data_dir)
    if spec.version is not None:
        kwargs["version"] = spec.version
    if spec.unlabeled or uses_unlabeled:
        kwargs["unlabeled"] = True

    dataset = wilds.get_dataset(**kwargs)
    include_label = not (spec.unlabeled or uses_unlabeled)
    return [
        _split_to_dataset(
            dataset,
            split,
            spec=spec,
            include_label=include_label,
        )
        for split in splits
    ]

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

import tensorflow as tf
from loguru import logger

from justdata.core.adapters import get_adapter
from justdata.core.sources import get_source_loader


def _concatenate_tf_datasets(
    dataset_list: list[tf.data.Dataset],
) -> Optional[tf.data.Dataset]:
    """Concatenates a list of tf.data.Dataset objects sequentially."""
    if not dataset_list:
        return None

    concatenated_ds = dataset_list[0]
    for i in range(1, len(dataset_list)):
        concatenated_ds = concatenated_ds.concatenate(dataset_list[i])

    return concatenated_ds


def fetch_ds(
    dataset_names: list[str],
    splits_info: Union[list[str], Dict[str, list[str]]],
    data_dir: Union[None, str, os.PathLike] = None,
) -> Optional[tf.data.Dataset]:
    """
    Loads and concatenates multiple TensorFlow Datasets and their splits.

    This function performs a two-level concatenation:
    1. For each dataset name in `dataset_names`:
       - If `splits_info` is a list, those splits are loaded.
       - If `splits_info` is a dictionary, the splits for the current
         dataset name are looked up in the dictionary. If not found or
         empty, the dataset is skipped.
       All successfully loaded splits for the current dataset are concatenated.
    2. The resulting datasets (one for each successfully processed name in
       `dataset_names`, now containing its specified splits) are then
       concatenated into a single, final `tf.data.Dataset`.

    Args:
        dataset_names: A list of dataset names to attempt to load
                       (e.g., ['cifar10', 'mnist']). The order is preserved.
        splits_info: Defines the splits to load.
            - If a List[str] (e.g., ['train', 'test']): these splits are
              loaded for every dataset in `dataset_names`. If the list is
              empty, no data will be loaded for any dataset.
            - If a Dict[str, List[str]] (e.g.,
              {'cifar10': ['train'], 'mnist': ['train', 'test']}):
              For each dataset name in `dataset_names` that is also a key
              in this dict, the corresponding list of splits is loaded.
              Datasets in `dataset_names` not found as keys, or with an
              empty list of splits in the dict, will be skipped.
        data_dir: Optional path to the directory where datasets are stored or
                  will be downloaded. If None, defaults to
                  `tfds.builder.DEFAULT_DATA_DIR`.

    Returns:
        A `tf.data.Dataset` instance representing the concatenation of all
        specified and successfully loaded datasets and splits. Returns `None`
        if `dataset_names` is empty, if `splits_info` leads to no loadable
        splits for any dataset, or if all loading attempts fail.

    Note:
        - `tfds.load` is called with `as_supervised=False` and
          `shuffle_files=False`. These parameters are fixed internally.
        - If loading splits for a particular dataset name fails (e.g., due
          to an invalid name or unavailable data), a warning is logged,
          and that dataset is skipped. The function will attempt to proceed
          with other dataset names.
    """
    if not dataset_names:
        logger.info("Empty 'dataset_names' list provided. Nothing to load.")
        return None

    datasets_with_splits = []

    for dataset_name in dataset_names:
        splits_to_load: Optional[list[str]] = None

        if isinstance(splits_info, list):
            if not splits_info:
                logger.warning(
                    f"Global 'splits_info' list is empty. "
                    f"Skipping dataset '{dataset_name}'."
                )
                continue
            splits_to_load = splits_info
        elif isinstance(splits_info, dict):
            if dataset_name not in splits_info or not splits_info[dataset_name]:
                logger.warning(
                    f"Dataset '{dataset_name}' not found in 'splits_info' "
                    f"dict, or its split list is empty. Skipping."
                )
                continue
            splits_to_load = splits_info[dataset_name]
        else:
            logger.error(
                "Invalid type for 'splits_info'. Must be List[str] or "
                "Dict[str, List[str]]. Cannot proceed."
            )
            return None

        try:
            loaded_splits = get_source_loader(dataset_name)(
                dataset_name,
                splits_to_load,
                data_dir,
            )
        except Exception as e:
            logger.warning(
                f"Warning: Failed to load splits for dataset '{dataset_name}'. "
                f"Error: {e}. Skipping this dataset."
            )
            continue

        if not loaded_splits:
            logger.warning(
                f"No splits data resulted for dataset '{dataset_name}' "
                f"after attempting to load. Skipping."
            )
            continue

        concatenated_splits = _concatenate_tf_datasets(loaded_splits)

        if concatenated_splits:
            adapter = get_adapter(dataset_name)
            concatenated_splits = concatenated_splits.map(
                adapter, num_parallel_calls=tf.data.AUTOTUNE
            )
            datasets_with_splits.append(concatenated_splits)

    if not datasets_with_splits:
        logger.info(
            "No datasets were successfully loaded and processed "
            "based on the provided names and splits configuration."
        )
        return None

    return _concatenate_tf_datasets(datasets_with_splits)


def _pad_dataset(ds, batch_size, *, metadata_mode: str = "full"):
    """
    Applies padding to a batched dataset.
    If the last batch is smaller than batch_size, it pads it with zeros
    and adds a 'padding_mask' key.
    """

    def _first_tensor(value):
        if isinstance(value, dict):
            for child in value.values():
                found = _first_tensor(child)
                if found is not None:
                    return found
            return None
        return value

    def _get_batch_dim(batch):
        """Get current batch size from the first tensor in the batch."""
        for v in batch.values():
            tensor = _first_tensor(v)
            if tensor is not None:
                return tf.shape(tensor)[0]
        return tf.constant(0, dtype=tf.int32)

    def _pad_value(value, pad_size):
        if isinstance(value, dict):
            padded = {}
            for k, v in value.items():
                child = _pad_value(v, pad_size)
                if child is not None:
                    padded[k] = child
            return padded

        if metadata_mode != "full" and value.dtype == tf.string:
            return None

        pad_shape = tf.concat([[pad_size], tf.shape(value)[1:]], axis=0)
        if value.dtype == tf.string:
            fill = tf.fill(pad_shape, tf.constant("", dtype=tf.string))
        else:
            fill = tf.zeros(pad_shape, dtype=value.dtype)
        return tf.concat([value, fill], axis=0)

    def pad_batch(batch):
        curr_size = _get_batch_dim(batch)
        pad_size = batch_size - curr_size

        mask = tf.concat(
            [
                tf.ones((curr_size,), dtype=tf.bool),
                tf.zeros((pad_size,), dtype=tf.bool),
            ],
            axis=0,
        )

        padded_batch = {}
        for k, v in batch.items():
            child = _pad_value(v, pad_size)
            if child is not None:
                padded_batch[k] = child

        padded_batch["padding_mask"] = mask
        return padded_batch

    return ds.map(
        lambda b: tf.cond(
            _get_batch_dim(b) < batch_size,
            lambda: pad_batch(b),
            lambda: b | {"padding_mask": tf.ones((batch_size,), dtype=tf.bool)},
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )


def _is_numeric_metadata_value(value: Any) -> bool:
    return hasattr(value, "dtype") and value.dtype != tf.string


def _numeric_metadata(metadata: dict) -> dict:
    numeric = {}
    for key, value in metadata.items():
        if isinstance(value, dict):
            child = _numeric_metadata(value)
            if child:
                numeric[key] = child
        elif _is_numeric_metadata_value(value):
            numeric[key] = value
    return numeric


def _apply_metadata_mode(sample: dict, metadata_mode: str) -> dict:
    if metadata_mode == "full" or "metadata" not in sample:
        return sample

    result = dict(sample)
    if metadata_mode == "none":
        result.pop("metadata", None)
        return result

    metadata = result.get("metadata")
    if isinstance(metadata, dict):
        metadata = _numeric_metadata(metadata)
        if metadata:
            result["metadata"] = metadata
        else:
            result.pop("metadata", None)
    return result


def _flatten_metadata(metadata: dict, prefix: tuple[str, ...] = ()):
    leaves = []
    for key, value in metadata.items():
        path = prefix + (key,)
        if isinstance(value, dict):
            leaves.extend(_flatten_metadata(value, path))
        else:
            leaves.append((path, value))
    return leaves


def _identity_structure(value):
    if isinstance(value, dict):
        return {k: _identity_structure(v) for k, v in value.items()}
    return tf.identity(value)


def _python_json_value(value: Any) -> Any:
    if hasattr(value, "numpy"):
        value = value.numpy()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "tolist"):
        return _python_json_value(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_python_json_value(child) for child in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _stable_int64_hash(value: Any) -> int:
    value = _python_json_value(value)
    if value is None:
        value = ""
    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False) & ((1 << 63) - 1)


def _assign_nested(target: dict, path: tuple[str, ...], value: Any) -> None:
    current = target
    for key in path[:-1]:
        current = current.setdefault(key, {})
    current[path[-1]] = value


def _sidecar_example_id(values_by_path: dict[tuple[str, ...], Any]) -> int:
    by_name = {path[-1]: value for path, value in values_by_path.items()}
    if {"dataset", "split", "clip_id"}.issubset(by_name):
        return _stable_int64_hash(
            f"{_python_json_value(by_name['dataset'])}::"
            f"{_python_json_value(by_name['split'])}::"
            f"{_python_json_value(by_name['clip_id'])}"
        )

    example_id = by_name.get("example_id")
    value = _python_json_value(example_id)
    if isinstance(value, int):
        return value
    return _stable_int64_hash(value)


def _attach_sidecar_writer(ds, path: str):
    output = Path(path)
    if output.parent != Path("."):
        output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("", encoding="utf-8")
    lock = threading.Lock()

    def add_writer(sample):
        metadata = sample.get("metadata")
        if not isinstance(metadata, dict):
            return sample

        leaves = _flatten_metadata(metadata)
        string_leaves = [
            (path, value) for path, value in leaves if value.dtype == tf.string
        ]
        if not string_leaves:
            return sample

        all_paths = [path for path, _ in leaves]
        all_values = [value for _, value in leaves]
        string_paths = {path for path, _ in string_leaves}

        def write_record(*values):
            values_by_path = {
                path: _python_json_value(value)
                for path, value in zip(all_paths, values)
            }
            string_metadata = {}
            for path in string_paths:
                _assign_nested(string_metadata, path, values_by_path[path])

            record = {
                "example_id": _sidecar_example_id(values_by_path),
                "metadata": string_metadata,
            }
            with lock:
                with output.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
            return 0

        marker = tf.py_function(write_record, all_values, Tout=tf.int64)
        with tf.control_dependencies([marker]):
            return _identity_structure(sample)

    return ds.map(add_writer, num_parallel_calls=tf.data.AUTOTUNE)


def load_ds(
    dataset_names_arg: Union[str, list[str]],
    splits_arg: Union[str, list[str], Dict[str, list[str]]],
    dataset_type: str,
    batch_size: int,
    seed: int,
    num_classes: Optional[int] = None,
    pipeline: Optional[Any] = None,
    preprocess_fn=None,
    augment_fn=None,
    late_augment_fn=None,
    postprocess_fn=None,
    shuffle_buffer: int = 10_000,
    cache_dataset: bool = True,
    cache_path: str = "",
    drop_remainder: bool = False,
    data_dir: Union[None, str, os.PathLike] = None,
    return_raw_ds: bool = False,
    deterministic: bool = False,
    as_numpy: bool = False,
    metadata_mode: Literal["full", "numeric_only", "none"] = "full",
    sidecar_metadata_path: str | None = None,
    filter_fn=None,
):
    """
    Loads, preprocesses, and batches HuggingFace or TensorFlow Datasets.

    Args:
        dataset_names_arg: A single dataset name (str) or a list of dataset names.
        splits_arg: Defines the splits to load.
            - If str (e.g., "train"): This split is loaded for all datasets.
            - If List[str] (e.g., ["train", "test"]): These splits are loaded
              for all datasets.
            - If Dict[str, List[str]] (e.g., {"cifar10": ["train"]}): Specifies
              splits per dataset. See `fetch_ds` for details.
        dataset_type: Type of dataset, typically "train", "validation", or "test".
                      Affects caching, shuffling, and augmentation.
        batch_size: The batch size for the output dataset.
        seed: Random seed for shuffling and augmentations.
        num_classes: Number of classes (needed for one-hot encoding/mixup).
        pipeline: A `DataPipeline` object from `justdata.core.registry`. If provided,
                  it builds the necessary functions based on `dataset_type`.
        preprocess_fn: Preprocessing function (fallback if `pipeline` is None).
        augment_fn: Augmentation function (fallback if `pipeline` is None).
        late_augment_fn: Late augmentation function (fallback if `pipeline` is None).
        postprocess_fn: Postprocessing function (fallback if `pipeline` is None).
        shuffle_buffer: Size of the shuffle buffer.
        cache_dataset: Whether to cache the dataset after preprocessing.
        cache_path: The name of a directory on the filesystem to use for caching
            elements in this Dataset.
            If a filename is not provided, the dataset will be cached in memory.
        drop_remainder: Choose to drop or pad batches without the correct size,
        data_dir: Optional path to the TFDS data directory.
        return_raw_ds: If True, returns the dataset immediately after
                       preprocessing (and caching) but BEFORE standard
                       augmentation, postprocessing, or batching.
                       Returns (ds, tools_dict) where tools_dict contains
                       ``postprocess_fn`` and ``rng``.
        deterministic: If false, sacrifices determinism for performance.
        as_numpy: If True, returns an iterator yielding NumPy arrays.
        metadata_mode: Controls metadata in output batches. ``full`` keeps all
                       metadata, ``numeric_only`` drops strings from batches,
                       and ``none`` removes metadata.
        sidecar_metadata_path: JSONL path for string metadata when
                               ``metadata_mode="numeric_only"``.
        filter_fn: Optional predicate applied after preprocessing and caching,
                   before augmentation, postprocessing, and batching.

    Returns:
        A batched `tf.data.Dataset` converted to NumPy arrays.

    Raises:
        ValueError: If dataset loading fails and `fetch_ds` returns None.
        TypeError: If `dataset_names_arg` or `splits_arg` have invalid types.
    """
    rng = tf.random.Generator.from_seed(seed)
    if metadata_mode not in {"full", "numeric_only", "none"}:
        raise ValueError(
            "metadata_mode must be one of 'full', 'numeric_only', or 'none'."
        )

    dataset_names: list[str]
    if isinstance(dataset_names_arg, str):
        dataset_names = [dataset_names_arg]
    elif isinstance(dataset_names_arg, list):
        dataset_names = dataset_names_arg
    else:
        raise TypeError("dataset_names_arg must be a string or a list of strings.")

    splits: Union[list[str], Dict[str, list[str]]]
    if isinstance(splits_arg, str):
        splits = [splits_arg]
    elif isinstance(splits_arg, list):
        splits = splits_arg
    elif isinstance(splits_arg, dict):
        splits = splits_arg
    else:
        raise TypeError(
            "splits_arg must be a string, a list of strings, or a "
            "dictionary mapping dataset names to lists of strings."
        )

    is_training = dataset_type == "train"

    ds = fetch_ds(dataset_names, splits, data_dir)

    if ds is None:
        raise ValueError(
            "Dataset loading failed (fetch_ds returned None). "
            "Check logs for details on dataset names, splits, or data issues."
        )

    if pipeline is not None:
        preprocess_fn, augment_fn, late_augment_fn, postprocess_fn = pipeline.build(
            is_training=is_training
        )

    def seeded_augment(sample):
        seed = rng.make_seeds(1)[:, 0]
        return augment_fn(sample, seed=seed)

    def seeded_late_augment(batch):
        seed = rng.make_seeds(1)[:, 0]
        return late_augment_fn(batch, num_classes=num_classes, seed=seed)

    ds = ds.map(
        preprocess_fn,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=deterministic if is_training else None,
    )

    # For big datasets or datasets with big images, caching can put your RAM on
    # fire and destroy your computer
    if cache_dataset:
        ds = ds.cache(cache_path)
    else:
        logger.info(
            f"Caching disabled for '{dataset_type}' dataset. "
            "Samples will be re-read each epoch (streaming-friendly)."
        )

    if filter_fn is not None:
        ds = ds.filter(filter_fn)

    if return_raw_ds:
        # Return necessary components to build custom pipelines
        return (ds, {"postprocess_fn": postprocess_fn, "rng": rng})

    if is_training:
        ds = ds.map(
            seeded_augment,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=deterministic,
        )
        ds = ds.shuffle(shuffle_buffer, seed=seed)

    ds = ds.map(
        lambda x: postprocess_fn(x, num_classes=num_classes),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=deterministic if is_training else None,
    )
    if metadata_mode == "numeric_only" and sidecar_metadata_path is not None:
        ds = _attach_sidecar_writer(ds, sidecar_metadata_path)
    if metadata_mode != "full":
        ds = ds.map(
            lambda x: _apply_metadata_mode(x, metadata_mode),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=deterministic if is_training else None,
        )
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)

    # Ensure padding_mask is always present for API consistency
    if drop_remainder:
        ds = ds.map(
            lambda b: b | {"padding_mask": tf.ones((batch_size,), dtype=tf.bool)},
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=deterministic if is_training else None,
        )
    else:
        ds = _pad_dataset(ds, batch_size, metadata_mode=metadata_mode)

    if is_training:
        ds = ds.map(
            seeded_late_augment,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=deterministic,
        )

    ds = ds.prefetch(tf.data.AUTOTUNE)

    N = tf.data.Dataset.cardinality(ds)
    if as_numpy:
        return ds.as_numpy_iterator(), N
    return ds, N

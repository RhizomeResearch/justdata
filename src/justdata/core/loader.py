import os
from functools import partial
from typing import Any, Dict, Literal, Optional, Union

import tensorflow as tf
from loguru import logger

from justdata.core.adapters import get_adapter
from justdata.core.finalization import finalize_dataset
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
    cache_model_inputs: bool = False,
    model_input_cache_path: str = "",
    allow_train_model_input_cache: bool = False,
    drop_remainder: bool = False,
    data_dir: Union[None, str, os.PathLike] = None,
    return_raw_ds: bool = False,
    deterministic: bool = False,
    as_numpy: bool = False,
    metadata_mode: Literal["full", "numeric_only", "none"] | None = None,
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
        cache_model_inputs: Whether to cache samples after deterministic
            postprocessing, before batching. Use ``model_input_cache_path`` for
            local SSD caching of resized/model-ready tensors.
        model_input_cache_path: Filesystem path for ``cache_model_inputs``.
            If a filename is not provided, model inputs will be cached in memory.
        allow_train_model_input_cache: Required opt-in for model-input caching
            with ``dataset_type="train"``. Only use this for deterministic
            training views, because stochastic augmentations will be materialized
            into the cache on first fill.
        drop_remainder: Choose to drop or pad batches without the correct size,
        data_dir: Optional path to the TFDS data directory.
        return_raw_ds: If True, returns the dataset immediately after
                       preprocessing (and caching) but BEFORE standard
                       augmentation, postprocessing, or batching.
                       Returns (ds, tools_dict) where tools_dict contains
                       ``finalize_fn``, ``postprocess_fn``, and ``rng``.
        deterministic: If false, sacrifices determinism for performance.
        as_numpy: If True, returns an iterator yielding NumPy arrays.
        metadata_mode: Controls metadata in output batches. When omitted, uses
                       ``pipeline.kwargs["metadata_mode"]`` if present, then
                       defaults to ``full``. ``numeric_only`` drops strings
                       from batches and ``none`` removes metadata.
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
    if metadata_mode is None:
        metadata_mode = (
            pipeline.kwargs.get("metadata_mode", "full")
            if pipeline is not None
            else "full"
        )
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
    if cache_model_inputs and is_training and not allow_train_model_input_cache:
        raise ValueError(
            "cache_model_inputs with dataset_type='train' requires "
            "allow_train_model_input_cache=True."
        )
    if cache_dataset and cache_model_inputs and cache_path and model_input_cache_path:
        preprocess_cache_path = os.path.abspath(os.fspath(cache_path))
        model_cache_path = os.path.abspath(os.fspath(model_input_cache_path))
        if preprocess_cache_path == model_cache_path:
            raise ValueError(
                "cache_path and model_input_cache_path must be different when "
                "both cache stages are enabled."
            )

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

    finalize_fn = partial(
        finalize_dataset,
        postprocess_fn=postprocess_fn,
        num_classes=num_classes,
        batch_size=batch_size,
        metadata_mode=metadata_mode,
        sidecar_metadata_path=sidecar_metadata_path,
        cache_model_inputs=cache_model_inputs,
        model_input_cache_path=model_input_cache_path,
        drop_remainder=drop_remainder,
        is_training=is_training,
        late_augment_fn=late_augment_fn,
        rng=rng,
        deterministic=deterministic,
        shuffle_buffer=shuffle_buffer if is_training else None,
        shuffle_seed=seed,
        as_numpy=as_numpy,
    )

    if return_raw_ds:
        # Return necessary components to build custom pipelines
        return (
            ds,
            {
                "finalize_fn": finalize_fn,
                "postprocess_fn": postprocess_fn,
                "rng": rng,
            },
        )

    if is_training:
        ds = ds.map(
            seeded_augment,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=deterministic,
        )

    return finalize_fn(ds)

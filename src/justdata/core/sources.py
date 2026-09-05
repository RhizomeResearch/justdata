import os
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Optional, Protocol, Union

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


class SourceLoader(Protocol):
    def __call__(
        self,
        dataset_name: str,
        splits: list[str],
        data_dir: Union[None, str, os.PathLike] = None,
    ) -> list[tf.data.Dataset]: ...


_SOURCE_LOCK = threading.Lock()
_PREFIX_LOADERS: dict[str, SourceLoader] = {}
_DEFAULT_LOADER: Optional[SourceLoader] = None


def _dataset_from_materialized_records(
    generator: Callable[[], Iterator[dict]],
    *,
    output_signature: dict,
) -> tf.data.Dataset:
    """Tensorize an existing finite snapshot of scalar records, without media I/O."""
    specs = tf.nest.flatten(output_signature)
    columns = [[] for _ in specs]
    try:
        for record in generator():
            tf.nest.assert_same_structure(output_signature, record)
            for column, value in zip(columns, tf.nest.flatten(record)):
                column.append(value)
        tensors = []
        for column, spec in zip(columns, specs):
            values = np.asarray(column, dtype=spec.dtype.as_numpy_dtype)
            tensor = tf.convert_to_tensor(values, dtype=spec.dtype)
            tensor.set_shape([None, *spec.shape.as_list()])
            tensors.append(tensor)
    except (
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        tf.errors.InvalidArgumentError,
    ):
        # Preserve the generator's iteration-time errors for malformed records.
        return tf.data.Dataset.from_generator(
            generator, output_signature=output_signature
        )
    return tf.data.Dataset.from_tensor_slices(
        tf.nest.pack_sequence_as(output_signature, tensors)
    )


def source_cache_dir(
    data_dir: Union[None, str, os.PathLike],
    *parts: str,
) -> Path:
    root = (
        Path(data_dir).expanduser()
        if data_dir is not None
        else Path.home() / ".cache" / "justdata"
    )
    return root.joinpath(*parts)


def register_source_loader(prefix: str):
    def decorator(fn: SourceLoader) -> SourceLoader:
        with _SOURCE_LOCK:
            if prefix in _PREFIX_LOADERS:
                raise ValueError(f"Source loader for prefix '{prefix}' already exists.")
            _PREFIX_LOADERS[prefix] = fn
        return fn

    return decorator


def register_default_source_loader(fn: SourceLoader) -> SourceLoader:
    global _DEFAULT_LOADER
    with _SOURCE_LOCK:
        if _DEFAULT_LOADER is not None:
            raise ValueError("Default source loader already exists.")
        _DEFAULT_LOADER = fn
    return fn


def get_source_loader(dataset_name: str) -> SourceLoader:
    with _SOURCE_LOCK:
        prefix_loaders = tuple(_PREFIX_LOADERS.items())
        default_loader = _DEFAULT_LOADER

    for prefix, loader in sorted(
        prefix_loaders, key=lambda kv: len(kv[0]), reverse=True
    ):
        if dataset_name.startswith(prefix):
            return loader

    if default_loader is None:
        raise ValueError("No default source loader registered.")
    return default_loader


@register_default_source_loader
def load_tfds_splits(
    dataset_name: str,
    splits: list[str],
    data_dir: Union[None, str, os.PathLike] = None,
) -> list[tf.data.Dataset]:
    return [
        tfds.load(
            dataset_name,
            split=s,
            as_supervised=False,
            shuffle_files=False,
            data_dir=os.fspath(source_cache_dir(data_dir, "tfds")),
        )
        for s in splits
    ]

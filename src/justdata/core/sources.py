import os
import threading
from typing import Optional, Protocol, Union

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
    for prefix, loader in sorted(
        _PREFIX_LOADERS.items(), key=lambda kv: len(kv[0]), reverse=True
    ):
        if dataset_name.startswith(prefix):
            return loader

    if _DEFAULT_LOADER is None:
        raise ValueError("No default source loader registered.")
    return _DEFAULT_LOADER


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
            data_dir=data_dir,
        )
        for s in splits
    ]

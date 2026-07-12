from typing import Union

import tensorflow as tf

from justdata.core.loader import load_ds
from justdata.vision.corruptions.registry import (
    _metadata_with_corruption,
    apply_minic_corruption,
)


_MAX_SEED = tf.constant(2**31 - 1, dtype=tf.int64)


def create_minic_datasets(
    corruption_types: Union[str, list[str]], severity: int = 3, **load_ds_kwargs
) -> Union[tuple[tf.data.Dataset, int], tuple[list[tf.data.Dataset], int]]:
    """
    Create Mini-C corruption benchmark datasets from a preprocessed base dataset.
    """

    load_ds_kwargs["return_raw_ds"] = True

    ds, tools = load_ds(**load_ds_kwargs)

    finalize_fn = tools["finalize_fn"]
    seed = load_ds_kwargs.get("seed", 0)

    return_list = isinstance(corruption_types, list)
    c_list = corruption_types if return_list else [corruption_types]

    datasets_out = []

    for c_index, c_name in enumerate(c_list):

        def corrupt_fn(index, sample, c_name=c_name, c_index=c_index):
            s_seed = _seed_from_index(seed, index, salt=c_index * 1_000_003)
            img = sample["image"]
            img_corrupted = apply_minic_corruption(img, c_name, severity, s_seed)
            return _metadata_with_corruption(
                sample | {"image": img_corrupted},
                name=c_name,
                severity=severity,
                domain="image",
            )

        ds_c = ds.enumerate().map(
            corrupt_fn,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True,
        )
        ds_c, n_batches = finalize_fn(ds_c)
        datasets_out.append(ds_c)

    if not datasets_out:
        n_batches = 0

    if return_list:
        return datasets_out, n_batches
    return datasets_out[0], n_batches


def _seed_from_index(
    seed: int | tf.Tensor, index: tf.Tensor, salt: int = 0
) -> tf.Tensor:
    base = tf.cast(seed, tf.int64)
    idx = tf.cast(index, tf.int64)
    salted = tf.math.floormod(base * 1_103_515_245 + idx + salt, _MAX_SEED)
    return tf.cast(tf.stack([tf.math.floormod(base, _MAX_SEED), salted]), tf.int32)

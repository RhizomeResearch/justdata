from typing import Union

import tensorflow as tf

from justdata.core.loader import _pad_dataset, load_ds
from justdata.vision.corruptions.registry import apply_minic_corruption


def create_minic_datasets(
    corruption_types: Union[str, list[str]], severity: int = 3, **load_ds_kwargs
) -> Union[tuple[tf.data.Dataset, int], tuple[list[tf.data.Dataset], int]]:
    """
    Create Mini-C corruption benchmark datasets from a preprocessed base dataset.
    """

    load_ds_kwargs["return_raw_ds"] = True

    ds, tools = load_ds(**load_ds_kwargs)

    postprocess_fn = tools["postprocess_fn"]
    rng = tools["rng"]

    batch_size = load_ds_kwargs.get("batch_size", 32)
    num_classes = load_ds_kwargs.get("num_classes")
    drop_remainder = load_ds_kwargs.get("drop_remainder", False)

    return_list = isinstance(corruption_types, list)
    c_list = corruption_types if return_list else [corruption_types]

    datasets_out = []

    for c_name in c_list:

        def corrupt_fn(sample, c_name=c_name):
            s_seed = rng.make_seeds(1)[:, 0]
            img = sample["image"]
            img_corrupted = apply_minic_corruption(img, c_name, severity, s_seed)
            return sample | {"image": img_corrupted}

        ds_c = ds.map(corrupt_fn, num_parallel_calls=tf.data.AUTOTUNE)
        ds_c = ds_c.map(
            lambda x: postprocess_fn(x, num_classes=num_classes),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        ds_c = ds_c.batch(batch_size, drop_remainder=drop_remainder)

        if drop_remainder:
            ds_c = ds_c.map(
                lambda b: b | {"padding_mask": tf.ones((batch_size,), dtype=tf.bool)},
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        else:
            ds_c = _pad_dataset(ds_c, batch_size)

        ds_c = ds_c.prefetch(tf.data.AUTOTUNE)
        datasets_out.append(ds_c)

    if datasets_out:
        n_batches = tf.data.Dataset.cardinality(datasets_out[0])
    else:
        n_batches = 0

    if return_list:
        return datasets_out, n_batches
    return datasets_out[0], n_batches

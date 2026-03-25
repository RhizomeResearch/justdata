import os
from typing import Union

import datasets
import numpy as np
import tensorflow as tf
from loguru import logger

from justdata.core.sources import register_source_loader


@register_source_loader("hf:")
def load_huggingface_vision_splits(
    dataset_name: str,
    splits: list[str],
    data_dir: Union[None, str, os.PathLike] = None,
) -> list[tf.data.Dataset]:
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

        def gen(_ds=ds_hf):
            for batch in _ds.iter(batch_size=1024):
                for img, lbl in zip(batch["image"], batch["label"]):
                    yield {
                        "image": np.array(img),
                        "label": lbl,
                    }

        ds_tf = tf.data.Dataset.from_generator(
            gen,
            output_signature={
                "image": tf.TensorSpec(shape=None, dtype=tf.uint8),
                "label": tf.TensorSpec(shape=(), dtype=tf.int64),
            },
        )

        try:
            ds_tf = ds_tf.apply(tf.data.experimental.assert_cardinality(len(ds_hf)))
        except Exception as e:
            logger.warning(f"Failed to assert cardinality for HF dataset {hf_name}: {e}")

        loaded_splits.append(ds_tf)

    return loaded_splits

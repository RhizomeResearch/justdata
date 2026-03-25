from unittest.mock import patch

import pytest
import tensorflow as tf

from justdata.loader import create_minic_datasets, load_ds
from justdata.registry import get_pipeline_for_dataset


def test_load_ds_classification(synthetic_classification_ds):
    preproc, aug, laug, postproc = get_pipeline_for_dataset(
        "cifar10",
        task_type="classification",
        apply_presets=False,
        is_training=True,
        aug_kwargs={"image_size": 32},
        postproc_kwargs={"image_size": 32, "num_classes": 10},
    )

    with patch("justdata.loader.fetch_ds", return_value=synthetic_classification_ds):
        ds, N = load_ds(
            dataset_names_arg="mock",
            splits_arg="train",
            dataset_type="train",
            batch_size=4,
            seed=42,
            preprocess_fn=preproc,
            augment_fn=aug,
            late_augment_fn=laug,
            postprocess_fn=postproc,
            num_classes=10,
            shuffle_buffer=10,
            cache_dataset=False,
            drop_remainder=False,
        )

        assert N == 5
        batch = next(iter(ds))
        assert "image" in batch
        assert "label" in batch
        assert batch["image"].shape == (4, 3, 32, 32)
        assert batch["image"].dtype == tf.float32


def test_create_minic_datasets(synthetic_classification_ds):
    preproc, aug, laug, postproc = get_pipeline_for_dataset(
        "cifar10",
        task_type="classification",
        apply_presets=False,
        is_training=False,
        aug_kwargs={"image_size": 32},
        postproc_kwargs={"image_size": 32, "num_classes": 10},
    )

    with patch("justdata.loader.fetch_ds", return_value=synthetic_classification_ds):
        ds_list, N = create_minic_datasets(
            corruption_types=["noise", "blur"],
            severity=1,
            dataset_names_arg="mock",
            splits_arg="val",
            dataset_type="validation",
            batch_size=4,
            seed=42,
            preprocess_fn=preproc,
            augment_fn=aug,
            late_augment_fn=laug,
            postprocess_fn=postproc,
            num_classes=10,
            cache_dataset=False,
        )

        assert isinstance(ds_list, list)
        assert len(ds_list) == 2
        assert N == 5

        for ds in ds_list:
            batch = next(iter(ds))
            assert "image" in batch
            assert batch["image"].shape == (4, 3, 32, 32)
            assert batch["image"].dtype == tf.float32

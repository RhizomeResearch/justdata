from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import tensorflow as tf

from justdata.core.metadata import apply_metadata_mode, attach_sidecar_writer


MetadataMode = Literal["full", "numeric_only", "none"]


def _seed_for_index(seed: tf.Tensor, index: tf.Tensor) -> tf.Tensor:
    seed = tf.convert_to_tensor(seed)
    return tf.random.experimental.stateless_fold_in(
        seed,
        tf.cast(index, seed.dtype),
    )


def _pad_dataset(ds: tf.data.Dataset, batch_size: int) -> tf.data.Dataset:
    def first_tensor(value):
        if isinstance(value, dict):
            for child in value.values():
                found = first_tensor(child)
                if found is not None:
                    return found
            return None
        return value

    def get_batch_dim(batch):
        for value in batch.values():
            tensor = first_tensor(value)
            if tensor is not None:
                return tf.shape(tensor)[0]
        return tf.constant(0, dtype=tf.int32)

    def pad_value(value, pad_size):
        if isinstance(value, dict):
            return {key: pad_value(child, pad_size) for key, child in value.items()}

        pad_shape = tf.concat([[pad_size], tf.shape(value)[1:]], axis=0)
        if value.dtype == tf.string:
            fill = tf.fill(pad_shape, tf.constant("", dtype=tf.string))
        else:
            fill = tf.zeros(pad_shape, dtype=value.dtype)
        return tf.concat([value, fill], axis=0)

    def pad_batch(batch):
        current_size = get_batch_dim(batch)
        pad_size = batch_size - current_size
        padded = {key: pad_value(value, pad_size) for key, value in batch.items()}
        padded["padding_mask"] = tf.concat(
            [
                tf.ones((current_size,), dtype=tf.bool),
                tf.zeros((pad_size,), dtype=tf.bool),
            ],
            axis=0,
        )
        return padded

    return ds.map(
        lambda batch: tf.cond(
            get_batch_dim(batch) < batch_size,
            lambda: pad_batch(batch),
            lambda: batch | {"padding_mask": tf.ones((batch_size,), dtype=tf.bool)},
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )


def finalize_dataset(
    ds: tf.data.Dataset,
    *,
    postprocess_fn: Callable,
    num_classes: int | None,
    batch_size: int,
    post_postprocess_transform: Callable | None = None,
    metadata_mode: MetadataMode = "full",
    sidecar_metadata_path: str | None = None,
    cache_model_inputs: bool = False,
    model_input_cache_path: str = "",
    drop_remainder: bool = False,
    is_training: bool = False,
    apply_late_augment: bool | None = None,
    late_augment_fn: Callable | None = None,
    rng: tf.random.Generator | None = None,
    late_augment_seed: tf.Tensor | None = None,
    deterministic: bool = False,
    shuffle_buffer: int | None = None,
    shuffle_seed: int | None = None,
    reshuffle_each_iteration: bool = True,
    prefetch: bool = True,
    as_numpy: bool = False,
) -> tuple[Any, int | None]:
    """Finalize preprocessed samples into padded, model-ready batches."""
    if metadata_mode not in {"full", "numeric_only", "none"}:
        raise ValueError(
            "metadata_mode must be one of 'full', 'numeric_only', or 'none'."
        )
    if apply_late_augment is None:
        apply_late_augment = is_training
    if (
        apply_late_augment
        and late_augment_fn is not None
        and rng is None
        and late_augment_seed is None
    ):
        raise ValueError(
            "rng or late_augment_seed is required when late augmentation is enabled."
        )

    map_deterministic = deterministic if is_training else None

    def apply_postprocessing(dataset):
        dataset = dataset.map(
            lambda sample: postprocess_fn(sample, num_classes=num_classes),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=map_deterministic,
        )
        if post_postprocess_transform is not None:
            dataset = post_postprocess_transform(dataset)
        return dataset

    def apply_metadata(dataset):
        if metadata_mode == "numeric_only" and sidecar_metadata_path is not None:
            dataset = attach_sidecar_writer(dataset, sidecar_metadata_path)
        if metadata_mode != "full":
            dataset = dataset.map(
                lambda sample: apply_metadata_mode(sample, metadata_mode),
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=map_deterministic,
            )
        return dataset

    def apply_metadata_and_cache(dataset):
        if cache_model_inputs and sidecar_metadata_path is not None:
            dataset = dataset.cache(model_input_cache_path)
            return apply_metadata(dataset)
        dataset = apply_metadata(dataset)
        if cache_model_inputs:
            dataset = dataset.cache(model_input_cache_path)
        return dataset

    should_shuffle = is_training and shuffle_buffer is not None
    if should_shuffle and not cache_model_inputs:
        ds = ds.shuffle(
            shuffle_buffer,
            seed=shuffle_seed,
            reshuffle_each_iteration=reshuffle_each_iteration,
        )

    ds = apply_postprocessing(ds)
    ds = apply_metadata_and_cache(ds)

    if should_shuffle and cache_model_inputs:
        ds = ds.shuffle(
            shuffle_buffer,
            seed=shuffle_seed,
            reshuffle_each_iteration=reshuffle_each_iteration,
        )

    ds = ds.batch(batch_size, drop_remainder=drop_remainder)

    if apply_late_augment and late_augment_fn is not None:
        if late_augment_seed is not None:
            ds = ds.enumerate().map(
                lambda index, batch: late_augment_fn(
                    batch,
                    num_classes=num_classes,
                    seed=_seed_for_index(late_augment_seed, index),
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=deterministic,
            )
        else:
            ds = ds.map(
                lambda batch: late_augment_fn(
                    batch,
                    num_classes=num_classes,
                    seed=rng.make_seeds(1)[:, 0],
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=deterministic,
            )

    if drop_remainder:
        ds = ds.map(
            lambda batch: (
                batch | {"padding_mask": tf.ones((batch_size,), dtype=tf.bool)}
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=map_deterministic,
        )
    else:
        ds = _pad_dataset(ds, batch_size)

    if prefetch:
        ds = ds.prefetch(tf.data.AUTOTUNE)

    cardinality = int(tf.data.Dataset.cardinality(ds).numpy())
    n_batches = cardinality if cardinality >= 0 else None
    if as_numpy:
        return ds.as_numpy_iterator(), n_batches
    return ds, n_batches


__all__ = ["finalize_dataset"]

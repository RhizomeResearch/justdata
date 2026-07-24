from collections.abc import Mapping
from typing import Union

import tensorflow as tf

from justdata.core.loader import load_ds
from justdata.vision.corruptions.registry import (
    _metadata_with_corruption,
    apply_corruption,
    get_corruption_descriptor,
    list_corruption_versions,
)


_MAX_SEED = tf.constant(2**31 - 1, dtype=tf.int64)
_LEGACY_MINIC_VERSIONS = {
    "blur": "1.0.0",
    "digital": "1.0.0",
    "noise": "1.0.0",
    "weather": "1.0.0",
}


def create_minic_datasets(
    corruption_types: Union[str, list[str]],
    severity: int = 3,
    *,
    corruption_versions: Mapping[str, str] | None = None,
    **load_ds_kwargs,
) -> Union[tuple[tf.data.Dataset, int], tuple[list[tf.data.Dataset], int]]:
    """
    Create finalized corruption datasets from a preprocessed base dataset.

    The legacy Mini-C names stay pinned to version 1.0.0. Other names must have
    exactly one registered version unless ``corruption_versions`` selects one.
    Dataset-derived seeds retain the historical list-position salt; callers
    with scientific seed lineages should use ``apply_corruption`` directly.
    """

    return_list = isinstance(corruption_types, list)
    c_list = corruption_types if return_list else [corruption_types]
    resolved_corruptions = [
        (
            name,
            _resolve_corruption_version(name, corruption_versions),
        )
        for name in c_list
    ]
    descriptors = [
        get_corruption_descriptor(name, version)
        for name, version in resolved_corruptions
    ]
    for descriptor in descriptors:
        if int(severity) not in descriptor.severity_values:
            raise ValueError(
                f"Corruption '{descriptor.name}@{descriptor.version}' severity "
                f"must be one of {descriptor.severity_values}."
            )

    load_ds_kwargs["return_raw_ds"] = True

    ds, tools = load_ds(**load_ds_kwargs)

    finalize_fn = tools["finalize_fn"]
    seed = load_ds_kwargs.get("seed", 0)

    datasets_out = []

    for c_index, descriptor in enumerate(descriptors):

        def corrupt_fn(
            index,
            sample,
            descriptor=descriptor,
            c_index=c_index,
        ):
            s_seed = _seed_from_index(seed, index, salt=c_index * 1_000_003)
            img = sample["image"]
            img_corrupted = apply_corruption(
                img,
                name=descriptor.name,
                version=descriptor.version,
                severity=severity,
                seed=s_seed,
            )
            return _metadata_with_corruption(
                sample | {"image": img_corrupted},
                descriptor=descriptor,
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


def _resolve_corruption_version(
    name: str,
    requested: Mapping[str, str] | None,
) -> str:
    if requested is not None and name in requested:
        version = requested[name]
        get_corruption_descriptor(name, version)
        return version
    if name in _LEGACY_MINIC_VERSIONS:
        return _LEGACY_MINIC_VERSIONS[name]

    versions = list_corruption_versions(name)
    if len(versions) == 1:
        return versions[0]
    if not versions:
        raise ValueError(f"Unknown corruption '{name}'.")
    raise ValueError(
        f"Corruption '{name}' has multiple versions {versions}; "
        "select one with corruption_versions."
    )


def _seed_from_index(
    seed: int | tf.Tensor, index: tf.Tensor, salt: int = 0
) -> tf.Tensor:
    base = tf.cast(seed, tf.int64)
    idx = tf.cast(index, tf.int64)
    salted = tf.math.floormod(base * 1_103_515_245 + idx + salt, _MAX_SEED)
    return tf.cast(tf.stack([tf.math.floormod(base, _MAX_SEED), salted]), tf.int32)

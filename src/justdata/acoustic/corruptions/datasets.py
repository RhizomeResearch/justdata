from __future__ import annotations

from typing import Literal, Union

import tensorflow as tf

from justdata.acoustic.corruptions.registry import (
    _metadata_with_corruption,
    _seed_from_index,
    apply_audio_corruption,
)
from justdata.acoustic.schema import FEATURES, SAMPLE_RATE, WAVEFORM
from justdata.core.loader import load_ds
from justdata.core.registry import get_pipeline


def create_audio_corruption_datasets(
    corruption_types: Union[str, list[str]],
    severity: int,
    base_dataset: str,
    preset: str | None,
    *,
    split: str = "validation",
    seed: int = 0,
    corruption_domain: Literal["waveform", "spectrogram"] = "waveform",
    **load_kwargs,
) -> Union[tuple[tf.data.Dataset, int], tuple[list[tf.data.Dataset], int]]:
    """Create severity-parameterized acoustic corruption evaluation datasets."""
    if corruption_domain not in {"waveform", "spectrogram"}:
        raise ValueError("corruption_domain must be 'waveform' or 'spectrogram'.")

    kwargs = dict(load_kwargs)
    corruption_configs = kwargs.pop("corruption_configs", None)
    kwargs["return_raw_ds"] = True
    kwargs.setdefault("dataset_names_arg", base_dataset)
    kwargs.setdefault("splits_arg", split)
    kwargs.setdefault("dataset_type", "train" if split == "train" else "validation")
    kwargs.setdefault("batch_size", 32)
    kwargs.setdefault("seed", seed)
    kwargs.setdefault("deterministic", True)

    if "pipeline" not in kwargs and "postprocess_fn" not in kwargs:
        kwargs["pipeline"] = get_pipeline(dataset=base_dataset, preset=preset)

    ds, tools = load_ds(**kwargs)
    finalize_fn = tools["finalize_fn"]

    return_list = isinstance(corruption_types, list)
    names = corruption_types if return_list else [corruption_types]
    datasets_out = []

    for index, c_name in enumerate(names):
        config = None
        if isinstance(corruption_configs, dict):
            config = corruption_configs.get(c_name)

        if corruption_domain == "waveform":
            ds_c = ds.enumerate().map(
                lambda i, sample, c_name=c_name, config=config, index=index: (
                    _corrupt_sample(
                        i,
                        sample,
                        c_name,
                        severity,
                        seed,
                        "waveform",
                        WAVEFORM,
                        index,
                        config,
                    )
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=True,
            )
            ds_c, n_batches = finalize_fn(ds_c)
        else:

            def corrupt_after_postprocess(
                dataset,
                c_name=c_name,
                config=config,
                index=index,
            ):
                return dataset.enumerate().map(
                    lambda i, sample: _corrupt_sample(
                        i,
                        sample,
                        c_name,
                        severity,
                        seed,
                        "spectrogram",
                        FEATURES,
                        index,
                        config,
                    ),
                    num_parallel_calls=tf.data.AUTOTUNE,
                    deterministic=True,
                )

            ds_c, n_batches = finalize_fn(
                ds,
                post_postprocess_transform=corrupt_after_postprocess,
            )

        datasets_out.append(ds_c)

    if not datasets_out:
        n_batches = 0
    if return_list:
        return datasets_out, n_batches
    return datasets_out[0], n_batches


def _corrupt_sample(
    index: tf.Tensor,
    sample: dict,
    name: str,
    severity: int,
    seed: int,
    domain: str,
    key: str,
    salt: int,
    config: dict | None,
) -> dict:
    if key not in sample:
        raise ValueError(
            f"Audio corruption domain '{domain}' requires sample key '{key}'."
        )
    corruption_config = config
    if domain == "waveform":
        if SAMPLE_RATE not in sample:
            raise ValueError(
                f"Audio corruption domain '{domain}' requires sample key "
                f"'{SAMPLE_RATE}'."
            )
        corruption_config = dict(config) if config is not None else {}
        corruption_config.setdefault("sample_rate", sample[SAMPLE_RATE])
    sample_seed = _seed_from_index(seed, index, salt=salt * 1_000_003)
    result = dict(sample)
    result[key] = apply_audio_corruption(
        result[key],
        name,
        severity,
        sample_seed,
        config=corruption_config,
    )
    return _metadata_with_corruption(
        result,
        name=name,
        severity=severity,
        domain=domain,
    )


__all__ = ["create_audio_corruption_datasets"]

from __future__ import annotations

from typing import Literal

import tensorflow as tf

from justdata.acoustic.dcase2025 import assert_stats_allowed
from justdata.core.stats import compute_feature_stats as _compute_feature_stats
from justdata.core.stats import make_stats_iterator as _make_stats_iterator


def make_stats_iterator(
    dataset: tf.data.Dataset,
    split: str,
    groupby: str | list[str] | None,
    batch_size: int,
    deterministic: bool = True,
    augment: bool = False,
    allow_override: bool = False,
    metadata_mode: Literal["full", "numeric_only", "none"] = "numeric_only",
):
    assert_stats_allowed(split, allow_override=allow_override)
    return _make_stats_iterator(
        dataset,
        groupby=groupby,
        batch_size=batch_size,
        deterministic=deterministic,
        augment=augment,
        metadata_mode=metadata_mode,
    )


def compute_feature_stats(
    dataset,
    axes=("time",),
    groupby="device",
    feature_key="inputs",
) -> dict:
    return _compute_feature_stats(
        dataset,
        axes=tuple(axes),
        groupby=groupby,
        feature_key=feature_key,
    )


__all__ = [
    "compute_feature_stats",
    "make_stats_iterator",
]

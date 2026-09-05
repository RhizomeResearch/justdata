from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import tensorflow as tf

from justdata.acoustic._random import _seed_tensor
from justdata.acoustic.augment._common import _config_data as _merge_config
from justdata.acoustic.augment.spectrogram import (
    SpectrogramLayout,
    _as_tfc,
    _restore_from_tfc,
    _stateless_shuffled_range,
)
from justdata.acoustic.registry import register_audio_spectrogram_augment
from justdata.acoustic.schema import FEATURES, METADATA


def _non_negative_count(name: str, value: int | tf.Tensor | None) -> int | tf.Tensor:
    if value is None:
        return 0
    if isinstance(value, int) and value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _is_zero_count(value: int | tf.Tensor) -> bool:
    if isinstance(value, int):
        return value == 0
    static_value = tf.get_static_value(value)
    return static_value is not None and int(static_value) == 0


def _drop_axis(
    x: tf.Tensor,
    *,
    axis: int,
    count: int | tf.Tensor,
    seed: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    count = tf.cast(count, tf.int32)
    size = tf.shape(x)[axis]
    count = tf.minimum(tf.maximum(count, 0), size)
    dropped = tf.sort(_stateless_shuffled_range(size, seed)[:count])
    positions = tf.range(size)
    keep_mask = ~tf.reduce_any(
        tf.equal(positions[:, tf.newaxis], dropped[tf.newaxis, :]),
        axis=1,
    )
    keep = tf.reshape(tf.where(keep_mask), [-1])
    return tf.gather(x, keep, axis=axis), dropped


def _structured_patchout_frequency_with_indices(
    x: tf.Tensor,
    n_freq_patches: int | tf.Tensor,
    seed: tf.Tensor | int | None,
    *,
    layout: SpectrogramLayout | None = None,
) -> tuple[tf.Tensor, tf.Tensor]:
    seed = _seed_tensor(seed)
    grid, resolved_layout = _as_tfc(x, layout)
    dropped_grid, dropped = _drop_axis(
        grid,
        axis=1,
        count=n_freq_patches,
        seed=seed,
    )
    return _restore_from_tfc(dropped_grid, resolved_layout), dropped


def _structured_patchout_time_with_indices(
    x: tf.Tensor,
    n_time_patches: int | tf.Tensor,
    seed: tf.Tensor | int | None,
    *,
    layout: SpectrogramLayout | None = None,
) -> tuple[tf.Tensor, tf.Tensor]:
    seed = _seed_tensor(seed)
    grid, resolved_layout = _as_tfc(x, layout)
    dropped_grid, dropped = _drop_axis(
        grid,
        axis=0,
        count=n_time_patches,
        seed=seed,
    )
    return _restore_from_tfc(dropped_grid, resolved_layout), dropped


def _unstructured_patchout_with_indices(
    x: tf.Tensor,
    n_patches: int | tf.Tensor,
    seed: tf.Tensor | int | None,
    *,
    layout: SpectrogramLayout | None = None,
) -> tuple[tf.Tensor, tf.Tensor]:
    seed = _seed_tensor(seed)
    grid, _ = _as_tfc(x, layout)
    time = tf.shape(grid)[0]
    freq = tf.shape(grid)[1]
    channels = tf.shape(grid)[2]
    flat = tf.reshape(grid, [time * freq, channels])
    dropped_flat, dropped = _drop_axis(flat, axis=0, count=n_patches, seed=seed)

    if tf.convert_to_tensor(x).shape.rank == 2:
        return tf.squeeze(dropped_flat, axis=-1), dropped
    return dropped_flat, dropped


def structured_patchout_frequency(
    x: tf.Tensor,
    n_freq_patches: int | tf.Tensor,
    seed: tf.Tensor | int | None = None,
    *,
    layout: SpectrogramLayout | None = None,
    is_training: bool = True,
    augment_eval: bool = False,
) -> tf.Tensor:
    if not (is_training or augment_eval):
        return tf.convert_to_tensor(x)
    n_freq_patches = _non_negative_count("n_freq_patches", n_freq_patches)
    result, _ = _structured_patchout_frequency_with_indices(
        x,
        n_freq_patches,
        seed,
        layout=layout,
    )
    return result


def structured_patchout_time(
    x: tf.Tensor,
    n_time_patches: int | tf.Tensor,
    seed: tf.Tensor | int | None = None,
    *,
    layout: SpectrogramLayout | None = None,
    is_training: bool = True,
    augment_eval: bool = False,
) -> tf.Tensor:
    if not (is_training or augment_eval):
        return tf.convert_to_tensor(x)
    n_time_patches = _non_negative_count("n_time_patches", n_time_patches)
    result, _ = _structured_patchout_time_with_indices(
        x,
        n_time_patches,
        seed,
        layout=layout,
    )
    return result


def unstructured_patchout(
    x: tf.Tensor,
    n_patches: int | tf.Tensor,
    seed: tf.Tensor | int | None = None,
    *,
    layout: SpectrogramLayout | None = None,
    is_training: bool = True,
    augment_eval: bool = False,
) -> tf.Tensor:
    if not (is_training or augment_eval):
        return tf.convert_to_tensor(x)
    n_patches = _non_negative_count("n_patches", n_patches)
    if _is_zero_count(n_patches):
        return tf.convert_to_tensor(x)
    result, _ = _unstructured_patchout_with_indices(x, n_patches, seed, layout=layout)
    return result


def _resolve_patchout_counts(
    config: Mapping[str, Any] | None,
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
    cfg = _merge_config(config, overrides)
    return {
        "n_freq_patches": _non_negative_count(
            "n_freq_patches",
            cfg.get("n_freq_patches", cfg.get("structured_frequency", 0)),
        ),
        "n_time_patches": _non_negative_count(
            "n_time_patches",
            cfg.get("n_time_patches", cfg.get("structured_time", 0)),
        ),
        "n_patches": _non_negative_count(
            "n_patches",
            cfg.get("n_patches", cfg.get("unstructured", 0)),
        ),
        "layout": cfg.get("layout"),
        "debug": bool(cfg.get("debug", False)),
        "feature_key": cfg.get("feature_key", FEATURES),
    }


def _patchout_tensor(
    x: tf.Tensor,
    *,
    seed: tf.Tensor | int | None,
    n_freq_patches: int | tf.Tensor,
    n_time_patches: int | tf.Tensor,
    n_patches: int | tf.Tensor,
    layout: SpectrogramLayout | None,
) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
    seed = _seed_tensor(seed)
    freq_seed, time_seed, unstructured_seed = tf.unstack(tf.random.split(seed, 3))

    result, dropped_freq = _structured_patchout_frequency_with_indices(
        x,
        n_freq_patches,
        freq_seed,
        layout=layout,
    )
    result, dropped_time = _structured_patchout_time_with_indices(
        result,
        n_time_patches,
        time_seed,
        layout=layout,
    )
    dropped_unstructured = tf.zeros([0], dtype=tf.int32)
    if not _is_zero_count(n_patches):
        result, dropped_unstructured = _unstructured_patchout_with_indices(
            result,
            n_patches,
            unstructured_seed,
            layout=layout,
        )
    return result, {
        "structured_frequency": dropped_freq,
        "structured_time": dropped_time,
        "unstructured": dropped_unstructured,
    }


def _patchout_sample(
    sample: Mapping[str, Any],
    *,
    seed: tf.Tensor | int | None,
    n_freq_patches: int | tf.Tensor,
    n_time_patches: int | tf.Tensor,
    n_patches: int | tf.Tensor,
    layout: SpectrogramLayout | None,
    feature_key: str,
    debug: bool,
) -> dict:
    if feature_key not in sample:
        return dict(sample)

    result = dict(sample)
    patched, dropped = _patchout_tensor(
        result[feature_key],
        seed=seed,
        n_freq_patches=n_freq_patches,
        n_time_patches=n_time_patches,
        n_patches=n_patches,
        layout=layout,
    )
    result[feature_key] = patched

    if debug:
        metadata = dict(result.get(METADATA, {}))
        metadata["patchout"] = dropped
        result[METADATA] = metadata

    return result


@register_audio_spectrogram_augment("passt_patchout")
def passt_patchout(
    x: tf.Tensor | Mapping[str, Any],
    *,
    seed: tf.Tensor | int | None = None,
    config: Mapping[str, Any] | None = None,
    n_freq_patches: int | tf.Tensor | None = None,
    n_time_patches: int | tf.Tensor | None = None,
    n_patches: int | tf.Tensor | None = None,
    structured_frequency: int | tf.Tensor | None = None,
    structured_time: int | tf.Tensor | None = None,
    unstructured: int | tf.Tensor | None = None,
    layout: SpectrogramLayout | None = None,
    feature_key: str = FEATURES,
    debug: bool | None = None,
    is_training: bool = True,
    augment_eval: bool = False,
) -> tf.Tensor | dict:
    if not (is_training or augment_eval):
        return dict(x) if isinstance(x, Mapping) else tf.convert_to_tensor(x)

    cfg = _resolve_patchout_counts(
        config,
        {
            "n_freq_patches": n_freq_patches,
            "n_time_patches": n_time_patches,
            "n_patches": n_patches,
            "structured_frequency": structured_frequency,
            "structured_time": structured_time,
            "unstructured": unstructured,
            "layout": layout,
            "feature_key": feature_key,
            "debug": debug,
        },
    )

    if isinstance(x, Mapping):
        return _patchout_sample(
            x,
            seed=seed,
            n_freq_patches=cfg["n_freq_patches"],
            n_time_patches=cfg["n_time_patches"],
            n_patches=cfg["n_patches"],
            layout=cfg["layout"],
            feature_key=cfg["feature_key"],
            debug=cfg["debug"],
        )

    patched, _ = _patchout_tensor(
        x,
        seed=seed,
        n_freq_patches=cfg["n_freq_patches"],
        n_time_patches=cfg["n_time_patches"],
        n_patches=cfg["n_patches"],
        layout=cfg["layout"],
    )
    return patched


__all__ = [
    "passt_patchout",
    "structured_patchout_frequency",
    "structured_patchout_time",
    "unstructured_patchout",
]

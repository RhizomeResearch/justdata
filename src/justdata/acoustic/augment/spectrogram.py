from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import tensorflow as tf

from justdata.acoustic.registry import (
    get_audio_spectrogram_augment,
    register_audio_spectrogram_augment,
)
from justdata.acoustic.schema import FEATURES


SpectrogramLayout = Literal["tf", "tfc", "cft"]
BatchSpectrogramLayout = Literal["tf", "tfc", "cft", "btf", "btfc", "bcft"]

_EPS = tf.constant(1e-6, dtype=tf.float32)


def _seed_tensor(seed: tf.Tensor | int | None) -> tf.Tensor:
    if seed is None:
        return tf.constant([0, 0], dtype=tf.int32)

    seed = tf.cast(tf.convert_to_tensor(seed), tf.int32)
    if seed.shape.rank == 0:
        return tf.stack([seed, tf.constant(0, dtype=tf.int32)])
    if seed.shape.rank == 1 and seed.shape[0] == 1:
        return tf.stack([seed[0], tf.constant(0, dtype=tf.int32)])
    return seed[:2]


def _enabled(is_training: bool, augment_eval: bool) -> bool:
    return bool(is_training or augment_eval)


def _uniform_int(seed: tf.Tensor, minval: tf.Tensor, maxval: tf.Tensor) -> tf.Tensor:
    minval = tf.cast(minval, tf.int32)
    maxval = tf.cast(maxval, tf.int32)
    return tf.cond(
        tf.equal(minval, maxval),
        lambda: minval,
        lambda: tf.random.stateless_uniform(
            [],
            seed=seed,
            minval=minval,
            maxval=maxval + 1,
            dtype=tf.int32,
        ),
    )


def _uniform_float(seed: tf.Tensor, minval: float, maxval: float) -> tf.Tensor:
    if minval == maxval:
        return tf.constant(minval, dtype=tf.float32)
    return tf.random.stateless_uniform(
        [],
        seed=seed,
        minval=tf.cast(minval, tf.float32),
        maxval=tf.cast(maxval, tf.float32),
        dtype=tf.float32,
    )


def _stateless_shuffled_range(size: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
    random_values = tf.random.stateless_uniform(tf.reshape(size, [1]), seed=seed)
    return tf.argsort(random_values, stable=True)


def _maybe_apply(x: tf.Tensor, prob: float, seed: tf.Tensor, apply_fn) -> tf.Tensor:
    if prob <= 0.0:
        return x
    if prob >= 1.0:
        return apply_fn()
    should_apply = tf.random.stateless_uniform([], seed=seed) < tf.cast(prob, tf.float32)
    return tf.cond(should_apply, apply_fn, lambda: x)


def _merge_config(config: Any, overrides: Mapping[str, Any]) -> dict[str, Any]:
    data: dict[str, Any] = {}
    if config is not None:
        if dataclasses.is_dataclass(config):
            data.update(dataclasses.asdict(config))
        elif isinstance(config, Mapping):
            data.update(config)
        else:
            raise TypeError("config must be a dataclass, mapping, or None")
    data.update({key: value for key, value in overrides.items() if value is not None})
    return data


def _resolve_layout(
    x: tf.Tensor,
    layout: SpectrogramLayout | None,
) -> SpectrogramLayout:
    if layout is not None:
        return layout
    if x.shape.rank == 2:
        return "tf"
    if x.shape.rank == 3:
        return "tfc"
    raise ValueError("Spectrogram augment input must have rank 2 or 3")


def _as_tfc(
    spectrogram: tf.Tensor,
    layout: SpectrogramLayout | None = None,
) -> tuple[tf.Tensor, SpectrogramLayout]:
    x = tf.convert_to_tensor(spectrogram)
    resolved = _resolve_layout(x, layout)

    if resolved == "tf":
        if x.shape.rank != 2:
            raise ValueError("layout='tf' expects shape [T, F]")
        return x[:, :, tf.newaxis], resolved
    if resolved == "tfc":
        if x.shape.rank != 3:
            raise ValueError("layout='tfc' expects shape [T, F, C]")
        return x, resolved
    if resolved == "cft":
        if x.shape.rank != 3:
            raise ValueError("layout='cft' expects shape [C, F, T]")
        return tf.transpose(x, [2, 1, 0]), resolved
    raise ValueError("layout must be one of 'tf', 'tfc', or 'cft'")


def _restore_from_tfc(x: tf.Tensor, layout: SpectrogramLayout) -> tf.Tensor:
    if layout == "tf":
        return tf.squeeze(x, axis=-1)
    if layout == "tfc":
        return x
    if layout == "cft":
        return tf.transpose(x, [2, 1, 0])
    raise ValueError("layout must be one of 'tf', 'tfc', or 'cft'")


def _resolve_batch_layout(
    x: tf.Tensor,
    layout: BatchSpectrogramLayout | None,
) -> BatchSpectrogramLayout:
    if layout is not None:
        return layout
    if x.shape.rank == 2:
        return "tf"
    if x.shape.rank == 3:
        return "btf"
    if x.shape.rank == 4:
        return "btfc"
    raise ValueError("MixStyle input must have rank 2, 3, or 4")


def _as_btfc(
    spectrogram: tf.Tensor,
    layout: BatchSpectrogramLayout | None = None,
) -> tuple[tf.Tensor, BatchSpectrogramLayout]:
    x = tf.convert_to_tensor(spectrogram)
    resolved = _resolve_batch_layout(x, layout)

    if resolved in {"tf", "tfc", "cft"}:
        sample, sample_layout = _as_tfc(x, resolved)
        del sample_layout
        return sample[tf.newaxis, ...], resolved
    if resolved == "btf":
        if x.shape.rank != 3:
            raise ValueError("layout='btf' expects shape [B, T, F]")
        return x[:, :, :, tf.newaxis], resolved
    if resolved == "btfc":
        if x.shape.rank != 4:
            raise ValueError("layout='btfc' expects shape [B, T, F, C]")
        return x, resolved
    if resolved == "bcft":
        if x.shape.rank != 4:
            raise ValueError("layout='bcft' expects shape [B, C, F, T]")
        return tf.transpose(x, [0, 3, 2, 1]), resolved
    raise ValueError("unsupported MixStyle layout")


def _restore_from_btfc(x: tf.Tensor, layout: BatchSpectrogramLayout) -> tf.Tensor:
    if layout == "tf":
        return _restore_from_tfc(x[0], "tf")
    if layout == "tfc":
        return _restore_from_tfc(x[0], "tfc")
    if layout == "cft":
        return _restore_from_tfc(x[0], "cft")
    if layout == "btf":
        return tf.squeeze(x, axis=-1)
    if layout == "btfc":
        return x
    if layout == "bcft":
        return tf.transpose(x, [0, 3, 2, 1])
    raise ValueError("unsupported MixStyle layout")


def _max_width(width: int | tf.Tensor | None, limit: tf.Tensor) -> tf.Tensor:
    if width is None:
        return tf.cast(limit, tf.int32)
    if isinstance(width, int) and width < 0:
        raise ValueError("mask width must be non-negative")
    return tf.minimum(tf.cast(width, tf.int32), tf.cast(limit, tf.int32))


def _fill_scalar(x: tf.Tensor, fill_value: Literal["zero", "mean", "min"]) -> tf.Tensor:
    if fill_value == "zero":
        return tf.zeros([], dtype=x.dtype)
    if fill_value == "mean":
        return tf.cast(tf.reduce_mean(tf.cast(x, tf.float32)), x.dtype)
    if fill_value == "min":
        return tf.reduce_min(x)
    raise ValueError("fill_value must be one of 'zero', 'mean', or 'min'")


def _sample_span(
    size: tf.Tensor,
    max_width: int | tf.Tensor | None,
    width_seed: tf.Tensor,
    start_seed: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    width_max = _max_width(max_width, size)
    width = _uniform_int(width_seed, tf.constant(0, tf.int32), width_max)
    max_start = tf.maximum(tf.cast(size, tf.int32) - width, 0)
    start = _uniform_int(start_seed, tf.constant(0, tf.int32), max_start)
    return start, width


@register_audio_spectrogram_augment("frequency_mask")
def frequency_mask(
    spectrogram: tf.Tensor,
    *,
    seed: tf.Tensor | int | None = None,
    config: Mapping[str, Any] | None = None,
    max_width: int | tf.Tensor | None = None,
    fill_value: Literal["zero", "mean", "min"] | None = None,
    prob: float | None = None,
    layout: SpectrogramLayout | None = None,
    is_training: bool = True,
    augment_eval: bool = False,
) -> tf.Tensor:
    if not _enabled(is_training, augment_eval):
        return tf.convert_to_tensor(spectrogram)

    cfg = _merge_config(
        config,
        {"max_width": max_width, "fill_value": fill_value, "prob": prob, "layout": layout},
    )
    max_width = cfg.get("max_width", 0)
    fill_value = cfg.get("fill_value", "zero")
    prob = cfg.get("prob", 1.0)
    layout = cfg.get("layout")

    seed = _seed_tensor(seed)
    gate_seed, width_seed, start_seed = tf.unstack(tf.random.split(seed, 3))
    x, resolved_layout = _as_tfc(spectrogram, layout)
    fill = _fill_scalar(x, fill_value)

    def apply() -> tf.Tensor:
        freq = tf.shape(x)[1]
        start, width = _sample_span(freq, max_width, width_seed, start_seed)
        positions = tf.range(freq)
        mask = tf.logical_and(positions >= start, positions < start + width)
        masked = tf.where(mask[tf.newaxis, :, tf.newaxis], fill, x)
        return _restore_from_tfc(masked, resolved_layout)

    return _maybe_apply(tf.convert_to_tensor(spectrogram), prob, gate_seed, apply)


@register_audio_spectrogram_augment("time_mask")
def time_mask(
    spectrogram: tf.Tensor,
    *,
    seed: tf.Tensor | int | None = None,
    config: Mapping[str, Any] | None = None,
    max_width: int | tf.Tensor | None = None,
    fill_value: Literal["zero", "mean", "min"] | None = None,
    prob: float | None = None,
    layout: SpectrogramLayout | None = None,
    is_training: bool = True,
    augment_eval: bool = False,
) -> tf.Tensor:
    if not _enabled(is_training, augment_eval):
        return tf.convert_to_tensor(spectrogram)

    cfg = _merge_config(
        config,
        {"max_width": max_width, "fill_value": fill_value, "prob": prob, "layout": layout},
    )
    max_width = cfg.get("max_width", 0)
    fill_value = cfg.get("fill_value", "zero")
    prob = cfg.get("prob", 1.0)
    layout = cfg.get("layout")

    seed = _seed_tensor(seed)
    gate_seed, width_seed, start_seed = tf.unstack(tf.random.split(seed, 3))
    x, resolved_layout = _as_tfc(spectrogram, layout)
    fill = _fill_scalar(x, fill_value)

    def apply() -> tf.Tensor:
        time = tf.shape(x)[0]
        start, width = _sample_span(time, max_width, width_seed, start_seed)
        positions = tf.range(time)
        mask = tf.logical_and(positions >= start, positions < start + width)
        masked = tf.where(mask[:, tf.newaxis, tf.newaxis], fill, x)
        return _restore_from_tfc(masked, resolved_layout)

    return _maybe_apply(tf.convert_to_tensor(spectrogram), prob, gate_seed, apply)


@register_audio_spectrogram_augment("spectrogram_time_roll")
def spectrogram_time_roll(
    spectrogram: tf.Tensor,
    *,
    seed: tf.Tensor | int | None = None,
    config: Mapping[str, Any] | None = None,
    max_shift: int | tf.Tensor | None = None,
    shift: int | tf.Tensor | None = None,
    prob: float | None = None,
    layout: SpectrogramLayout | None = None,
    is_training: bool = True,
    augment_eval: bool = False,
) -> tf.Tensor:
    if not _enabled(is_training, augment_eval):
        return tf.convert_to_tensor(spectrogram)

    cfg = _merge_config(
        config,
        {"max_shift": max_shift, "shift": shift, "prob": prob, "layout": layout},
    )
    max_shift = cfg.get("max_shift", 0)
    shift = cfg.get("shift")
    prob = cfg.get("prob", 1.0)
    layout = cfg.get("layout")

    seed = _seed_tensor(seed)
    gate_seed, shift_seed = tf.unstack(tf.random.split(seed, 2))
    x, resolved_layout = _as_tfc(spectrogram, layout)

    def apply() -> tf.Tensor:
        if shift is None:
            time = tf.shape(x)[0]
            limit = _max_width(max_shift, time)
            sampled_shift = _uniform_int(shift_seed, -limit, limit)
        else:
            sampled_shift = tf.cast(shift, tf.int32)
        return _restore_from_tfc(tf.roll(x, shift=sampled_shift, axis=0), resolved_layout)

    return _maybe_apply(tf.convert_to_tensor(spectrogram), prob, gate_seed, apply)


def _zero_shifted_frequency_edges(x: tf.Tensor, shift: tf.Tensor) -> tf.Tensor:
    freq = tf.shape(x)[1]
    shift = tf.clip_by_value(tf.cast(shift, tf.int32), -freq, freq)
    positions = tf.range(freq)

    def positive() -> tf.Tensor:
        mask = positions < shift
        return tf.where(mask[tf.newaxis, :, tf.newaxis], tf.zeros([], dtype=x.dtype), x)

    def negative() -> tf.Tensor:
        mask = positions >= freq + shift
        return tf.where(mask[tf.newaxis, :, tf.newaxis], tf.zeros([], dtype=x.dtype), x)

    return tf.cond(
        shift > 0,
        positive,
        lambda: tf.cond(shift < 0, negative, lambda: x),
    )


@register_audio_spectrogram_augment("mel_bin_shift")
def mel_bin_shift(
    spectrogram: tf.Tensor,
    *,
    seed: tf.Tensor | int | None = None,
    config: Mapping[str, Any] | None = None,
    max_bins: int | tf.Tensor | None = None,
    shift: int | tf.Tensor | None = None,
    zero_pad: bool | None = None,
    prob: float | None = None,
    layout: SpectrogramLayout | None = None,
    is_training: bool = True,
    augment_eval: bool = False,
) -> tf.Tensor:
    if not _enabled(is_training, augment_eval):
        return tf.convert_to_tensor(spectrogram)

    cfg = _merge_config(
        config,
        {
            "max_bins": max_bins,
            "shift": shift,
            "zero_pad": zero_pad,
            "prob": prob,
            "layout": layout,
        },
    )
    max_bins = cfg.get("max_bins", 0)
    shift = cfg.get("shift")
    zero_pad = cfg.get("zero_pad", True)
    prob = cfg.get("prob", 1.0)
    layout = cfg.get("layout")

    seed = _seed_tensor(seed)
    gate_seed, shift_seed = tf.unstack(tf.random.split(seed, 2))
    x, resolved_layout = _as_tfc(spectrogram, layout)

    def apply() -> tf.Tensor:
        if shift is None:
            freq = tf.shape(x)[1]
            limit = _max_width(max_bins, freq)
            sampled_shift = _uniform_int(shift_seed, -limit, limit)
        else:
            sampled_shift = tf.cast(shift, tf.int32)
        shifted = tf.roll(x, shift=sampled_shift, axis=1)
        if zero_pad:
            shifted = _zero_shifted_frequency_edges(shifted, sampled_shift)
        return _restore_from_tfc(shifted, resolved_layout)

    return _maybe_apply(tf.convert_to_tensor(spectrogram), prob, gate_seed, apply)


@register_audio_spectrogram_augment("time_frequency_erasing")
def time_frequency_erasing(
    spectrogram: tf.Tensor,
    *,
    seed: tf.Tensor | int | None = None,
    config: Mapping[str, Any] | None = None,
    max_time_width: int | tf.Tensor | None = None,
    max_freq_width: int | tf.Tensor | None = None,
    fill_value: Literal["zero", "mean", "min"] | None = None,
    prob: float | None = None,
    layout: SpectrogramLayout | None = None,
    is_training: bool = True,
    augment_eval: bool = False,
) -> tf.Tensor:
    if not _enabled(is_training, augment_eval):
        return tf.convert_to_tensor(spectrogram)

    cfg = _merge_config(
        config,
        {
            "max_time_width": max_time_width,
            "max_freq_width": max_freq_width,
            "fill_value": fill_value,
            "prob": prob,
            "layout": layout,
        },
    )
    max_time_width = cfg.get("max_time_width")
    max_freq_width = cfg.get("max_freq_width")
    fill_value = cfg.get("fill_value", "zero")
    prob = cfg.get("prob", 1.0)
    layout = cfg.get("layout")

    seed = _seed_tensor(seed)
    gate_seed, t_width_seed, t_start_seed, f_width_seed, f_start_seed = tf.unstack(
        tf.random.split(seed, 5)
    )
    x, resolved_layout = _as_tfc(spectrogram, layout)
    fill = _fill_scalar(x, fill_value)

    def apply() -> tf.Tensor:
        time = tf.shape(x)[0]
        freq = tf.shape(x)[1]
        t_start, t_width = _sample_span(time, max_time_width, t_width_seed, t_start_seed)
        f_start, f_width = _sample_span(freq, max_freq_width, f_width_seed, f_start_seed)
        time_positions = tf.range(time)
        freq_positions = tf.range(freq)
        time_mask = tf.logical_and(
            time_positions >= t_start,
            time_positions < t_start + t_width,
        )
        freq_mask = tf.logical_and(
            freq_positions >= f_start,
            freq_positions < f_start + f_width,
        )
        mask = tf.logical_and(time_mask[:, tf.newaxis], freq_mask[tf.newaxis, :])
        erased = tf.where(mask[:, :, tf.newaxis], fill, x)
        return _restore_from_tfc(erased, resolved_layout)

    return _maybe_apply(tf.convert_to_tensor(spectrogram), prob, gate_seed, apply)


def _sample_beta(shape: tf.Tensor | Sequence[int], alpha: float, seed: tf.Tensor) -> tf.Tensor:
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    first_seed, second_seed = tf.unstack(tf.random.split(seed, 2))
    first = tf.random.stateless_gamma(shape, seed=first_seed, alpha=alpha, dtype=tf.float32)
    second = tf.random.stateless_gamma(shape, seed=second_seed, alpha=alpha, dtype=tf.float32)
    return first / (first + second + _EPS)


def _mixstyle_impl(
    spectrogram: tf.Tensor,
    *,
    seed: tf.Tensor | int | None,
    alpha: float,
    prob: float,
    layout: BatchSpectrogramLayout | None,
    stat_axes: Sequence[int],
) -> tf.Tensor:
    seed = _seed_tensor(seed)
    gate_seed, shuffle_seed, beta_seed = tf.unstack(tf.random.split(seed, 3))
    x, resolved_layout = _as_btfc(spectrogram, layout)
    input_dtype = x.dtype
    x_float = tf.cast(x, tf.float32)

    def apply() -> tf.Tensor:
        batch = tf.shape(x_float)[0]

        def mix() -> tf.Tensor:
            axes = list(stat_axes)
            mean = tf.reduce_mean(x_float, axis=axes, keepdims=True)
            var = tf.reduce_mean(tf.square(x_float - mean), axis=axes, keepdims=True)
            std = tf.sqrt(var + _EPS)
            partner = _stateless_shuffled_range(batch, shuffle_seed)
            partner_mean = tf.gather(mean, partner, axis=0)
            partner_std = tf.gather(std, partner, axis=0)
            lam_shape = tf.concat(
                [tf.reshape(batch, [1]), tf.ones([tf.rank(mean) - 1], tf.int32)],
                axis=0,
            )
            lam = _sample_beta(lam_shape, alpha, beta_seed)
            mixed_mean = lam * mean + (1.0 - lam) * partner_mean
            mixed_std = lam * std + (1.0 - lam) * partner_std
            mixed = (x_float - mean) / std * mixed_std + mixed_mean
            return tf.cast(mixed, input_dtype)

        return tf.cond(batch > 1, mix, lambda: x)

    mixed = _maybe_apply(x, prob, gate_seed, apply)
    return _restore_from_btfc(mixed, resolved_layout)


@register_audio_spectrogram_augment("mixstyle")
def mixstyle(
    spectrogram: tf.Tensor,
    *,
    seed: tf.Tensor | int | None = None,
    config: Mapping[str, Any] | None = None,
    alpha: float | None = None,
    prob: float | None = None,
    layout: BatchSpectrogramLayout | None = None,
    is_training: bool = True,
    augment_eval: bool = False,
) -> tf.Tensor:
    if not _enabled(is_training, augment_eval):
        return tf.convert_to_tensor(spectrogram)

    cfg = _merge_config(config, {"alpha": alpha, "prob": prob, "layout": layout})
    return _mixstyle_impl(
        spectrogram,
        seed=seed,
        alpha=cfg.get("alpha", 0.1),
        prob=cfg.get("prob", 1.0),
        layout=cfg.get("layout"),
        stat_axes=(1, 2, 3),
    )


@register_audio_spectrogram_augment("frequency_mixstyle")
def frequency_mixstyle(
    spectrogram: tf.Tensor,
    *,
    seed: tf.Tensor | int | None = None,
    config: Mapping[str, Any] | None = None,
    alpha: float | None = None,
    prob: float | None = None,
    layout: BatchSpectrogramLayout | None = None,
    is_training: bool = True,
    augment_eval: bool = False,
) -> tf.Tensor:
    if not _enabled(is_training, augment_eval):
        return tf.convert_to_tensor(spectrogram)

    cfg = _merge_config(config, {"alpha": alpha, "prob": prob, "layout": layout})
    return _mixstyle_impl(
        spectrogram,
        seed=seed,
        alpha=cfg.get("alpha", 0.1),
        prob=cfg.get("prob", 1.0),
        layout=cfg.get("layout"),
        stat_axes=(1, 3),
    )


@register_audio_spectrogram_augment("random_eq")
def random_eq(
    spectrogram: tf.Tensor,
    *,
    seed: tf.Tensor | int | None = None,
    config: Mapping[str, Any] | None = None,
    num_bands: int | None = None,
    min_db: float | None = None,
    max_db: float | None = None,
    min_band_width: int | None = None,
    max_band_width: int | None = None,
    prob: float | None = None,
    layout: SpectrogramLayout | None = None,
    is_training: bool = True,
    augment_eval: bool = False,
) -> tf.Tensor:
    if not _enabled(is_training, augment_eval):
        return tf.convert_to_tensor(spectrogram)

    cfg = _merge_config(
        config,
        {
            "num_bands": num_bands,
            "min_db": min_db,
            "max_db": max_db,
            "min_band_width": min_band_width,
            "max_band_width": max_band_width,
            "prob": prob,
            "layout": layout,
        },
    )
    num_bands = int(cfg.get("num_bands", 1))
    min_db = cfg.get("min_db", -6.0)
    max_db = cfg.get("max_db", 6.0)
    min_band_width = int(cfg.get("min_band_width", 1))
    max_band_width = cfg.get("max_band_width")
    prob = cfg.get("prob", 1.0)
    layout = cfg.get("layout")

    if num_bands < 0:
        raise ValueError("num_bands must be non-negative")
    if min_band_width <= 0:
        raise ValueError("min_band_width must be positive")
    if min_db > max_db:
        raise ValueError("min_db must be <= max_db")

    seed = _seed_tensor(seed)
    seeds = tf.unstack(tf.random.split(seed, 1 + max(num_bands, 1) * 3))
    gate_seed = seeds[0]
    band_seeds = seeds[1:]
    x, resolved_layout = _as_tfc(spectrogram, layout)
    input_dtype = x.dtype

    def apply() -> tf.Tensor:
        freq = tf.shape(x)[1]
        positions = tf.range(freq)
        gains = tf.ones([freq], dtype=tf.float32)
        width_max = _max_width(max_band_width, freq)
        width_min = tf.minimum(tf.cast(min_band_width, tf.int32), width_max)

        for band_index in range(num_bands):
            width_seed = band_seeds[band_index * 3]
            start_seed = band_seeds[band_index * 3 + 1]
            gain_seed = band_seeds[band_index * 3 + 2]
            width = _uniform_int(width_seed, width_min, width_max)
            max_start = tf.maximum(freq - width, 0)
            start = _uniform_int(start_seed, tf.constant(0, tf.int32), max_start)
            gain_db = _uniform_float(gain_seed, min_db, max_db)
            gain = tf.pow(tf.constant(10.0, dtype=tf.float32), gain_db / 20.0)
            mask = tf.logical_and(positions >= start, positions < start + width)
            gains = tf.where(mask, gains * gain, gains)

        equalized = tf.cast(x, tf.float32) * gains[tf.newaxis, :, tf.newaxis]
        return _restore_from_tfc(tf.cast(equalized, input_dtype), resolved_layout)

    return _maybe_apply(tf.convert_to_tensor(spectrogram), prob, gate_seed, apply)


def normalize_spectrogram_augment_specs(
    augmentations: Mapping[str, Any] | Sequence[Any] | str | None,
) -> list[dict[str, Any]]:
    if augmentations is None:
        return []
    if isinstance(augmentations, str):
        return [{"name": augmentations}]
    if isinstance(augmentations, Mapping):
        if "name" in augmentations:
            spec = dict(augmentations)
            spec["name"] = str(spec["name"])
            return [spec]

        specs = []
        for name, value in augmentations.items():
            if value is None or value is False:
                continue
            registered_name = "passt_patchout" if name == "patchout" else str(name)
            if value is True:
                specs.append({"name": registered_name})
            elif isinstance(value, Mapping):
                specs.append({"name": registered_name, **dict(value)})
            else:
                specs.append({"name": registered_name, "config": value})
        return specs

    specs = []
    for value in augmentations:
        specs.extend(normalize_spectrogram_augment_specs(value))
    return specs


def apply_spectrogram_augmentations(
    sample: dict,
    augmentations: Mapping[str, Any] | Sequence[Any] | str | None,
    *,
    seed: tf.Tensor | int | None = None,
    is_training: bool = True,
    augment_eval: bool = False,
    feature_key: str = FEATURES,
    layout: SpectrogramLayout | None = None,
) -> dict:
    specs = normalize_spectrogram_augment_specs(augmentations)
    if not specs or not _enabled(is_training, augment_eval) or feature_key not in sample:
        return sample

    seed = _seed_tensor(seed)
    seeds = tf.random.split(seed, len(specs))
    result = dict(sample)

    for spec, spec_seed in zip(specs, tf.unstack(seeds)):
        kwargs = dict(spec)
        name = kwargs.pop("name")
        spec_layout = kwargs.pop("layout", layout)
        fn = get_audio_spectrogram_augment(name)
        if name == "passt_patchout":
            result = fn(
                result,
                seed=spec_seed,
                feature_key=feature_key,
                layout=spec_layout,
                is_training=is_training,
                augment_eval=augment_eval,
                **kwargs,
            )
        else:
            result[feature_key] = fn(
                result[feature_key],
                seed=spec_seed,
                layout=spec_layout,
                is_training=is_training,
                augment_eval=augment_eval,
                **kwargs,
            )

    return result


def make_spectrogram_augmentation_stage(
    augmentations: Mapping[str, Any] | Sequence[Any] | str | None,
    *,
    is_training: bool = True,
    augment_eval: bool = False,
    feature_key: str = FEATURES,
    layout: SpectrogramLayout | None = None,
):
    specs = normalize_spectrogram_augment_specs(augmentations)

    def stage(sample: dict, seed: tf.Tensor | int | None = None) -> dict:
        return apply_spectrogram_augmentations(
            sample,
            specs,
            seed=seed,
            is_training=is_training,
            augment_eval=augment_eval,
            feature_key=feature_key,
            layout=layout,
        )

    return stage


__all__ = [
    "apply_spectrogram_augmentations",
    "frequency_mask",
    "frequency_mixstyle",
    "make_spectrogram_augmentation_stage",
    "mel_bin_shift",
    "mixstyle",
    "normalize_spectrogram_augment_specs",
    "random_eq",
    "spectrogram_time_roll",
    "time_frequency_erasing",
    "time_mask",
]

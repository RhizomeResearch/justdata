from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import tensorflow as tf

from justdata.acoustic.augment.configs import (
    AudioBatchMixStyleConfig,
    AudioCutMixSpecConfig,
    AudioMixupConfig,
    AudioWavMixConfig,
)
from justdata.acoustic.augment.spectrogram import (
    BatchSpectrogramLayout,
    _as_btfc,
    _restore_from_btfc,
)
from justdata.acoustic.registry import (
    get_audio_batch_augment,
    register_audio_batch_augment,
)
from justdata.acoustic.schema import FEATURES, LABEL, WAVEFORM
from justdata.core.label_mixing import (
    LabelMixMode,
    blend_prepared_labels,
    mix_labels,
    prepare_labels_for_mixing,
)


_EPS = tf.constant(1e-6, dtype=tf.float32)

_ALIASES = {
    "cutmix": "cutmix_spec",
    "mixstyle": "batch_mixstyle",
}
_LABEL_MIXING_AUGMENTS = {"mixup", "cutmix_spec", "wavmix"}


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


def _merge_config(
    config: Any,
    config_cls: type,
    defaults: Mapping[str, Any],
    overrides: Mapping[str, Any],
):
    data = dict(defaults)
    if config is not None:
        if isinstance(config, config_cls):
            data.update(dataclasses.asdict(config))
        elif dataclasses.is_dataclass(config):
            data.update(dataclasses.asdict(config))
        elif isinstance(config, Mapping):
            data.update(config)
        else:
            raise TypeError(f"config must be a {config_cls.__name__}, mapping, or None")
    data.update({key: value for key, value in overrides.items() if value is not None})
    return config_cls(**data)


def _sample_beta(shape: tf.Tensor | Sequence[int], alpha: float, seed: tf.Tensor) -> tf.Tensor:
    first_seed, second_seed = tf.unstack(tf.random.split(seed, 2))
    first = tf.random.stateless_gamma(shape, seed=first_seed, alpha=alpha, dtype=tf.float32)
    second = tf.random.stateless_gamma(shape, seed=second_seed, alpha=alpha, dtype=tf.float32)
    return first / (first + second + _EPS)


def _sample_lambda(alpha: float, seed: tf.Tensor, keep_max_lambda: bool) -> tf.Tensor:
    lam = _sample_beta([], alpha, seed)
    if keep_max_lambda:
        lam = tf.maximum(lam, 1.0 - lam)
    return lam


def _stateless_shuffled_range(size: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
    size = tf.cast(size, tf.int32)
    order = tf.argsort(
        tf.random.stateless_uniform(tf.reshape(size, [1]), seed=seed),
        stable=True,
    )
    identity = tf.range(size)
    is_identity = tf.reduce_all(tf.equal(order, identity))
    return tf.cond(
        tf.logical_and(size > 1, is_identity),
        lambda: tf.roll(order, shift=1, axis=0),
        lambda: order,
    )


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


def _mix_tensor(x: tf.Tensor, partner: tf.Tensor, lam: tf.Tensor) -> tf.Tensor:
    dtype = x.dtype
    mixed = tf.cast(lam, tf.float32) * tf.cast(x, tf.float32)
    mixed += (1.0 - tf.cast(lam, tf.float32)) * tf.cast(partner, tf.float32)
    return tf.cast(mixed, dtype)


def _maybe_apply_pair(
    x: tf.Tensor,
    labels: tf.Tensor,
    prob: float,
    seed: tf.Tensor,
    apply_fn,
    skip_fn,
) -> tuple[tf.Tensor, tf.Tensor]:
    if prob <= 0.0:
        return x, labels
    if prob >= 1.0:
        return apply_fn()
    should_apply = tf.random.stateless_uniform([], seed=seed) < tf.cast(prob, tf.float32)
    return tf.cond(should_apply, apply_fn, skip_fn)


def _maybe_apply_tensor(
    x: tf.Tensor,
    prob: float,
    seed: tf.Tensor,
    apply_fn,
) -> tf.Tensor:
    if prob <= 0.0:
        return x
    if prob >= 1.0:
        return apply_fn()
    should_apply = tf.random.stateless_uniform([], seed=seed) < tf.cast(prob, tf.float32)
    return tf.cond(should_apply, apply_fn, lambda: x)


@register_audio_batch_augment("mixup", requires_labels=True)
def audio_mixup(
    inputs: tf.Tensor,
    labels: tf.Tensor,
    *,
    seed: tf.Tensor | int | None = None,
    config: AudioMixupConfig | Mapping[str, Any] | None = None,
    alpha: float | None = None,
    prob: float | None = None,
    label_mode: LabelMixMode | None = None,
    num_classes: int | None = None,
    keep_max_lambda: bool = True,
    is_training: bool = True,
    augment_eval: bool = False,
) -> tuple[tf.Tensor, tf.Tensor]:
    if not _enabled(is_training, augment_eval):
        return tf.convert_to_tensor(inputs), tf.convert_to_tensor(labels)

    cfg = _merge_config(
        config,
        AudioMixupConfig,
        {"alpha": 0.8, "prob": 1.0, "label_mode": label_mode or "single_label"},
        {"alpha": alpha, "prob": prob, "label_mode": label_mode},
    )
    x = tf.convert_to_tensor(inputs)
    y = tf.convert_to_tensor(labels)
    if cfg.prob <= 0.0:
        return x, y

    seed = _seed_tensor(seed)
    gate_seed, shuffle_seed, beta_seed = tf.unstack(tf.random.split(seed, 3))
    prepared_labels = prepare_labels_for_mixing(
        y,
        label_mode=cfg.label_mode,
        num_classes=num_classes,
    )

    def apply() -> tuple[tf.Tensor, tf.Tensor]:
        batch = tf.shape(x)[0]
        partner = _stateless_shuffled_range(batch, shuffle_seed)
        lam = _sample_lambda(cfg.alpha, beta_seed, keep_max_lambda)
        mixed_x = _mix_tensor(x, tf.gather(x, partner, axis=0), lam)
        mixed_y = mix_labels(
            y,
            tf.gather(y, partner, axis=0),
            lam,
            label_mode=cfg.label_mode,
            num_classes=num_classes,
        )
        return mixed_x, mixed_y

    return _maybe_apply_pair(x, prepared_labels, cfg.prob, gate_seed, apply, lambda: (x, prepared_labels))


@register_audio_batch_augment("wavmix", requires_labels=True)
def audio_wavmix(
    waveform: tf.Tensor,
    labels: tf.Tensor | None = None,
    *,
    seed: tf.Tensor | int | None = None,
    config: AudioWavMixConfig | Mapping[str, Any] | None = None,
    alpha: float | None = None,
    prob: float | None = None,
    label_mode: LabelMixMode | None = None,
    num_classes: int | None = None,
    input_kind: Literal["waveform", "spectrogram"] = "waveform",
    keep_max_lambda: bool = True,
    is_training: bool = True,
    augment_eval: bool = False,
):
    if input_kind != "waveform":
        raise ValueError("WavMix only supports waveform inputs; got input_kind != 'waveform'.")

    x = tf.convert_to_tensor(waveform)
    if x.shape.rank not in (2, 3):
        raise ValueError("WavMix input must have shape [B, T] or [B, T, C].")

    if not _enabled(is_training, augment_eval):
        return (x, tf.convert_to_tensor(labels)) if labels is not None else x

    cfg = _merge_config(
        config,
        AudioWavMixConfig,
        {"alpha": 0.8, "prob": 1.0},
        {"alpha": alpha, "prob": prob},
    )
    if cfg.prob <= 0.0:
        return (x, tf.convert_to_tensor(labels)) if labels is not None else x

    seed = _seed_tensor(seed)
    gate_seed, shuffle_seed, beta_seed = tf.unstack(tf.random.split(seed, 3))

    def mix_waveform() -> tf.Tensor:
        partner = _stateless_shuffled_range(tf.shape(x)[0], shuffle_seed)
        lam = _sample_lambda(cfg.alpha, beta_seed, keep_max_lambda)
        return _mix_tensor(x, tf.gather(x, partner, axis=0), lam)

    if labels is None:
        return _maybe_apply_tensor(x, cfg.prob, gate_seed, mix_waveform)

    y = tf.convert_to_tensor(labels)
    resolved_label_mode = label_mode or "single_label"
    prepared_labels = prepare_labels_for_mixing(
        y,
        label_mode=resolved_label_mode,
        num_classes=num_classes,
    )

    def apply() -> tuple[tf.Tensor, tf.Tensor]:
        partner = _stateless_shuffled_range(tf.shape(x)[0], shuffle_seed)
        lam = _sample_lambda(cfg.alpha, beta_seed, keep_max_lambda)
        return (
            _mix_tensor(x, tf.gather(x, partner, axis=0), lam),
            mix_labels(
                y,
                tf.gather(y, partner, axis=0),
                lam,
                label_mode=resolved_label_mode,
                num_classes=num_classes,
            ),
        )

    return _maybe_apply_pair(x, prepared_labels, cfg.prob, gate_seed, apply, lambda: (x, prepared_labels))


def _selected_width(size: tf.Tensor, fraction: tf.Tensor) -> tf.Tensor:
    size = tf.cast(size, tf.int32)
    width = tf.cast(tf.round(tf.cast(size, tf.float32) * fraction), tf.int32)
    width = tf.clip_by_value(width, 0, size)
    return tf.where(size > 0, tf.maximum(width, 1), width)


def _cutmix_widths(
    time: tf.Tensor,
    frequency: tf.Tensor,
    lam: tf.Tensor,
    axes: Literal["time", "frequency", "time_frequency"],
) -> tuple[tf.Tensor, tf.Tensor]:
    cut_fraction = 1.0 - tf.cast(lam, tf.float32)
    if axes == "time":
        return _selected_width(time, cut_fraction), tf.cast(frequency, tf.int32)
    if axes == "frequency":
        return tf.cast(time, tf.int32), _selected_width(frequency, cut_fraction)
    if axes == "time_frequency":
        side_fraction = tf.sqrt(cut_fraction)
        return _selected_width(time, side_fraction), _selected_width(frequency, side_fraction)
    raise ValueError("axes must be one of 'time', 'frequency', or 'time_frequency'")


def _spliced_event_labels(
    labels: tf.Tensor,
    partner_labels: tf.Tensor,
    time_mask: tf.Tensor,
    spectrogram_time: tf.Tensor,
) -> tf.Tensor:
    labels = tf.cast(labels, tf.float32)
    partner_labels = tf.cast(partner_labels, tf.float32)
    if labels.shape.rank != 3:
        raise ValueError("event_frames labels must have shape [B, T_frames, C].")
    if (
        labels.shape[1] is not None
        and time_mask.shape[0] is not None
        and labels.shape[1] != time_mask.shape[0]
    ):
        raise ValueError(
            "CutMixSpec event-frame splicing requires label frames to match "
            "spectrogram time bins."
        )
    with tf.control_dependencies(
        [
            tf.debugging.assert_equal(
                tf.shape(labels)[1],
                spectrogram_time,
                message=(
                    "CutMixSpec event-frame splicing requires label frames to match "
                    "spectrogram time bins."
                ),
            )
        ]
    ):
        return tf.where(time_mask[tf.newaxis, :, tf.newaxis], partner_labels, labels)


@register_audio_batch_augment("cutmix_spec", requires_labels=True)
def audio_cutmix_spec(
    spectrogram: tf.Tensor,
    labels: tf.Tensor,
    *,
    seed: tf.Tensor | int | None = None,
    config: AudioCutMixSpecConfig | Mapping[str, Any] | None = None,
    alpha: float | None = None,
    prob: float | None = None,
    axes: Literal["time", "frequency", "time_frequency"] | None = None,
    label_mode: LabelMixMode | None = None,
    num_classes: int | None = None,
    layout: BatchSpectrogramLayout | None = None,
    input_kind: Literal["waveform", "spectrogram"] = "spectrogram",
    keep_max_lambda: bool = True,
    is_training: bool = True,
    augment_eval: bool = False,
) -> tuple[tf.Tensor, tf.Tensor]:
    if input_kind != "spectrogram":
        raise ValueError("CutMixSpec only supports spectrogram inputs.")
    if not _enabled(is_training, augment_eval):
        return tf.convert_to_tensor(spectrogram), tf.convert_to_tensor(labels)

    cfg = _merge_config(
        config,
        AudioCutMixSpecConfig,
        {"alpha": 1.0, "prob": 1.0, "axes": axes or "time_frequency"},
        {"alpha": alpha, "prob": prob, "axes": axes},
    )

    original = tf.convert_to_tensor(spectrogram)
    y = tf.convert_to_tensor(labels)
    if cfg.prob <= 0.0:
        return original, y

    x, resolved_layout = _as_btfc(original, layout)
    seed = _seed_tensor(seed)
    gate_seed, shuffle_seed, beta_seed, time_seed, freq_seed = tf.unstack(
        tf.random.split(seed, 5)
    )
    resolved_label_mode = label_mode or "single_label"
    if (
        resolved_label_mode == "event_frames"
        and y.shape.rank == 3
        and x.shape[1] is not None
        and y.shape[1] is not None
        and x.shape[1] != y.shape[1]
    ):
        raise ValueError(
            "CutMixSpec event-frame labels must match spectrogram time bins unless "
            "an explicit alignment config is provided."
        )
    prepared_labels = prepare_labels_for_mixing(
        y,
        label_mode=resolved_label_mode,
        num_classes=num_classes,
    )

    def apply() -> tuple[tf.Tensor, tf.Tensor]:
        batch = tf.shape(x)[0]
        time = tf.shape(x)[1]
        freq = tf.shape(x)[2]
        partner = _stateless_shuffled_range(batch, shuffle_seed)
        lam = _sample_lambda(cfg.alpha, beta_seed, keep_max_lambda)
        time_width, freq_width = _cutmix_widths(time, freq, lam, cfg.axes)
        time_start = _uniform_int(time_seed, tf.constant(0, tf.int32), tf.maximum(time - time_width, 0))
        freq_start = _uniform_int(freq_seed, tf.constant(0, tf.int32), tf.maximum(freq - freq_width, 0))

        time_positions = tf.range(time)
        freq_positions = tf.range(freq)
        time_mask = tf.logical_and(
            time_positions >= time_start,
            time_positions < time_start + time_width,
        )
        freq_mask = tf.logical_and(
            freq_positions >= freq_start,
            freq_positions < freq_start + freq_width,
        )
        mask = tf.logical_and(time_mask[:, tf.newaxis], freq_mask[tf.newaxis, :])
        mixed = tf.where(
            mask[tf.newaxis, :, :, tf.newaxis],
            tf.gather(x, partner, axis=0),
            x,
        )

        replaced_area = tf.cast(time_width * freq_width, tf.float32)
        total_area = tf.cast(time * freq, tf.float32)
        lam_effective = 1.0 - tf.math.divide_no_nan(replaced_area, total_area)
        partner_y = tf.gather(y, partner, axis=0)
        if resolved_label_mode == "event_frames" and cfg.axes == "time":
            mixed_labels = _spliced_event_labels(y, partner_y, time_mask, time)
        else:
            mixed_labels = mix_labels(
                y,
                partner_y,
                lam_effective,
                label_mode=resolved_label_mode,
                num_classes=num_classes,
            )
        return _restore_from_btfc(mixed, resolved_layout), mixed_labels

    return _maybe_apply_pair(
        original,
        prepared_labels,
        cfg.prob,
        gate_seed,
        apply,
        lambda: (original, prepared_labels),
    )


@register_audio_batch_augment("batch_mixstyle")
def audio_batch_mixstyle(
    spectrogram: tf.Tensor,
    labels: tf.Tensor | None = None,
    *,
    seed: tf.Tensor | int | None = None,
    config: AudioBatchMixStyleConfig | Mapping[str, Any] | None = None,
    prob: float | None = None,
    mix: Literal["global", "frequency", "channel"] | None = None,
    alpha: float = 0.1,
    layout: BatchSpectrogramLayout | None = None,
    input_kind: Literal["waveform", "spectrogram"] = "spectrogram",
    is_training: bool = True,
    augment_eval: bool = False,
):
    if input_kind != "spectrogram":
        raise ValueError("Batch MixStyle only supports spectrogram inputs.")

    original = tf.convert_to_tensor(spectrogram)
    if not _enabled(is_training, augment_eval):
        return (original, labels) if labels is not None else original

    cfg = _merge_config(
        config,
        AudioBatchMixStyleConfig,
        {"prob": 1.0, "mix": mix or "global"},
        {"prob": prob, "mix": mix},
    )
    x, resolved_layout = _as_btfc(original, layout)
    seed = _seed_tensor(seed)
    gate_seed, shuffle_seed, beta_seed = tf.unstack(tf.random.split(seed, 3))
    stat_axes = {
        "global": (1, 2, 3),
        "frequency": (1, 3),
        "channel": (1, 2),
    }[cfg.mix]

    def apply() -> tf.Tensor:
        batch = tf.shape(x)[0]

        def mix_batch() -> tf.Tensor:
            x_float = tf.cast(x, tf.float32)
            mean = tf.reduce_mean(x_float, axis=stat_axes, keepdims=True)
            var = tf.reduce_mean(tf.square(x_float - mean), axis=stat_axes, keepdims=True)
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
            return tf.cast(mixed, x.dtype)

        return tf.cond(batch > 1, mix_batch, lambda: x)

    mixed = _maybe_apply_tensor(x, cfg.prob, gate_seed, apply)
    restored = _restore_from_btfc(mixed, resolved_layout)
    return (restored, labels) if labels is not None else restored


def _canonical_name(name: str) -> str:
    return _ALIASES.get(name, name)


def normalize_batch_augment_specs(
    augmentations: Mapping[str, Any] | Sequence[Any] | str | None,
) -> list[dict[str, Any]]:
    if augmentations is None:
        return []
    if isinstance(augmentations, str):
        return [{"name": _canonical_name(augmentations)}]
    if isinstance(augmentations, Mapping):
        if "name" in augmentations:
            spec = dict(augmentations)
            spec["name"] = _canonical_name(str(spec["name"]))
            return [spec]

        specs = []
        for name, value in augmentations.items():
            if value is None or value is False:
                continue
            canonical = _canonical_name(str(name))
            if value is True:
                specs.append({"name": canonical})
            elif isinstance(value, Mapping):
                specs.append({"name": canonical, **dict(value)})
            else:
                specs.append({"name": canonical, "config": value})
        return specs

    specs = []
    for value in augmentations:
        specs.extend(normalize_batch_augment_specs(value))
    return specs


def _label_transform_data(label_transform: Any) -> Mapping[str, Any] | None:
    if label_transform is None:
        return None
    if dataclasses.is_dataclass(label_transform):
        return dataclasses.asdict(label_transform)
    if isinstance(label_transform, Mapping):
        return label_transform
    return None


def _label_mode_from_transform(label_transform: Any) -> str | None:
    data = _label_transform_data(label_transform)
    if data is None:
        return None
    mode = data.get("mode")
    if mode in {"index", "one_hot"}:
        return "single_label"
    if mode == "multi_hot":
        return "multi_label"
    if mode == "event_frames":
        return "event_frames"
    if mode == "text":
        return "text"
    return None


def _num_classes_from_transform(label_transform: Any) -> int | None:
    data = _label_transform_data(label_transform)
    if data is None:
        return None
    return data.get("num_classes")


def _config_has_key(config: Any, key: str) -> bool:
    if config is None:
        return False
    if dataclasses.is_dataclass(config):
        return hasattr(config, key)
    if isinstance(config, Mapping):
        return key in config
    return False


def _is_text_labels(labels: Any, label_mode: str | None) -> bool:
    if label_mode == "text":
        return True
    if labels is None:
        return False
    try:
        return tf.convert_to_tensor(labels).dtype == tf.string
    except (TypeError, ValueError):
        return False


def _resolve_input_key(
    sample: Mapping[str, Any],
    name: str,
    input_key: str | None,
    input_kind: str | None,
) -> str | None:
    if input_key is not None:
        return input_key
    if name == "wavmix":
        return WAVEFORM if WAVEFORM in sample else None
    if name in {"cutmix_spec", "batch_mixstyle"}:
        return FEATURES if FEATURES in sample else None
    if input_kind == "waveform" and WAVEFORM in sample:
        return WAVEFORM
    if input_kind == "spectrogram" and FEATURES in sample:
        return FEATURES
    if FEATURES in sample:
        return FEATURES
    if WAVEFORM in sample:
        return WAVEFORM
    return None


def _input_kind_from_key(input_key: str, input_kind: str | None) -> str:
    if input_kind is not None:
        return input_kind
    return "waveform" if input_key == WAVEFORM else "spectrogram"


def apply_batch_augmentations(
    sample: Mapping[str, Any],
    augmentations: Mapping[str, Any] | Sequence[Any] | str | None,
    *,
    seed: tf.Tensor | int | None = None,
    is_training: bool = True,
    augment_eval: bool = False,
    input_key: str | None = None,
    input_kind: Literal["waveform", "spectrogram"] | None = None,
    label_mode: LabelMixMode | Literal["text"] | None = None,
    num_classes: int | None = None,
    label_transform: Mapping[str, Any] | None = None,
    spectrogram_layout: BatchSpectrogramLayout | None = None,
) -> dict:
    specs = normalize_batch_augment_specs(augmentations)
    if not specs or not _enabled(is_training, augment_eval):
        return dict(sample)

    resolved_label_mode = label_mode or _label_mode_from_transform(label_transform) or "single_label"
    resolved_num_classes = num_classes
    if resolved_num_classes is None:
        resolved_num_classes = _num_classes_from_transform(label_transform)
    seed = _seed_tensor(seed)
    seeds = tf.random.split(seed, len(specs))
    result = dict(sample)

    for spec, spec_seed in zip(specs, tf.unstack(seeds)):
        kwargs = dict(spec)
        name = kwargs.pop("name")
        key = _resolve_input_key(result, name, kwargs.pop("input_key", input_key), input_kind)
        if key is None or key not in result:
            continue

        labels = result.get(LABEL)
        if name in _LABEL_MIXING_AUGMENTS:
            if LABEL not in result or _is_text_labels(labels, resolved_label_mode):
                continue
            if "label_mode" not in kwargs and not _config_has_key(
                kwargs.get("config"),
                "label_mode",
            ):
                kwargs["label_mode"] = resolved_label_mode
            kwargs.setdefault("num_classes", resolved_num_classes)

        kind = _input_kind_from_key(key, kwargs.pop("input_kind", input_kind))
        fn = get_audio_batch_augment(name)
        if name == "batch_mixstyle":
            output = fn(
                result[key],
                labels,
                seed=spec_seed,
                layout=kwargs.pop("layout", spectrogram_layout),
                input_kind=kind,
                is_training=is_training,
                augment_eval=augment_eval,
                **kwargs,
            )
            if isinstance(output, tuple):
                result[key], result[LABEL] = output
            else:
                result[key] = output
        elif name == "cutmix_spec":
            result[key], result[LABEL] = fn(
                result[key],
                labels,
                seed=spec_seed,
                layout=kwargs.pop("layout", spectrogram_layout),
                input_kind=kind,
                is_training=is_training,
                augment_eval=augment_eval,
                **kwargs,
            )
        elif name == "wavmix":
            result[key], result[LABEL] = fn(
                result[key],
                labels,
                seed=spec_seed,
                input_kind=kind,
                is_training=is_training,
                augment_eval=augment_eval,
                **kwargs,
            )
        else:
            result[key], result[LABEL] = fn(
                result[key],
                labels,
                seed=spec_seed,
                is_training=is_training,
                augment_eval=augment_eval,
                **kwargs,
            )

    return result


def make_batch_augmentation_stage(
    augmentations: Mapping[str, Any] | Sequence[Any] | str | None,
    *,
    is_training: bool = True,
    augment_eval: bool = False,
    input_key: str | None = None,
    input_kind: Literal["waveform", "spectrogram"] | None = None,
    label_mode: LabelMixMode | Literal["text"] | None = None,
    label_transform: Mapping[str, Any] | None = None,
    spectrogram_layout: BatchSpectrogramLayout | None = None,
):
    specs = normalize_batch_augment_specs(augmentations)

    def stage(sample: Mapping[str, Any], num_classes=None, seed=None) -> dict:
        return apply_batch_augmentations(
            sample,
            specs,
            seed=seed,
            is_training=is_training,
            augment_eval=augment_eval,
            input_key=input_key,
            input_kind=input_kind,
            label_mode=label_mode,
            num_classes=num_classes,
            label_transform=label_transform,
            spectrogram_layout=spectrogram_layout,
        )

    return stage


cutmix_spec = audio_cutmix_spec
mixup = audio_mixup
wavmix = audio_wavmix
batch_mixstyle = audio_batch_mixstyle


__all__ = [
    "AudioBatchMixStyleConfig",
    "AudioCutMixSpecConfig",
    "AudioMixupConfig",
    "AudioWavMixConfig",
    "apply_batch_augmentations",
    "audio_batch_mixstyle",
    "audio_cutmix_spec",
    "audio_mixup",
    "audio_wavmix",
    "batch_mixstyle",
    "cutmix_spec",
    "make_batch_augmentation_stage",
    "mixup",
    "normalize_batch_augment_specs",
    "wavmix",
]

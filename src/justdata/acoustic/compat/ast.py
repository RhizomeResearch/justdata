from __future__ import annotations

from typing import Any, Mapping

import tensorflow as tf

from justdata.acoustic.configs import (
    AudioPreprocessConfig,
    AudioPreset,
    FrontendConfig,
    LabelTransformConfig,
    LogCompressionConfig,
    MelConfig,
    STFTConfig,
    SegmentStrategyConfig,
)
from justdata.acoustic.labels import transform_label
from justdata.acoustic.layouts import convert_audio_layout
from justdata.acoustic.preprocessing import make_preprocessing
from justdata.acoustic.schema import FEATURES, LABEL, METADATA, SAMPLE_RATE, WAVEFORM
from justdata.acoustic.segment import segment_waveform
from justdata.acoustic.frontends.ast import ast_kaldi_fbank


AST_REFERENCE_COMMIT = "31088be8a3f6ef96416145c4b8d43c81f99eba7a"
AST_MIX_WAVEFORM = "ast_mix_waveform"
AST_MIX_LABEL = "ast_mix_label"
AST_MIX_LAMBDA = "ast_mix_lambda"

AST_RECIPE_CONFIGS = {
    "audioset": {
        "target_length": 1024,
        "freqm": 48,
        "timem": 192,
        "mixup": 0.5,
        "mean": -4.2677393,
        "std": 4.5689974,
        "noise": False,
        "input_duration": 10.0,
        "num_classes": 527,
    },
    "esc50": {
        "target_length": 512,
        "freqm": 24,
        "timem": 96,
        "mixup": 0.0,
        "mean": -6.6268077,
        "std": 5.358466,
        "noise": False,
        "input_duration": 5.0,
        "num_classes": 50,
    },
    "speechcommands": {
        "target_length": 128,
        "freqm": 48,
        "timem": 48,
        "mixup": 0.6,
        "mean": -6.845978,
        "std": 5.5654526,
        "noise": True,
        "input_duration": 1.0,
        "num_classes": 35,
    },
}


def _seed_tensor(seed: tf.Tensor | int | None) -> tf.Tensor:
    if seed is None:
        return tf.constant([0, 0], dtype=tf.int32)

    seed = tf.cast(tf.convert_to_tensor(seed), tf.int32)
    if seed.shape.rank == 0:
        return tf.stack([seed, tf.constant(0, dtype=tf.int32)])
    if seed.shape.rank == 1 and seed.shape[0] == 1:
        return tf.stack([seed[0], tf.constant(0, dtype=tf.int32)])
    return seed[:2]


def _uniform_int(seed: tf.Tensor, minval: tf.Tensor, maxval_exclusive: tf.Tensor) -> tf.Tensor:
    minval = tf.cast(minval, tf.int32)
    maxval_exclusive = tf.cast(maxval_exclusive, tf.int32)
    return tf.cond(
        maxval_exclusive <= minval,
        lambda: minval,
        lambda: tf.random.stateless_uniform(
            [],
            seed=seed,
            minval=minval,
            maxval=maxval_exclusive,
            dtype=tf.int32,
        ),
    )


def _sample_beta(seed: tf.Tensor, alpha: float = 10.0) -> tf.Tensor:
    first_seed, second_seed = tf.unstack(tf.random.split(seed, 2))
    first = tf.random.stateless_gamma([], seed=first_seed, alpha=alpha, dtype=tf.float32)
    second = tf.random.stateless_gamma([], seed=second_seed, alpha=alpha, dtype=tf.float32)
    return first / (first + second + tf.constant(1e-6, dtype=tf.float32))


def ast_frontend(num_mel_bins: int = 128) -> FrontendConfig:
    return FrontendConfig(
        name="ast_kaldi_fbank",
        stft=STFTConfig(
            sample_rate=16000,
            n_fft=512,
            win_length=400,
            hop_length=160,
            window="hann",
            window_periodic=False,
            center=False,
            power=2.0,
        ),
        mel=MelConfig(
            n_mels=num_mel_bins,
            f_min=20.0,
            f_max=None,
            mel_scale="htk",
            mel_norm="none",
            filterbank_impl="kaldi_compatible",
        ),
        log=LogCompressionConfig(kind="log"),
    )


def ast_pad_or_crop_fbank(fbank: tf.Tensor, target_length: int | tf.Tensor) -> tf.Tensor:
    fbank = tf.convert_to_tensor(fbank)
    target = tf.cast(target_length, tf.int32)
    current = tf.shape(fbank)[0]
    cropped = fbank[:target]
    pad = tf.maximum(target - current, 0)
    return tf.pad(cropped, [[0, pad], [0, 0], [0, 0]])


def ast_normalize_fbank(fbank: tf.Tensor, mean: float, std: float) -> tf.Tensor:
    fbank = tf.cast(tf.convert_to_tensor(fbank), tf.float32)
    return (fbank - tf.cast(mean, tf.float32)) / (tf.cast(std, tf.float32) * 2.0)


def _sample_mask_span(
    axis_size: tf.Tensor,
    max_width: int,
    width_seed: tf.Tensor,
    start_seed: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    limit = tf.minimum(tf.cast(max_width, tf.int32), tf.cast(axis_size, tf.int32))
    width = _uniform_int(width_seed, tf.constant(0, tf.int32), limit)
    max_start = tf.maximum(tf.cast(axis_size, tf.int32) - width, 0)
    start = _uniform_int(start_seed, tf.constant(0, tf.int32), max_start + 1)
    return start, width


def ast_frequency_mask(
    fbank: tf.Tensor,
    *,
    max_width: int,
    seed: tf.Tensor | int | None = None,
    start: int | tf.Tensor | None = None,
    width: int | tf.Tensor | None = None,
) -> tf.Tensor:
    fbank = tf.convert_to_tensor(fbank)
    if max_width <= 0:
        return fbank

    seed = _seed_tensor(seed)
    width_seed, start_seed = tf.unstack(tf.random.split(seed, 2))
    freq = tf.shape(fbank)[1]
    if start is None or width is None:
        sampled_start, sampled_width = _sample_mask_span(
            freq,
            max_width,
            width_seed,
            start_seed,
        )
    else:
        sampled_start = tf.cast(start, tf.int32)
        sampled_width = tf.cast(width, tf.int32)

    positions = tf.range(freq)
    mask = tf.logical_and(
        positions >= sampled_start,
        positions < sampled_start + sampled_width,
    )
    return tf.where(mask[tf.newaxis, :, tf.newaxis], tf.zeros([], dtype=fbank.dtype), fbank)


def ast_time_mask(
    fbank: tf.Tensor,
    *,
    max_width: int,
    seed: tf.Tensor | int | None = None,
    start: int | tf.Tensor | None = None,
    width: int | tf.Tensor | None = None,
) -> tf.Tensor:
    fbank = tf.convert_to_tensor(fbank)
    if max_width <= 0:
        return fbank

    seed = _seed_tensor(seed)
    width_seed, start_seed = tf.unstack(tf.random.split(seed, 2))
    time = tf.shape(fbank)[0]
    if start is None or width is None:
        sampled_start, sampled_width = _sample_mask_span(
            time,
            max_width,
            width_seed,
            start_seed,
        )
    else:
        sampled_start = tf.cast(start, tf.int32)
        sampled_width = tf.cast(width, tf.int32)

    positions = tf.range(time)
    mask = tf.logical_and(
        positions >= sampled_start,
        positions < sampled_start + sampled_width,
    )
    return tf.where(mask[:, tf.newaxis, tf.newaxis], tf.zeros([], dtype=fbank.dtype), fbank)


def ast_specaugment(
    fbank: tf.Tensor,
    *,
    freqm: int,
    timem: int,
    seed: tf.Tensor | int | None = None,
) -> tf.Tensor:
    seed = _seed_tensor(seed)
    freq_seed, time_seed = tf.unstack(tf.random.split(seed, 2))
    fbank = ast_frequency_mask(fbank, max_width=freqm, seed=freq_seed)
    return ast_time_mask(fbank, max_width=timem, seed=time_seed)


def ast_add_noise_and_roll(
    fbank: tf.Tensor,
    *,
    seed: tf.Tensor | int | None = None,
    noise_scale: float | tf.Tensor | None = None,
    roll_shift: int | tf.Tensor | None = None,
) -> tf.Tensor:
    seed = _seed_tensor(seed)
    scale_seed, noise_seed, roll_seed = tf.unstack(tf.random.split(seed, 3))
    fbank = tf.convert_to_tensor(fbank)

    if noise_scale is None:
        scale = tf.random.stateless_uniform([], seed=scale_seed, minval=0.0, maxval=0.1)
    else:
        scale = tf.cast(noise_scale, tf.float32)
    noise = tf.random.stateless_uniform(tf.shape(fbank), seed=noise_seed, dtype=tf.float32)
    fbank = fbank + tf.cast(noise * scale, fbank.dtype)

    if roll_shift is None:
        shift = _uniform_int(roll_seed, tf.constant(-10, tf.int32), tf.constant(10, tf.int32))
    else:
        shift = tf.cast(roll_shift, tf.int32)
    return tf.roll(fbank, shift=shift, axis=0)


def _metadata_for_recipe(recipe_name: str) -> dict[str, Any]:
    recipe = AST_RECIPE_CONFIGS[recipe_name]
    return {
        "preset_version": 1,
        "model_family": "ast",
        "recipe": recipe_name,
        "frontend_contract": "ast-kaldi-fbank-v1",
        "compatibility_status": "reference_frontend_declared_golden_pending",
        "reference_repo": "YuanGongND/ast",
        "reference_commit": AST_REFERENCE_COMMIT,
        "ast": {
            key: recipe[key]
            for key in ("target_length", "freqm", "timem", "mixup", "mean", "std", "noise")
        },
    }


def _preprocess() -> AudioPreprocessConfig:
    return AudioPreprocessConfig(
        target_sample_rate=16000,
        channel_strategy="mono_mean",
        normalize_waveform="none",
    )


def _segment(duration: float) -> SegmentStrategyConfig:
    return SegmentStrategyConfig(
        clip_duration=duration,
        train_mode="full",
        eval_mode="full",
        pad_mode="zero",
        pad_position="right",
    )


def _recipe_preset(name: str, recipe_name: str) -> AudioPreset:
    recipe = AST_RECIPE_CONFIGS[recipe_name]
    return AudioPreset(
        name=name,
        input_duration=recipe["input_duration"],
        target_sample_rate=16000,
        preprocess=_preprocess(),
        segment=_segment(recipe["input_duration"]),
        frontend=ast_frontend(),
        label_transform=LabelTransformConfig(
            mode="multi_hot",
            num_classes=recipe["num_classes"],
        ),
        layout="btf",
        static_shape=(recipe["target_length"], 128),
        train_augment={"ast": _metadata_for_recipe(recipe_name)["ast"]},
        metadata=_metadata_for_recipe(recipe_name),
    )


def ast_audioset_16k_10s_fbank128() -> AudioPreset:
    return _recipe_preset("ast_audioset_16k_10s_fbank128", "audioset")


def ast_esc50_16k_5s_fbank128() -> AudioPreset:
    return _recipe_preset("ast_esc50_16k_5s_fbank128", "esc50")


def ast_speechcommands_16k_1s_fbank128() -> AudioPreset:
    return _recipe_preset("ast_speechcommands_16k_1s_fbank128", "speechcommands")


def _ast_config(metadata: Mapping[str, Any] | None, train_augment: Mapping[str, Any] | None) -> dict:
    ast = {}
    if isinstance(metadata, Mapping) and isinstance(metadata.get("ast"), Mapping):
        ast.update(metadata["ast"])
    if isinstance(train_augment, Mapping) and isinstance(train_augment.get("ast"), Mapping):
        ast.update(train_augment["ast"])
    return ast


def _target_length(layout: str, static_shape: tuple[int | None, ...] | None, ast: Mapping[str, Any]) -> int:
    if "target_length" in ast:
        return int(ast["target_length"])
    if static_shape is None:
        raise ValueError("AST pipeline requires static_shape or metadata['ast']['target_length']")
    if layout == "btf":
        return int(static_shape[0])
    if layout == "bft":
        return int(static_shape[1])
    if layout == "bcft":
        return int(static_shape[2])
    if layout == "btfc":
        return int(static_shape[0])
    raise ValueError(f"Unsupported AST layout: {layout!r}")


def _transform_sample_label(result: dict, label_config: LabelTransformConfig | None) -> dict:
    if label_config is None or LABEL not in result:
        return result
    label, metadata = transform_label(
        result[LABEL],
        label_config,
        metadata=result.get(METADATA),
    )
    result = dict(result)
    result[LABEL] = label
    result[METADATA] = metadata
    return result


def _prepare_waveform_for_ast_mix(waveform: tf.Tensor) -> tf.Tensor:
    waveform = tf.cast(tf.convert_to_tensor(waveform), tf.float32)
    if waveform.shape.rank == 1:
        waveform = waveform[:, tf.newaxis]
    return waveform - tf.reduce_mean(waveform)


def ast_mix_waveforms(
    waveform: tf.Tensor,
    partner_waveform: tf.Tensor,
    mix_lambda: tf.Tensor | float,
) -> tf.Tensor:
    waveform = _prepare_waveform_for_ast_mix(waveform)
    partner = _prepare_waveform_for_ast_mix(partner_waveform)
    target = tf.shape(waveform)[0]
    partner_len = tf.shape(partner)[0]

    def _pad_partner() -> tf.Tensor:
        return tf.pad(partner, [[0, target - partner_len], [0, 0]])

    partner = tf.cond(partner_len < target, _pad_partner, lambda: partner[:target])
    lam = tf.cast(mix_lambda, tf.float32)
    mixed = lam * waveform + (1.0 - lam) * partner
    return mixed - tf.reduce_mean(mixed)


def _maybe_mix_waveform_and_label(
    result: dict,
    *,
    label_config: LabelTransformConfig | None,
    seed: tf.Tensor,
) -> dict:
    if AST_MIX_WAVEFORM not in result:
        return _transform_sample_label(result, label_config)

    lam = result.get(AST_MIX_LAMBDA)
    if lam is None:
        lam = _sample_beta(seed, alpha=10.0)
    lam = tf.cast(lam, tf.float32)

    mixed = ast_mix_waveforms(result[WAVEFORM], result[AST_MIX_WAVEFORM], lam)
    result = dict(result)
    result[WAVEFORM] = mixed

    if label_config is not None and LABEL in result:
        label, metadata = transform_label(
            result[LABEL],
            label_config,
            metadata=result.get(METADATA),
        )
        if AST_MIX_LABEL in result:
            partner_label, _partner_metadata = transform_label(
                result[AST_MIX_LABEL],
                label_config,
                metadata=None,
            )
            label = lam * tf.cast(label, tf.float32) + (1.0 - lam) * tf.cast(
                partner_label,
                tf.float32,
            )
        result[LABEL] = label
        result[METADATA] = metadata
    return result


def _make_ast_feature_stage(
    *,
    frontend: FrontendConfig,
    layout: str,
    dtype: str,
    output_key: str,
    static_shape: tuple[int | None, ...] | None,
    target_length: int,
    norm_mean: float,
    norm_std: float,
    freqm: int,
    timem: int,
    noise: bool,
    segment_config: SegmentStrategyConfig | None,
    label_config: LabelTransformConfig | None,
    is_training: bool,
):
    def stage(sample: dict, seed: tf.Tensor | int | None = None, num_classes=None) -> dict:
        del num_classes
        seed = _seed_tensor(seed)
        segment_seed, mix_seed, spec_seed, noise_seed = tf.unstack(tf.random.split(seed, 4))
        result = dict(sample)

        if segment_config is not None:
            segmented = segment_waveform(
                result[WAVEFORM],
                result[SAMPLE_RATE],
                segment_config,
                is_training=is_training,
                seed=segment_seed,
            )
            metadata = dict(result.get(METADATA, {}))
            metadata.update(segmented[METADATA])
            result[WAVEFORM] = segmented["audio"]
            result[METADATA] = metadata

        if is_training:
            result = _maybe_mix_waveform_and_label(
                result,
                label_config=label_config,
                seed=mix_seed,
            )

        features = ast_kaldi_fbank(result[WAVEFORM], frontend)
        features = ast_pad_or_crop_fbank(features, target_length)
        if is_training:
            features = ast_specaugment(features, freqm=freqm, timem=timem, seed=spec_seed)
        features = ast_normalize_fbank(features, norm_mean, norm_std)
        if is_training and noise:
            features = ast_add_noise_and_roll(features, seed=noise_seed)

        features = convert_audio_layout(features, layout, output_kind="features")
        features = tf.cast(features, tf.as_dtype(dtype))
        if static_shape is not None:
            features.set_shape(static_shape)

        result[output_key] = features
        if not is_training:
            result = _transform_sample_label(result, label_config)
        return result

    return stage


def ast_pipeline(
    preproc_kwargs: dict | None = None,
    aug_kwargs: dict | None = None,
    laug_kwargs: dict | None = None,
    postproc_kwargs: dict | None = None,
    preprocess: dict | None = None,
    segment: dict | None = None,
    frontend: dict | FrontendConfig | None = None,
    layout: str = "btf",
    dtype: str = "float32",
    output_key: str | None = None,
    static_shape: tuple[int | None, ...] | None = None,
    label_transform: dict | LabelTransformConfig | None = None,
    train_augment: dict | None = None,
    metadata: dict | None = None,
    **kwargs,
):
    del aug_kwargs, laug_kwargs, kwargs
    preproc_kwargs = preproc_kwargs or {}
    postproc_kwargs = postproc_kwargs or {}
    if preprocess is not None:
        preproc_kwargs.setdefault("config", preprocess)

    is_training = postproc_kwargs.get("is_training", False)
    frontend_config = FrontendConfig.from_dict(frontend or ast_frontend())
    segment_config = SegmentStrategyConfig.from_dict(segment) if segment is not None else None
    label_config = (
        LabelTransformConfig.from_dict(label_transform)
        if label_transform is not None
        else None
    )
    ast = _ast_config(metadata, train_augment)
    target_length = _target_length(layout, static_shape, ast)
    output_key = output_key or FEATURES

    feature_stage = _make_ast_feature_stage(
        frontend=frontend_config,
        layout=layout,
        dtype=dtype,
        output_key=output_key,
        static_shape=static_shape,
        target_length=target_length,
        norm_mean=float(ast.get("mean", -4.2677393)),
        norm_std=float(ast.get("std", 4.5689974)),
        freqm=int(ast.get("freqm", 0)),
        timem=int(ast.get("timem", 0)),
        noise=bool(ast.get("noise", False)),
        segment_config=segment_config,
        label_config=label_config,
        is_training=is_training,
    )

    def identity_sample(sample, *args, **_kwargs):
        return sample

    def identity_batch(batch, num_classes=None, seed=None, **_kwargs):
        del num_classes, seed
        return batch

    def train_postprocess(sample, num_classes=None):
        del num_classes
        return _transform_sample_label(dict(sample), label_config)

    return (
        make_preprocessing(**preproc_kwargs),
        feature_stage if is_training else identity_sample,
        identity_batch,
        train_postprocess if is_training else feature_stage,
    )


__all__ = [
    "AST_MIX_LABEL",
    "AST_MIX_LAMBDA",
    "AST_MIX_WAVEFORM",
    "AST_RECIPE_CONFIGS",
    "AST_REFERENCE_COMMIT",
    "ast_add_noise_and_roll",
    "ast_audioset_16k_10s_fbank128",
    "ast_esc50_16k_5s_fbank128",
    "ast_frequency_mask",
    "ast_frontend",
    "ast_mix_waveforms",
    "ast_normalize_fbank",
    "ast_pad_or_crop_fbank",
    "ast_pipeline",
    "ast_specaugment",
    "ast_speechcommands_16k_1s_fbank128",
    "ast_time_mask",
]

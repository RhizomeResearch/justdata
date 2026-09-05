from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import tensorflow as tf

from justdata.acoustic._random import _maybe_apply, _seed_tensor, _uniform_float
from justdata.acoustic._signal import (
    _add_at_snr,
    _convolve_channels,
    _fit_length,
    _normalize_noise,
    _pink_noise,
)
from justdata.acoustic.adapters import infer_duration
from justdata.acoustic.augment._common import _config_data, _normalize_augment_specs
from justdata.acoustic.augment.configs import (
    AdditiveNoiseConfig,
    CodecSimulationConfig,
    DynamicRangeCompressionConfig,
    RIRConvolutionConfig,
    RandomClippingConfig,
    RandomGainConfig,
    RandomTimeShiftConfig,
    SpeedPerturbConfig,
)
from justdata.acoustic.registry import (
    get_audio_waveform_augment,
    register_audio_waveform_augment,
)
from justdata.acoustic.schema import DURATION, SAMPLE_RATE, WAVEFORM


_EPS = tf.constant(1e-8, dtype=tf.float32)


def _as_waveform_tc(waveform: tf.Tensor) -> tuple[tf.Tensor, int | None]:
    waveform = tf.convert_to_tensor(waveform)
    rank = waveform.shape.rank
    if rank == 1:
        return tf.expand_dims(waveform, -1), rank
    if rank != 2:
        raise ValueError("Waveform augment input must have shape [T] or [T, C]")
    return waveform, rank


def _restore_rank(waveform: tf.Tensor, rank: int | None) -> tf.Tensor:
    if rank == 1:
        return tf.squeeze(waveform, axis=-1)
    return waveform


def _config_from(
    config: Any,
    config_cls: type,
    overrides: Mapping[str, Any],
) -> Any:
    return config_cls(**_config_data(config, overrides, config_cls=config_cls))


def _enabled(is_training: bool, augment_eval: bool) -> bool:
    return bool(is_training or augment_eval)


def _uniform_int(seed: tf.Tensor, minval: tf.Tensor, maxval: tf.Tensor) -> tf.Tensor:
    return tf.cond(
        tf.equal(minval, maxval),
        lambda: tf.cast(minval, tf.int32),
        lambda: tf.random.stateless_uniform(
            [],
            seed=seed,
            minval=tf.cast(minval, tf.int32),
            maxval=tf.cast(maxval + 1, tf.int32),
            dtype=tf.int32,
        ),
    )


def _sample_target_length(
    sample_rate: tf.Tensor | int | None,
    target_length: int | tf.Tensor | None,
    crop_duration: float | None,
) -> tf.Tensor:
    if target_length is not None:
        return tf.cast(target_length, tf.int32)
    if crop_duration is None:
        raise ValueError("random_crop requires target_length or crop_duration")
    if sample_rate is None:
        raise ValueError("random_crop with crop_duration requires sample_rate")
    target = tf.round(
        tf.cast(crop_duration, tf.float32) * tf.cast(sample_rate, tf.float32)
    )
    return tf.cast(tf.maximum(target, 1.0), tf.int32)


def make_colored_noise(
    shape: tf.Tensor | Sequence[int],
    seed: tf.Tensor | int | None,
    kind: str = "white",
) -> tf.Tensor:
    """Create unit-RMS white, pink, or brown noise with stateless randomness."""
    seed = _seed_tensor(seed)
    shape = tf.cast(tf.convert_to_tensor(shape), tf.int32)
    noise = tf.random.stateless_normal(shape, seed=seed, dtype=tf.float32)
    if kind == "white":
        return _normalize_noise(noise)

    audio, rank = _as_waveform_tc(noise)
    if kind == "brown":
        brown = tf.cumsum(audio, axis=0)
        return _restore_rank(_normalize_noise(brown), rank)
    if kind != "pink":
        raise ValueError("noise kind must be one of 'white', 'pink', or 'brown'")

    return _restore_rank(_pink_noise(audio), rank)


@register_audio_waveform_augment("random_gain")
def random_gain(
    waveform: tf.Tensor,
    sample_rate: tf.Tensor | int | None = None,
    *,
    seed: tf.Tensor | int | None = None,
    config: RandomGainConfig | Mapping[str, Any] | None = None,
    prob: float | None = None,
    min_db: float | None = None,
    max_db: float | None = None,
    is_training: bool = True,
    augment_eval: bool = False,
) -> tf.Tensor:
    del sample_rate
    if not _enabled(is_training, augment_eval):
        return tf.convert_to_tensor(waveform)

    cfg = _config_from(
        config,
        RandomGainConfig,
        {"prob": prob, "min_db": min_db, "max_db": max_db},
    )
    seed = _seed_tensor(seed)
    gate_seed, gain_seed = tf.unstack(tf.random.split(seed, 2))
    input_dtype = tf.convert_to_tensor(waveform).dtype
    audio = tf.cast(waveform, tf.float32)

    def apply() -> tf.Tensor:
        gain_db = _uniform_float(gain_seed, cfg.min_db, cfg.max_db)
        scale = tf.pow(tf.constant(10.0, dtype=tf.float32), gain_db / 20.0)
        return tf.cast(audio * scale, input_dtype)

    return _maybe_apply(tf.convert_to_tensor(waveform), cfg.prob, gate_seed, apply)


@register_audio_waveform_augment("time_shift")
def time_shift(
    waveform: tf.Tensor,
    sample_rate: tf.Tensor | int | None = None,
    *,
    seed: tf.Tensor | int | None = None,
    config: RandomTimeShiftConfig | Mapping[str, Any] | None = None,
    prob: float | None = None,
    max_shift_seconds: float | None = None,
    mode: str | None = None,
    shift_samples: int | tf.Tensor | None = None,
    is_training: bool = True,
    augment_eval: bool = False,
) -> tf.Tensor:
    if not _enabled(is_training, augment_eval):
        return tf.convert_to_tensor(waveform)
    if sample_rate is None and shift_samples is None:
        raise ValueError("time_shift requires sample_rate unless shift_samples is set")

    cfg = _config_from(
        config,
        RandomTimeShiftConfig,
        {"prob": prob, "max_shift_seconds": max_shift_seconds, "mode": mode},
    )
    seed = _seed_tensor(seed)
    gate_seed, shift_seed = tf.unstack(tf.random.split(seed, 2))
    audio, rank = _as_waveform_tc(waveform)

    def apply() -> tf.Tensor:
        if shift_samples is not None:
            shift = tf.cast(shift_samples, tf.int32)
        else:
            max_shift = tf.cast(
                tf.round(
                    tf.cast(cfg.max_shift_seconds, tf.float32)
                    * tf.cast(sample_rate, tf.float32)
                ),
                tf.int32,
            )
            shift = _uniform_int(shift_seed, -max_shift, max_shift)
        if cfg.mode == "roll":
            shifted = tf.roll(audio, shift=shift, axis=0)
        elif cfg.mode == "zero":
            shifted = _zero_shift(audio, shift)
        else:
            raise ValueError(f"Unknown time shift mode: {cfg.mode!r}")
        return _restore_rank(shifted, rank)

    return _maybe_apply(tf.convert_to_tensor(waveform), cfg.prob, gate_seed, apply)


def _zero_shift(audio: tf.Tensor, shift: tf.Tensor) -> tf.Tensor:
    time = tf.shape(audio)[0]
    channels = tf.shape(audio)[1]
    shift = tf.clip_by_value(shift, -time, time)
    amount = tf.abs(shift)
    zeros = tf.zeros([amount, channels], dtype=audio.dtype)

    def positive() -> tf.Tensor:
        return tf.concat([zeros, audio[: time - amount]], axis=0)

    def negative() -> tf.Tensor:
        return tf.concat([audio[amount:], zeros], axis=0)

    return tf.cond(
        shift > 0,
        positive,
        lambda: tf.cond(shift < 0, negative, lambda: audio),
    )


@register_audio_waveform_augment("random_crop")
def random_crop(
    waveform: tf.Tensor,
    sample_rate: tf.Tensor | int | None = None,
    *,
    seed: tf.Tensor | int | None = None,
    target_length: int | tf.Tensor | None = None,
    crop_duration: float | None = None,
    prob: float = 1.0,
    is_training: bool = True,
    augment_eval: bool = False,
) -> tf.Tensor:
    if not _enabled(is_training, augment_eval):
        return tf.convert_to_tensor(waveform)

    seed = _seed_tensor(seed)
    gate_seed, crop_seed = tf.unstack(tf.random.split(seed, 2))
    audio, rank = _as_waveform_tc(waveform)
    target = _sample_target_length(sample_rate, target_length, crop_duration)

    def apply() -> tf.Tensor:
        time = tf.shape(audio)[0]
        max_start = tf.maximum(time - target, 0)
        start = tf.cond(
            max_start > 0,
            lambda: tf.random.stateless_uniform(
                [],
                seed=crop_seed,
                minval=0,
                maxval=max_start + 1,
                dtype=tf.int32,
            ),
            lambda: tf.constant(0, dtype=tf.int32),
        )
        cropped = _fit_length(audio[start:], target)
        return _restore_rank(cropped, rank)

    return _maybe_apply(tf.convert_to_tensor(waveform), prob, gate_seed, apply)


@register_audio_waveform_augment("additive_noise")
def additive_noise(
    waveform: tf.Tensor,
    sample_rate: tf.Tensor | int | None = None,
    *,
    seed: tf.Tensor | int | None = None,
    config: AdditiveNoiseConfig | Mapping[str, Any] | None = None,
    prob: float | None = None,
    snr_db_min: float | None = None,
    snr_db_max: float | None = None,
    noise_kind: str | None = None,
    is_training: bool = True,
    augment_eval: bool = False,
) -> tf.Tensor:
    del sample_rate
    if not _enabled(is_training, augment_eval):
        return tf.convert_to_tensor(waveform)

    cfg = _config_from(
        config,
        AdditiveNoiseConfig,
        {
            "prob": prob,
            "snr_db_min": snr_db_min,
            "snr_db_max": snr_db_max,
            "noise_kind": noise_kind,
        },
    )
    seed = _seed_tensor(seed)
    gate_seed, snr_seed, noise_seed = tf.unstack(tf.random.split(seed, 3))
    input_dtype = tf.convert_to_tensor(waveform).dtype
    audio = tf.cast(waveform, tf.float32)

    def apply() -> tf.Tensor:
        noise = make_colored_noise(tf.shape(audio), noise_seed, cfg.noise_kind)
        snr_db = _uniform_float(snr_seed, cfg.snr_db_min, cfg.snr_db_max)
        return tf.cast(_add_at_snr(audio, noise, snr_db), input_dtype)

    return _maybe_apply(tf.convert_to_tensor(waveform), cfg.prob, gate_seed, apply)


@register_audio_waveform_augment("colored_noise")
def colored_noise(
    waveform: tf.Tensor,
    sample_rate: tf.Tensor | int | None = None,
    *,
    seed: tf.Tensor | int | None = None,
    config: AdditiveNoiseConfig | Mapping[str, Any] | None = None,
    prob: float | None = None,
    snr_db_min: float | None = None,
    snr_db_max: float | None = None,
    noise_kind: str = "pink",
    is_training: bool = True,
    augment_eval: bool = False,
) -> tf.Tensor:
    return additive_noise(
        waveform,
        sample_rate,
        seed=seed,
        config=config,
        prob=prob,
        snr_db_min=snr_db_min,
        snr_db_max=snr_db_max,
        noise_kind=noise_kind,
        is_training=is_training,
        augment_eval=augment_eval,
    )


@register_audio_waveform_augment("polarity_inversion")
def polarity_inversion(
    waveform: tf.Tensor,
    sample_rate: tf.Tensor | int | None = None,
    *,
    seed: tf.Tensor | int | None = None,
    prob: float = 0.5,
    is_training: bool = True,
    augment_eval: bool = False,
) -> tf.Tensor:
    del sample_rate
    if not _enabled(is_training, augment_eval):
        return tf.convert_to_tensor(waveform)

    seed = _seed_tensor(seed)
    audio = tf.convert_to_tensor(waveform)
    return _maybe_apply(audio, prob, seed, lambda: -audio)


@register_audio_waveform_augment("random_clipping")
def random_clipping(
    waveform: tf.Tensor,
    sample_rate: tf.Tensor | int | None = None,
    *,
    seed: tf.Tensor | int | None = None,
    config: RandomClippingConfig | Mapping[str, Any] | None = None,
    prob: float | None = None,
    min_threshold: float | None = None,
    max_threshold: float | None = None,
    is_training: bool = True,
    augment_eval: bool = False,
) -> tf.Tensor:
    del sample_rate
    if not _enabled(is_training, augment_eval):
        return tf.convert_to_tensor(waveform)

    cfg = _config_from(
        config,
        RandomClippingConfig,
        {
            "prob": prob,
            "min_threshold": min_threshold,
            "max_threshold": max_threshold,
        },
    )
    seed = _seed_tensor(seed)
    gate_seed, threshold_seed = tf.unstack(tf.random.split(seed, 2))
    audio = tf.convert_to_tensor(waveform)

    def apply() -> tf.Tensor:
        threshold = tf.cast(
            _uniform_float(threshold_seed, cfg.min_threshold, cfg.max_threshold),
            audio.dtype,
        )
        return tf.clip_by_value(audio, -threshold, threshold)

    return _maybe_apply(audio, cfg.prob, gate_seed, apply)


@register_audio_waveform_augment("dynamic_range_compression")
def dynamic_range_compression(
    waveform: tf.Tensor,
    sample_rate: tf.Tensor | int | None = None,
    *,
    seed: tf.Tensor | int | None = None,
    config: DynamicRangeCompressionConfig | Mapping[str, Any] | None = None,
    prob: float | None = None,
    threshold_db_min: float | None = None,
    threshold_db_max: float | None = None,
    ratio_min: float | None = None,
    ratio_max: float | None = None,
    eps: float = 1e-8,
    is_training: bool = True,
    augment_eval: bool = False,
) -> tf.Tensor:
    del sample_rate
    if not _enabled(is_training, augment_eval):
        return tf.convert_to_tensor(waveform)

    cfg = _config_from(
        config,
        DynamicRangeCompressionConfig,
        {
            "prob": prob,
            "threshold_db_min": threshold_db_min,
            "threshold_db_max": threshold_db_max,
            "ratio_min": ratio_min,
            "ratio_max": ratio_max,
        },
    )
    seed = _seed_tensor(seed)
    gate_seed, threshold_seed, ratio_seed = tf.unstack(tf.random.split(seed, 3))
    input_dtype = tf.convert_to_tensor(waveform).dtype
    audio = tf.cast(waveform, tf.float32)

    def apply() -> tf.Tensor:
        threshold_db = _uniform_float(
            threshold_seed,
            cfg.threshold_db_min,
            cfg.threshold_db_max,
        )
        ratio = _uniform_float(ratio_seed, cfg.ratio_min, cfg.ratio_max)
        magnitude = tf.abs(audio)
        level_db = (
            20.0 * tf.math.log(magnitude + tf.cast(eps, tf.float32)) / tf.math.log(10.0)
        )
        over = level_db - threshold_db
        compressed_over = over / ratio
        gain_db = tf.where(over > 0.0, compressed_over - over, tf.zeros_like(over))
        gain = tf.pow(tf.constant(10.0, dtype=tf.float32), gain_db / 20.0)
        return tf.cast(tf.sign(audio) * magnitude * gain, input_dtype)

    return _maybe_apply(tf.convert_to_tensor(waveform), cfg.prob, gate_seed, apply)


@register_audio_waveform_augment("rir_convolution")
def rir_convolution(
    waveform: tf.Tensor,
    sample_rate: tf.Tensor | int | None = None,
    *,
    seed: tf.Tensor | int | None = None,
    config: RIRConvolutionConfig | Mapping[str, Any] | None = None,
    prob: float | None = None,
    rir: tf.Tensor | Sequence[float] | None = None,
    rir_dataset: str | None = None,
    normalize_ir: bool | None = None,
    compensate_delay: bool | None = None,
    is_training: bool = True,
    augment_eval: bool = False,
) -> tf.Tensor:
    del sample_rate
    if not _enabled(is_training, augment_eval):
        return tf.convert_to_tensor(waveform)

    cfg = _config_from(
        config,
        RIRConvolutionConfig,
        {
            "prob": prob,
            "rir_dataset": rir_dataset,
            "normalize_ir": normalize_ir,
            "compensate_delay": compensate_delay,
        },
    )
    if cfg.rir_dataset is not None and rir is None:
        raise ValueError(
            "rir_dataset loading is not implemented; pass an explicit rir tensor"
        )

    seed = _seed_tensor(seed)
    gate_seed, rir_seed = tf.unstack(tf.random.split(seed, 2))
    input_dtype = tf.convert_to_tensor(waveform).dtype
    audio, rank = _as_waveform_tc(tf.cast(waveform, tf.float32))
    ir = _prepare_rir(rir, rir_seed, cfg.normalize_ir)

    def apply() -> tf.Tensor:
        convolved = _convolve_channels(audio, ir, cfg.compensate_delay)
        return tf.cast(_restore_rank(convolved, rank), input_dtype)

    return _maybe_apply(tf.convert_to_tensor(waveform), cfg.prob, gate_seed, apply)


def _prepare_rir(
    rir: tf.Tensor | Sequence[float] | None,
    seed: tf.Tensor,
    normalize: bool,
) -> tf.Tensor:
    if rir is None:
        tail = tf.random.stateless_normal(
            [63], seed=seed, stddev=0.05, dtype=tf.float32
        )
        decay = tf.exp(-tf.linspace(0.0, 4.0, 63))
        rir = tf.concat([tf.ones([1], dtype=tf.float32), tail * decay], axis=0)
    ir = tf.reshape(tf.cast(tf.convert_to_tensor(rir), tf.float32), [-1])
    ir = tf.cond(
        tf.shape(ir)[0] > 0, lambda: ir, lambda: tf.ones([1], dtype=tf.float32)
    )
    if normalize:
        energy = tf.sqrt(tf.reduce_sum(tf.square(ir)))
        ir = tf.math.divide_no_nan(ir, tf.maximum(energy, _EPS))
    return ir


@register_audio_waveform_augment("speed_perturb")
def speed_perturb(
    waveform: tf.Tensor,
    sample_rate: tf.Tensor | int | None = None,
    *,
    seed: tf.Tensor | int | None = None,
    config: SpeedPerturbConfig | Mapping[str, Any] | None = None,
    prob: float | None = None,
    rates: Sequence[float] | None = None,
    is_training: bool = True,
    augment_eval: bool = False,
) -> tf.Tensor:
    del sample_rate
    if not _enabled(is_training, augment_eval):
        return tf.convert_to_tensor(waveform)

    cfg = _config_from(
        config,
        SpeedPerturbConfig,
        {"prob": prob, "rates": tuple(rates) if rates is not None else None},
    )
    seed = _seed_tensor(seed)
    gate_seed, rate_seed = tf.unstack(tf.random.split(seed, 2))
    input_dtype = tf.convert_to_tensor(waveform).dtype
    audio, rank = _as_waveform_tc(tf.cast(waveform, tf.float32))

    def apply() -> tf.Tensor:
        rates_tensor = tf.constant(cfg.rates, dtype=tf.float32)
        if len(cfg.rates) == 1:
            rate = rates_tensor[0]
        else:
            index = tf.random.stateless_uniform(
                [],
                seed=rate_seed,
                minval=0,
                maxval=len(cfg.rates),
                dtype=tf.int32,
            )
            rate = rates_tensor[index]
        time = tf.shape(audio)[0]
        new_time = tf.cast(
            tf.maximum(tf.round(tf.cast(time, tf.float32) / rate), 1.0),
            tf.int32,
        )
        resized = _resize_time(audio, new_time)
        return tf.cast(_restore_rank(_fit_length(resized, time), rank), input_dtype)

    return _maybe_apply(tf.convert_to_tensor(waveform), cfg.prob, gate_seed, apply)


def _resize_time(audio: tf.Tensor, new_time: tf.Tensor) -> tf.Tensor:
    resized = tf.image.resize(
        audio[tf.newaxis, :, tf.newaxis, :],
        [new_time, 1],
        method="bilinear",
        antialias=True,
    )
    return resized[0, :, 0, :]


@register_audio_waveform_augment("codec_simulation")
def codec_simulation(
    waveform: tf.Tensor,
    sample_rate: tf.Tensor | int | None = None,
    *,
    seed: tf.Tensor | int | None = None,
    config: CodecSimulationConfig | Mapping[str, Any] | None = None,
    prob: float | None = None,
    codec: str | None = None,
    bitrate: int | None = None,
    is_training: bool = True,
    augment_eval: bool = False,
) -> tf.Tensor:
    del sample_rate
    if not _enabled(is_training, augment_eval):
        return tf.convert_to_tensor(waveform)

    cfg = _config_from(
        config,
        CodecSimulationConfig,
        {"prob": prob, "codec": codec, "bitrate": bitrate},
    )
    gate_seed = _seed_tensor(seed)
    input_dtype = tf.convert_to_tensor(waveform).dtype
    audio, rank = _as_waveform_tc(tf.cast(waveform, tf.float32))

    def apply() -> tf.Tensor:
        clipped = tf.clip_by_value(audio, -1.0, 1.0)
        if cfg.codec == "mulaw":
            coded = _mulaw_roundtrip(clipped)
        elif cfg.codec == "alaw":
            coded = _alaw_roundtrip(clipped)
        elif cfg.codec == "mp3_proxy":
            coded = _mp3_proxy(clipped, cfg.bitrate)
        else:
            raise ValueError(f"Unknown codec simulation: {cfg.codec!r}")
        return tf.cast(
            _restore_rank(tf.clip_by_value(coded, -1.0, 1.0), rank), input_dtype
        )

    return _maybe_apply(tf.convert_to_tensor(waveform), cfg.prob, gate_seed, apply)


def _quantize_unit(x: tf.Tensor, levels: int) -> tf.Tensor:
    levels_f = tf.cast(levels - 1, tf.float32)
    x = tf.clip_by_value(x, -1.0, 1.0)
    return tf.round((x + 1.0) * 0.5 * levels_f) / levels_f * 2.0 - 1.0


def _mulaw_roundtrip(audio: tf.Tensor) -> tf.Tensor:
    mu = tf.constant(255.0, dtype=tf.float32)
    log_mu = tf.math.log1p(mu)
    encoded = tf.sign(audio) * tf.math.log1p(mu * tf.abs(audio)) / log_mu
    encoded = _quantize_unit(encoded, 256)
    return tf.sign(encoded) * tf.math.expm1(tf.abs(encoded) * log_mu) / mu


def _alaw_roundtrip(audio: tf.Tensor) -> tf.Tensor:
    a = tf.constant(87.6, dtype=tf.float32)
    denom = 1.0 + tf.math.log(a)
    abs_audio = tf.abs(audio)
    encoded_mag = tf.where(
        abs_audio < (1.0 / a),
        a * abs_audio / denom,
        (1.0 + tf.math.log(a * tf.maximum(abs_audio, _EPS))) / denom,
    )
    encoded = _quantize_unit(tf.sign(audio) * encoded_mag, 256)
    abs_encoded = tf.abs(encoded)
    decoded_mag = tf.where(
        abs_encoded < (1.0 / denom),
        abs_encoded * denom / a,
        tf.exp(abs_encoded * denom - 1.0) / a,
    )
    return tf.sign(encoded) * decoded_mag


def _mp3_proxy(audio: tf.Tensor, bitrate: int | None) -> tf.Tensor:
    del bitrate
    kernel = tf.constant([0.2, 0.6, 0.2], dtype=tf.float32)[:, tf.newaxis, tf.newaxis]

    def smooth_channel(channel: tf.Tensor) -> tf.Tensor:
        signal = channel[tf.newaxis, :, tf.newaxis]
        return tf.nn.conv1d(signal, kernel, stride=1, padding="SAME")[0, :, 0]

    smoothed = tf.map_fn(
        smooth_channel,
        tf.transpose(audio, [1, 0]),
        fn_output_signature=tf.float32,
    )
    smoothed = tf.transpose(smoothed, [1, 0])
    return _quantize_unit(smoothed, 128)


def normalize_waveform_augment_specs(
    augmentations: Mapping[str, Any] | Sequence[Any] | str | None,
) -> list[dict[str, Any]]:
    return _normalize_augment_specs(augmentations)


def apply_waveform_augmentations(
    sample: dict,
    augmentations: Mapping[str, Any] | Sequence[Any] | str | None,
    *,
    seed: tf.Tensor | int | None = None,
    is_training: bool = True,
    augment_eval: bool = False,
    audio_key: str = WAVEFORM,
) -> dict:
    specs = normalize_waveform_augment_specs(augmentations)
    if not specs or not _enabled(is_training, augment_eval):
        return sample

    seed = _seed_tensor(seed)
    seeds = tf.random.split(seed, len(specs))
    waveform = sample[audio_key]
    sample_rate = sample.get(SAMPLE_RATE)

    for spec, spec_seed in zip(specs, tf.unstack(seeds)):
        kwargs = dict(spec)
        name = kwargs.pop("name")
        fn = get_audio_waveform_augment(name)
        waveform = fn(
            waveform,
            sample_rate,
            seed=spec_seed,
            is_training=is_training,
            augment_eval=augment_eval,
            **kwargs,
        )

    result = dict(sample)
    result[audio_key] = waveform
    if DURATION in result and sample_rate is not None:
        result[DURATION] = infer_duration(
            tf.shape(tf.convert_to_tensor(waveform))[0], sample_rate
        )
    return result


def make_waveform_augmentation_stage(
    augmentations: Mapping[str, Any] | Sequence[Any] | str | None,
    *,
    is_training: bool = True,
    augment_eval: bool = False,
    audio_key: str = WAVEFORM,
):
    specs = normalize_waveform_augment_specs(augmentations)

    def stage(sample: dict, seed: tf.Tensor | int | None = None) -> dict:
        return apply_waveform_augmentations(
            sample,
            specs,
            seed=seed,
            is_training=is_training,
            augment_eval=augment_eval,
            audio_key=audio_key,
        )

    return stage


__all__ = [
    "additive_noise",
    "apply_waveform_augmentations",
    "codec_simulation",
    "colored_noise",
    "dynamic_range_compression",
    "make_colored_noise",
    "make_waveform_augmentation_stage",
    "normalize_waveform_augment_specs",
    "polarity_inversion",
    "random_clipping",
    "random_crop",
    "random_gain",
    "rir_convolution",
    "speed_perturb",
    "time_shift",
]

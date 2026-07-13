from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import tensorflow as tf

from justdata.acoustic.configs import AudioPreset, FrontendConfig
from justdata.acoustic.frontends.stft import compute_num_frames
from justdata.acoustic.layouts import (
    AudioLayout,
    OutputKind,
    convert_audio_layout,
    unbatched_shape_for_layout,
)
from justdata.acoustic.registry import get_audio_frontend
from justdata.acoustic.schema import FEATURES, WAVEFORM


def frontend_output_kind(frontend: FrontendConfig | dict) -> OutputKind:
    config = FrontendConfig.from_dict(frontend)
    return "waveform" if config.name == "raw_waveform" else "features"


def default_output_key(frontend: FrontendConfig | dict) -> str:
    return WAVEFORM if frontend_output_kind(frontend) == "waveform" else FEATURES


def _get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _frontend_from_preset(preset: AudioPreset | Mapping[str, Any]) -> FrontendConfig:
    return FrontendConfig.from_dict(_get(preset, "frontend"))


def _layout_from_preset(preset: AudioPreset | Mapping[str, Any]) -> AudioLayout:
    return _get(preset, "layout")


def _output_key_from_preset(preset: AudioPreset | Mapping[str, Any]) -> str:
    frontend = _frontend_from_preset(preset)
    return _get(preset, "output_key") or default_output_key(frontend)


def _duration_from_preset(preset: AudioPreset | Mapping[str, Any]) -> float | None:
    duration = _get(preset, "input_duration")
    if duration is not None:
        return float(duration)
    segment = _get(preset, "segment")
    if segment is not None:
        return float(_get(segment, "clip_duration"))
    return None


def _sample_rate_from_preset(
    preset: AudioPreset | Mapping[str, Any],
    frontend: FrontendConfig,
) -> int | None:
    sample_rate = _get(preset, "target_sample_rate")
    if sample_rate is not None:
        return int(sample_rate)
    if frontend.stft is not None:
        return frontend.stft.sample_rate
    preprocess = _get(preset, "preprocess")
    if preprocess is not None:
        return int(_get(preprocess, "target_sample_rate"))
    return None


def _channel_count_from_preset(preset: AudioPreset | Mapping[str, Any]) -> int | None:
    preprocess = _get(preset, "preprocess")
    if preprocess is None:
        return 1
    return None if _get(preprocess, "channel_strategy") == "keep" else 1


def _frequency_bins(frontend: FrontendConfig) -> int | None:
    if frontend.name == "raw_waveform":
        return None
    if frontend.name == "stft_magnitude":
        if frontend.stft is None:
            return None
        return (
            frontend.stft.n_fft // 2 + 1
            if frontend.stft.onesided
            else frontend.stft.n_fft
        )
    if frontend.name == "mfcc":
        return frontend.n_mfcc
    if frontend.mel is not None:
        return frontend.mel.n_mels
    return None


def _fixed_eval_view_count(
    preset: AudioPreset | Mapping[str, Any],
) -> int | None:
    segment = _get(preset, "segment")
    if segment is None:
        return None
    duration_policy = _get(segment, "duration_policy", "none")
    if duration_policy == "sliding_windows":
        raise ValueError(
            "sliding segmentation has a data-dependent number of views and "
            "cannot be assigned a fixed shape or batched"
        )
    if duration_policy != "none":
        return None
    eval_mode = _get(segment, "eval_mode")
    if eval_mode == "sliding":
        raise ValueError(
            "sliding segmentation has a data-dependent number of views and "
            "cannot be assigned a fixed shape or batched"
        )
    if eval_mode == "multi_crop":
        return int(_get(segment, "num_views"))
    return None


def expected_audio_static_shape(
    preset: AudioPreset | Mapping[str, Any],
) -> tuple[int | None, ...] | None:
    view_count = _fixed_eval_view_count(preset)
    explicit_shape = _get(preset, "static_shape")
    if explicit_shape is not None:
        shape = tuple(explicit_shape)
        return (view_count,) + shape if view_count is not None else shape

    frontend = _frontend_from_preset(preset)
    layout = _layout_from_preset(preset)
    output_kind = frontend_output_kind(frontend)
    duration = _duration_from_preset(preset)
    sample_rate = _sample_rate_from_preset(preset, frontend)
    channels = _channel_count_from_preset(preset)

    if duration is None or sample_rate is None:
        return None

    num_samples = round(duration * sample_rate)
    if output_kind == "waveform":
        time = num_samples
        frequency = None
    else:
        if frontend.stft is None:
            return None
        time = compute_num_frames(
            num_samples,
            n_fft=frontend.stft.n_fft,
            win_length=frontend.stft.win_length,
            hop_length=frontend.stft.hop_length,
            center=frontend.stft.center,
        )
        frequency = _frequency_bins(frontend)

    shape = unbatched_shape_for_layout(
        layout,
        output_kind=output_kind,
        time=time,
        frequency=frequency,
        channels=channels,
    )
    return (view_count,) + shape if view_count is not None else shape


def set_static_audio_shape(
    sample: dict,
    preset: AudioPreset | Mapping[str, Any],
) -> dict:
    output_key = _output_key_from_preset(preset)
    if output_key not in sample:
        return sample

    expected_shape = expected_audio_static_shape(preset)
    if expected_shape is None:
        return sample

    result = dict(sample)
    tensor = tf.convert_to_tensor(result[output_key])
    tensor.set_shape(expected_shape)
    result[output_key] = tensor
    return result


def cast_audio_dtype(x: tf.Tensor, dtype: str) -> tf.Tensor:
    return tf.cast(x, tf.as_dtype(dtype))


def _pad_or_crop_axis(x: tf.Tensor, target: tf.Tensor | int, axis: int) -> tf.Tensor:
    x = tf.convert_to_tensor(x)
    target = tf.cast(target, tf.int32)
    rank = x.shape.rank
    if rank is None:
        raise ValueError("Padding requires a tensor with known rank")
    if axis < 0:
        axis += rank

    current = tf.shape(x)[axis]
    begin = tf.zeros([rank], dtype=tf.int32)
    size = tf.shape(x)
    size = tf.tensor_scatter_nd_update(size, [[axis]], [tf.minimum(current, target)])
    cropped = tf.slice(x, begin, size)

    pad_after = tf.maximum(target - current, 0)
    paddings = tf.zeros([rank, 2], dtype=tf.int32)
    paddings = tf.tensor_scatter_nd_update(paddings, [[axis, 1]], [pad_after])
    return tf.pad(cropped, paddings)


def pad_or_crop_time(
    x: tf.Tensor, target_frames: int | tf.Tensor, *, time_axis: int = 0
) -> tf.Tensor:
    return _pad_or_crop_axis(x, target_frames, time_axis)


def pad_time_to_multiple(
    x: tf.Tensor, multiple: int, *, time_axis: int = 0
) -> tf.Tensor:
    if multiple <= 0:
        raise ValueError("multiple must be positive")
    current = tf.shape(x)[time_axis]
    target = tf.cast(
        tf.math.ceil(tf.cast(current, tf.float32) / float(multiple)) * float(multiple),
        tf.int32,
    )
    return pad_or_crop_time(x, target, time_axis=time_axis)


def pad_to_patch_multiple(
    x: tf.Tensor,
    *,
    time_multiple: int = 1,
    frequency_multiple: int = 1,
    time_axis: int = 0,
    frequency_axis: int = 1,
) -> tf.Tensor:
    if time_multiple <= 0 or frequency_multiple <= 0:
        raise ValueError("patch multiples must be positive")
    x = pad_time_to_multiple(x, time_multiple, time_axis=time_axis)
    return pad_time_to_multiple(x, frequency_multiple, time_axis=frequency_axis)


def make_model_input_stage(
    frontend: FrontendConfig | dict,
    *,
    layout: AudioLayout,
    dtype: str = "float32",
    output_key: str | None = None,
    preset: AudioPreset | Mapping[str, Any] | None = None,
) -> Any:
    import justdata.acoustic.frontends  # noqa: F401

    frontend_config = FrontendConfig.from_dict(frontend)
    frontend_fn = get_audio_frontend(frontend_config.name)
    output_kind = frontend_output_kind(frontend_config)
    resolved_output_key = output_key or default_output_key(frontend_config)

    if preset is not None:
        _fixed_eval_view_count(preset)

    def process_waveform(waveform: tf.Tensor) -> tf.Tensor:
        features = frontend_fn(waveform, frontend_config)
        features = convert_audio_layout(features, layout, output_kind=output_kind)
        return cast_audio_dtype(features, dtype)

    def stage(sample: dict, num_classes: int | None = None) -> dict:
        waveform = tf.convert_to_tensor(sample[WAVEFORM])
        if waveform.shape.rank == 2:
            features = process_waveform(waveform)
        elif waveform.shape.rank == 3:
            features = tf.map_fn(
                process_waveform,
                waveform,
                fn_output_signature=tf.TensorSpec(shape=None, dtype=tf.as_dtype(dtype)),
            )
        else:
            raise ValueError(
                "Model input stage expects waveform shape [T, C] or [V, T, C]"
            )

        result = dict(sample)
        result[resolved_output_key] = features
        if preset is not None:
            result = set_static_audio_shape(result, preset)
        return result

    return stage


def preset_info(
    *,
    frontend: FrontendConfig | dict,
    layout: AudioLayout,
    dtype: str,
    output_key: str | None = None,
    static_shape: tuple[int | None, ...] | None = None,
    input_duration: float | None = None,
    target_sample_rate: int | None = None,
    preprocess: Any = None,
    segment: Any = None,
) -> dict[str, Any]:
    return {
        "frontend": FrontendConfig.from_dict(frontend).to_dict(),
        "layout": layout,
        "dtype": dtype,
        "output_key": output_key,
        "static_shape": static_shape,
        "input_duration": input_duration,
        "target_sample_rate": target_sample_rate,
        "preprocess": preprocess,
        "segment": segment,
    }


__all__ = [
    "cast_audio_dtype",
    "default_output_key",
    "expected_audio_static_shape",
    "frontend_output_kind",
    "make_model_input_stage",
    "pad_or_crop_time",
    "pad_time_to_multiple",
    "pad_to_patch_multiple",
    "preset_info",
    "set_static_audio_shape",
]

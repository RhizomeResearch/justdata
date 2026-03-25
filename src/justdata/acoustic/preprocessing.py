from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import tensorflow as tf

from justdata.acoustic.adapters import (
    infer_duration,
    standardize_waveform_layout,
    to_float32_waveform,
)
from justdata.acoustic.channel import apply_channel_strategy
from justdata.acoustic.configs import AudioPreprocessConfig
from justdata.acoustic.decoding import decode_audio_file
from justdata.acoustic.resampling import resample_waveform
from justdata.acoustic.schema import (
    AUDIO,
    DURATION,
    METADATA,
    ORIGINAL_SAMPLE_RATE,
    PATH,
    SAMPLE_RATE,
    WAVEFORM,
)


def remove_dc_offset(audio: tf.Tensor) -> tf.Tensor:
    audio = tf.convert_to_tensor(audio, dtype=tf.float32)
    return audio - tf.reduce_mean(audio, axis=0, keepdims=True)


def normalize_waveform(audio: tf.Tensor, mode: str) -> tf.Tensor:
    audio = tf.convert_to_tensor(audio, dtype=tf.float32)
    eps = tf.constant(1e-8, dtype=tf.float32)

    if mode == "none":
        return audio
    if mode == "peak":
        peak = tf.reduce_max(tf.abs(audio))
        return tf.math.divide_no_nan(audio, tf.maximum(peak, eps))
    if mode == "rms":
        rms = tf.sqrt(tf.reduce_mean(tf.square(audio)))
        return tf.math.divide_no_nan(audio, tf.maximum(rms, eps))
    if mode == "lufs":
        raise ValueError(
            "normalize_waveform='lufs' requires an explicit loudness target; "
            "use 'peak' or 'rms' for built-in deterministic normalization."
        )
    raise ValueError(f"Unknown waveform normalization mode: {mode!r}")


def _metadata_from(sample: dict) -> dict:
    metadata = sample.get(METADATA)
    return dict(metadata) if isinstance(metadata, dict) else {}


def _get_from_audio_payload(audio_payload: Any, key: str) -> Any | None:
    if isinstance(audio_payload, Mapping):
        return audio_payload.get(key)
    return None


def _extract_audio(sample: dict) -> tuple[tf.Tensor, tf.Tensor | None]:
    if WAVEFORM in sample:
        return tf.convert_to_tensor(sample[WAVEFORM]), None

    audio_payload = sample.get(AUDIO)
    if isinstance(audio_payload, Mapping):
        waveform = audio_payload.get("array", audio_payload.get(WAVEFORM))
        sample_rate = audio_payload.get(SAMPLE_RATE, audio_payload.get("sampling_rate"))
        if waveform is not None:
            return tf.convert_to_tensor(waveform), sample_rate
    if audio_payload is not None:
        return tf.convert_to_tensor(audio_payload), None

    if PATH in sample:
        return decode_audio_file(sample[PATH])

    raise ValueError(
        f"Acoustic sample must contain '{WAVEFORM}', '{AUDIO}', or '{PATH}'."
    )


def _extract_sample_rate(
    sample: dict,
    audio_payload: Any,
    decoded_sample_rate: tf.Tensor | None,
) -> tf.Tensor:
    sample_rate = sample.get(SAMPLE_RATE)
    if sample_rate is None:
        sample_rate = sample.get("sampling_rate")
    if sample_rate is None:
        sample_rate = _get_from_audio_payload(audio_payload, SAMPLE_RATE)
    if sample_rate is None:
        sample_rate = _get_from_audio_payload(audio_payload, "sampling_rate")
    if sample_rate is None:
        sample_rate = decoded_sample_rate
    if sample_rate is None:
        raise ValueError("Acoustic sample is missing a sample rate.")
    return tf.cast(sample_rate, tf.int32)


def _extract_original_sample_rate(sample: dict, sample_rate: tf.Tensor) -> tf.Tensor:
    metadata = sample.get(METADATA)
    original_sample_rate = sample.get(ORIGINAL_SAMPLE_RATE)
    if original_sample_rate is None and isinstance(metadata, dict):
        original_sample_rate = metadata.get(ORIGINAL_SAMPLE_RATE)
    if original_sample_rate is None:
        original_sample_rate = sample_rate
    return tf.cast(original_sample_rate, tf.int32)


def make_preprocessing(config: AudioPreprocessConfig) -> Callable[[dict], dict]:
    config = AudioPreprocessConfig.from_dict(config)

    def preprocessing(sample: dict) -> dict:
        waveform, decoded_sample_rate = _extract_audio(sample)
        audio_payload = sample.get(AUDIO)
        sample_rate = _extract_sample_rate(sample, audio_payload, decoded_sample_rate)
        original_sample_rate = _extract_original_sample_rate(sample, sample_rate)

        input_dtype = waveform.dtype
        audio = to_float32_waveform(waveform, input_dtype=input_dtype)
        audio = standardize_waveform_layout(audio)

        original_duration = infer_duration(tf.shape(audio)[0], sample_rate)

        audio = resample_waveform(
            audio,
            sample_rate,
            config.target_sample_rate,
            method=config.resampler,
        )
        sample_rate = tf.cast(config.target_sample_rate, tf.int32)

        audio = apply_channel_strategy(audio, config.channel_strategy)

        if config.remove_dc_offset:
            audio = remove_dc_offset(audio)

        audio = normalize_waveform(audio, config.normalize_waveform)

        if config.clip_value is not None:
            clip_value = tf.cast(config.clip_value, tf.float32)
            audio = tf.clip_by_value(audio, -clip_value, clip_value)

        duration = infer_duration(tf.shape(audio)[0], sample_rate)

        metadata = _metadata_from(sample)
        metadata[ORIGINAL_SAMPLE_RATE] = original_sample_rate
        metadata["original_duration"] = original_duration
        metadata["num_channels"] = tf.shape(audio)[-1]
        metadata["channel_strategy"] = tf.constant(config.channel_strategy)
        metadata[DURATION] = duration

        result = {
            key: value
            for key, value in sample.items()
            if key not in {AUDIO, WAVEFORM, SAMPLE_RATE, DURATION, METADATA}
        }
        result[WAVEFORM] = audio
        result[SAMPLE_RATE] = sample_rate
        result[DURATION] = duration
        result[METADATA] = metadata
        return result

    return preprocessing


__all__ = [
    "make_preprocessing",
    "normalize_waveform",
    "remove_dc_offset",
]

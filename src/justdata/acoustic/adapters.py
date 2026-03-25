from __future__ import annotations

from typing import Any, Literal, Protocol

import tensorflow as tf

from justdata.acoustic.decoding import decode_audio_file
from justdata.acoustic.schema import (
    AUDIO,
    CLIP_ID,
    DATASET,
    DURATION,
    END_TIME,
    EXAMPLE_ID,
    FILENAME,
    LABEL,
    METADATA,
    ORIGINAL_SAMPLE_RATE,
    PATH,
    SAMPLE_RATE,
    SOURCE_ID,
    SPLIT,
    START_TIME,
    WAVEFORM,
)
from justdata.core.adapters import register_adapter


class AcousticAdapter(Protocol):
    def __call__(self, sample: dict) -> dict: ...


def ensure_audio_rank(audio: tf.Tensor) -> tf.Tensor:
    audio = tf.convert_to_tensor(audio)
    rank = audio.shape.rank
    if rank is not None and rank not in (1, 2):
        raise ValueError(f"Audio tensor must have rank 1 or 2; got rank {rank}.")
    return audio


def to_float32_waveform(
    audio: tf.Tensor,
    input_dtype: tf.DType | None = None,
    *,
    clip_value: float | None = None,
) -> tf.Tensor:
    dtype = tf.as_dtype(input_dtype or audio.dtype)
    audio = tf.cast(audio, tf.float32)

    if dtype == tf.int16:
        audio = audio / 32768.0
    elif dtype == tf.int32:
        audio = audio / 2147483648.0
    elif dtype == tf.uint8:
        audio = (audio - 128.0) / 128.0

    if clip_value is not None:
        audio = tf.clip_by_value(audio, -float(clip_value), float(clip_value))

    return audio


def standardize_waveform_layout(
    audio: tf.Tensor,
    layout_hint: Literal["tc", "ct", "t", "auto"] = "auto",
) -> tf.Tensor:
    audio = ensure_audio_rank(audio)

    def _rank2(value: tf.Tensor) -> tf.Tensor:
        if layout_hint == "ct":
            return tf.transpose(value, [1, 0])
        if layout_hint in {"tc", "t"}:
            return value
        if layout_hint != "auto":
            raise ValueError(f"Unknown audio layout_hint: {layout_hint!r}")

        shape = value.shape
        if shape.rank == 2 and shape[0] is not None and shape[-1] is not None:
            if shape[0] <= 8 and shape[-1] > 8:
                return tf.transpose(value, [1, 0])
            return value

        is_channels_first = tf.logical_and(
            tf.shape(value)[0] <= 8,
            tf.shape(value)[-1] > 8,
        )
        return tf.cond(is_channels_first, lambda: tf.transpose(value, [1, 0]), lambda: value)

    rank = audio.shape.rank
    if rank == 1:
        return audio[:, tf.newaxis]
    if rank == 2:
        return _rank2(audio)

    return tf.cond(
        tf.equal(tf.rank(audio), 1),
        lambda: audio[:, tf.newaxis],
        lambda: _rank2(audio),
    )


def infer_duration(num_samples: tf.Tensor, sample_rate: tf.Tensor) -> tf.Tensor:
    return tf.math.divide_no_nan(
        tf.cast(num_samples, tf.float32),
        tf.cast(sample_rate, tf.float32),
    )


def make_audio_metadata(
    *,
    dataset: Any | None = None,
    split: Any | None = None,
    example_id: Any | None = None,
    filename: Any | None = None,
    path: Any | None = None,
    clip_id: Any | None = None,
    source_id: Any | None = None,
    start_time: Any | None = None,
    end_time: Any | None = None,
    original_sample_rate: Any | None = None,
    duration: Any | None = None,
    **extra: Any,
) -> dict:
    fields = {
        DATASET: dataset,
        SPLIT: split,
        EXAMPLE_ID: example_id,
        FILENAME: filename,
        PATH: path,
        CLIP_ID: clip_id,
        SOURCE_ID: source_id,
        START_TIME: start_time,
        END_TIME: end_time,
        ORIGINAL_SAMPLE_RATE: original_sample_rate,
        DURATION: duration,
        **extra,
    }
    return {key: value for key, value in fields.items() if value is not None}


def _get(sample: dict, *keys: str) -> Any | None:
    for key in keys:
        if key in sample:
            return sample[key]
    return None


def _basename(path: Any | None) -> Any | None:
    if path is None:
        return None
    path = tf.convert_to_tensor(path, dtype=tf.string)
    return tf.strings.regex_replace(path, r"^.*[\\/]", "")


def _audio_payload(sample: dict, audio_key: str) -> tuple[Any | None, Any | None]:
    if WAVEFORM in sample:
        return sample[WAVEFORM], None

    audio = sample.get(audio_key)
    if isinstance(audio, dict):
        sample_rate = _get(audio, SAMPLE_RATE, "sampling_rate")
        return _get(audio, "array", WAVEFORM), sample_rate

    return audio, None


def _decode_or_get_waveform(
    sample: dict,
    *,
    audio_key: str,
    path_key: str,
) -> tuple[tf.Tensor, tf.Tensor | None]:
    waveform, sample_rate = _audio_payload(sample, audio_key)
    if waveform is not None:
        return tf.convert_to_tensor(waveform), sample_rate

    path = sample.get(path_key)
    if path is None:
        raise ValueError(
            f"Acoustic sample must contain '{WAVEFORM}', '{audio_key}', or '{path_key}'."
        )

    waveform, decoded_sample_rate = decode_audio_file(path)
    return waveform, decoded_sample_rate


def _slice_window(sample: dict, waveform: tf.Tensor, sample_rate: tf.Tensor) -> tf.Tensor:
    if START_TIME not in sample or END_TIME not in sample:
        return waveform

    start_time = tf.cast(sample[START_TIME], tf.float32)
    end_time = tf.cast(sample[END_TIME], tf.float32)

    def _slice() -> tf.Tensor:
        start = tf.cast(tf.round(start_time * tf.cast(sample_rate, tf.float32)), tf.int32)
        end = tf.cast(tf.round(end_time * tf.cast(sample_rate, tf.float32)), tf.int32)
        start = tf.clip_by_value(start, 0, tf.shape(waveform)[0])
        end = tf.clip_by_value(end, start, tf.shape(waveform)[0])
        return waveform[start:end]

    return tf.cond(tf.greater(end_time, start_time), _slice, lambda: waveform)


def adapt_acoustic_sample(
    sample: dict,
    *,
    dataset: str | None = None,
    audio_key: str = AUDIO,
    label_key: str | None = LABEL,
    sample_rate_key: str = SAMPLE_RATE,
    path_key: str = PATH,
    layout_hint: Literal["tc", "ct", "t", "auto"] = "auto",
    clip_value: float | None = None,
) -> dict:
    waveform, decoded_sample_rate = _decode_or_get_waveform(
        sample,
        audio_key=audio_key,
        path_key=path_key,
    )
    input_dtype = waveform.dtype

    sample_rate = _get(sample, sample_rate_key, "sampling_rate")
    if sample_rate is None:
        sample_rate = decoded_sample_rate
    if sample_rate is None:
        raise ValueError("Acoustic sample is missing a sample rate.")

    sample_rate = tf.cast(sample_rate, tf.int32)
    original_sample_rate = _get(sample, ORIGINAL_SAMPLE_RATE)
    if original_sample_rate is None:
        original_sample_rate = sample_rate
    original_sample_rate = tf.cast(original_sample_rate, tf.int32)

    waveform = to_float32_waveform(waveform, input_dtype=input_dtype, clip_value=clip_value)
    waveform = standardize_waveform_layout(waveform, layout_hint=layout_hint)
    waveform = _slice_window(sample, waveform, sample_rate)

    duration = infer_duration(tf.shape(waveform)[0], sample_rate)

    result = {
        WAVEFORM: waveform,
        SAMPLE_RATE: sample_rate,
        DURATION: duration,
    }
    if label_key is not None and label_key in sample:
        result[LABEL] = sample[label_key]

    metadata = {}
    if isinstance(sample.get(METADATA), dict):
        metadata.update(sample[METADATA])

    path = sample.get(path_key)
    clip_id = _get(sample, CLIP_ID)
    example_id = _get(sample, EXAMPLE_ID)
    if example_id is None:
        example_id = clip_id
    filename = _get(sample, FILENAME)
    if filename is None:
        filename = _basename(path)
    metadata.update(
        make_audio_metadata(
            dataset=_get(sample, DATASET) if DATASET in sample else dataset,
            split=_get(sample, SPLIT),
            example_id=example_id,
            filename=filename,
            path=path,
            clip_id=clip_id,
            source_id=_get(sample, SOURCE_ID),
            start_time=_get(sample, START_TIME),
            end_time=_get(sample, END_TIME),
            original_sample_rate=original_sample_rate,
            duration=duration,
        )
    )
    result[METADATA] = metadata
    return result


@register_adapter("hf_audio:")
@register_adapter("local_audio:")
def acoustic_source_adapter(sample: dict) -> dict:
    return adapt_acoustic_sample(sample)


@register_adapter("esc50")
def esc50_adapter(sample: dict) -> dict:
    return adapt_acoustic_sample(sample, dataset="esc50")


@register_adapter("speech_commands")
def speech_commands_adapter(sample: dict) -> dict:
    return adapt_acoustic_sample(sample, dataset="speech_commands")


@register_adapter("audioset")
def audioset_adapter(sample: dict) -> dict:
    return adapt_acoustic_sample(sample, dataset="audioset")


def dcase2025_task1_adapter(sample: dict) -> dict:
    from justdata.acoustic.dcase2025 import dcase2025_task1_adapter as adapter

    return adapter(sample)


__all__ = [
    "AcousticAdapter",
    "adapt_acoustic_sample",
    "audioset_adapter",
    "dcase2025_task1_adapter",
    "ensure_audio_rank",
    "esc50_adapter",
    "infer_duration",
    "make_audio_metadata",
    "speech_commands_adapter",
    "standardize_waveform_layout",
    "to_float32_waveform",
]

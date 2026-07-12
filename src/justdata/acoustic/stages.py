from __future__ import annotations

from collections.abc import Callable

import tensorflow as tf

from justdata.acoustic.configs import AudioPreprocessConfig, SegmentStrategyConfig
from justdata.acoustic.preprocessing import make_preprocessing
from justdata.acoustic.segment import segment_waveform
from justdata.acoustic.schema import DURATION, METADATA, SAMPLE_RATE, WAVEFORM


def make_segment_stage(
    config: SegmentStrategyConfig,
    *,
    is_training: bool,
    audio_key: str = WAVEFORM,
) -> Callable[[dict, tf.Tensor | int | None], dict]:
    config = SegmentStrategyConfig.from_dict(config)

    def stage(sample: dict, seed: tf.Tensor | int | None = None) -> dict:
        segmented = segment_waveform(
            sample[audio_key],
            sample[SAMPLE_RATE],
            config,
            is_training=is_training,
            seed=seed,
        )
        metadata = dict(sample.get(METADATA, {}))
        if "original_duration" not in metadata and DURATION in sample:
            metadata["original_duration"] = sample[DURATION]
        metadata.update(segmented[METADATA])

        audio = segmented["audio"]
        time_axis = 1 if audio.shape.rank == 3 else 0
        duration = tf.cast(tf.shape(audio)[time_axis], tf.float32) / tf.cast(
            sample[SAMPLE_RATE], tf.float32
        )
        metadata[DURATION] = duration

        result = dict(sample)
        result[audio_key] = audio
        result[DURATION] = duration
        result[METADATA] = metadata
        return result

    return stage


def make_preprocessing_stage(
    config: AudioPreprocessConfig,
) -> Callable[[dict], dict]:
    return make_preprocessing(config)


__all__ = [
    "make_preprocessing_stage",
    "make_segment_stage",
]

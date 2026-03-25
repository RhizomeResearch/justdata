from __future__ import annotations

from collections.abc import Callable

import tensorflow as tf

from justdata.acoustic.configs import SegmentStrategyConfig
from justdata.acoustic.registry import register_audio_eval_view_strategy
from justdata.acoustic.segment import segment_waveform
from justdata.acoustic.schema import METADATA, SAMPLE_RATE, WAVEFORM


@register_audio_eval_view_strategy("center_crop")
@register_audio_eval_view_strategy("full")
@register_audio_eval_view_strategy("sliding")
@register_audio_eval_view_strategy("multi_crop")
def generate_eval_views(
    audio: tf.Tensor,
    sample_rate: tf.Tensor | int,
    config: SegmentStrategyConfig,
) -> dict:
    return segment_waveform(
        audio,
        sample_rate,
        config,
        is_training=False,
        seed=None,
    )


def make_eval_views(
    config: SegmentStrategyConfig,
    *,
    audio_key: str = WAVEFORM,
) -> Callable[[dict], dict]:
    config = SegmentStrategyConfig.from_dict(config)

    def eval_views(sample: dict) -> dict:
        views = generate_eval_views(sample[audio_key], sample[SAMPLE_RATE], config)
        metadata = dict(sample.get(METADATA, {}))
        metadata.update(views[METADATA])

        result = dict(sample)
        result[audio_key] = views["audio"]
        result[METADATA] = metadata
        return result

    return eval_views


__all__ = [
    "generate_eval_views",
    "make_eval_views",
]

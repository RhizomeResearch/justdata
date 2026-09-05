from __future__ import annotations

import tensorflow as tf

from justdata.acoustic._random import _seed_tensor
from justdata.acoustic.configs import SegmentStrategyConfig
from justdata.acoustic.registry import register_audio_segment_strategy
from justdata.acoustic.schema import AUDIO, METADATA


def _target_samples(
    sample_rate: tf.Tensor | int, config: SegmentStrategyConfig
) -> tf.Tensor:
    target = tf.round(
        tf.cast(config.clip_duration, tf.float32) * tf.cast(sample_rate, tf.float32)
    )
    return tf.maximum(tf.cast(target, tf.int32), 1)


def _padding_counts(
    audio: tf.Tensor,
    target_samples: tf.Tensor,
    pad_position: str,
    seed: tf.Tensor | int | None,
) -> tuple[tf.Tensor, tf.Tensor]:
    deficit = tf.maximum(target_samples - tf.shape(audio)[0], 0)
    if pad_position == "right":
        before = tf.constant(0, dtype=tf.int32)
    elif pad_position == "center":
        before = deficit // 2
    elif pad_position == "random":
        before = tf.random.stateless_uniform(
            shape=[],
            seed=_seed_tensor(seed),
            minval=0,
            maxval=deficit + 1,
            dtype=tf.int32,
        )
    else:
        raise ValueError(f"Unknown audio pad_position: {pad_position!r}")
    return before, deficit - before


def _pad_zero(
    audio: tf.Tensor,
    target_samples: tf.Tensor,
    before: tf.Tensor,
    after: tf.Tensor,
) -> tf.Tensor:
    padded = tf.pad(audio, [[before, after], [0, 0]])
    return padded[:target_samples]


def _pad_repeat(
    audio: tf.Tensor,
    target_samples: tf.Tensor,
    before: tf.Tensor,
    after: tf.Tensor,
) -> tf.Tensor:
    t = tf.shape(audio)[0]
    channels = tf.shape(audio)[-1]

    def _empty() -> tf.Tensor:
        return tf.zeros([target_samples, channels], dtype=audio.dtype)

    def _repeat() -> tf.Tensor:
        indices = tf.range(-before, t + after)
        return tf.gather(audio, tf.math.floormod(indices, t))[:target_samples]

    return tf.cond(t > 0, _repeat, _empty)


def _pad_reflect(
    audio: tf.Tensor,
    target_samples: tf.Tensor,
    before: tf.Tensor,
    after: tf.Tensor,
) -> tf.Tensor:
    t = tf.shape(audio)[0]

    def _zero_pad() -> tf.Tensor:
        return _pad_zero(audio, target_samples, before, after)

    def _reflect() -> tf.Tensor:
        period = 2 * t - 2
        indices = tf.math.floormod(tf.range(-before, t + after), period)
        indices = tf.where(indices < t, indices, period - indices)
        return tf.gather(audio, indices)[:target_samples]

    return tf.cond(t > 1, _reflect, _zero_pad)


def pad_waveform(
    audio: tf.Tensor,
    target_samples: tf.Tensor,
    pad_mode: str,
    *,
    pad_position: str = "right",
    seed: tf.Tensor | int | None = None,
) -> tf.Tensor:
    audio = tf.convert_to_tensor(audio)
    before, after = _padding_counts(audio, target_samples, pad_position, seed)
    if pad_mode == "zero":
        return _pad_zero(audio, target_samples, before, after)
    if pad_mode == "repeat":
        return _pad_repeat(audio, target_samples, before, after)
    if pad_mode == "reflect":
        return _pad_reflect(audio, target_samples, before, after)
    raise ValueError(f"Unknown audio pad_mode: {pad_mode!r}")


def _crop_or_pad(
    audio: tf.Tensor,
    start: tf.Tensor,
    target_samples: tf.Tensor,
    pad_mode: str,
    pad_position: str,
    seed: tf.Tensor | int | None,
) -> tf.Tensor:
    t = tf.shape(audio)[0]
    start = tf.clip_by_value(start, 0, tf.maximum(t - 1, 0))
    available = tf.minimum(target_samples, tf.maximum(t - start, 0))
    cropped = audio[start : start + available]

    return tf.cond(
        tf.shape(cropped)[0] < target_samples,
        lambda: pad_waveform(
            cropped,
            target_samples,
            pad_mode,
            pad_position=pad_position,
            seed=seed,
        ),
        lambda: cropped[:target_samples],
    )


def _center_start(t: tf.Tensor, target_samples: tf.Tensor) -> tf.Tensor:
    return tf.maximum((t - target_samples) // 2, 0)


def _random_start(
    t: tf.Tensor,
    target_samples: tf.Tensor,
    seed: tf.Tensor | int | None,
) -> tf.Tensor:
    max_start = tf.maximum(t - target_samples, 0)

    def _sample() -> tf.Tensor:
        return tf.random.stateless_uniform(
            shape=[],
            seed=_seed_tensor(seed),
            minval=0,
            maxval=max_start + 1,
            dtype=tf.int32,
        )

    return tf.cond(max_start > 0, _sample, lambda: tf.constant(0, dtype=tf.int32))


def _multi_crop_starts(
    t: tf.Tensor,
    target_samples: tf.Tensor,
    num_views: int,
) -> tf.Tensor:
    if num_views == 1:
        return tf.reshape(_center_start(t, target_samples), [1])

    max_start = tf.maximum(t - target_samples, 0)
    starts = tf.round(
        tf.linspace(
            tf.constant(0.0, dtype=tf.float32),
            tf.cast(max_start, tf.float32),
            num_views,
        )
    )
    return tf.cast(starts, tf.int32)


def _sliding_starts(
    t: tf.Tensor,
    target_samples: tf.Tensor,
    sample_rate: tf.Tensor | int,
    config: SegmentStrategyConfig,
) -> tf.Tensor:
    max_start = tf.maximum(t - target_samples, 0)
    hop_duration = (
        config.sliding_hop_duration
        if config.sliding_hop_duration is not None
        else config.clip_duration
    )
    hop = tf.maximum(
        tf.cast(
            tf.round(
                tf.cast(hop_duration, tf.float32) * tf.cast(sample_rate, tf.float32)
            ),
            tf.int32,
        ),
        1,
    )
    starts = tf.range(0, max_start + 1, hop, dtype=tf.int32)
    return tf.cond(
        tf.equal(starts[-1], max_start),
        lambda: starts,
        lambda: tf.concat([starts, tf.reshape(max_start, [1])], axis=0),
    )


def _metadata(
    starts: tf.Tensor,
    lengths: tf.Tensor,
    sample_rate: tf.Tensor | int,
) -> dict:
    starts = tf.cast(starts, tf.int32)
    lengths = tf.cast(lengths, tf.int32)
    sample_rate = tf.cast(sample_rate, tf.float32)
    view_index = tf.range(tf.shape(starts)[0], dtype=tf.int32)
    return {
        "view_start_time": tf.cast(starts, tf.float32) / sample_rate,
        "view_end_time": tf.cast(starts + lengths, tf.float32) / sample_rate,
        "view_index": view_index,
    }


def _full_result(audio: tf.Tensor, sample_rate: tf.Tensor | int) -> dict:
    starts = tf.constant([0], dtype=tf.int32)
    lengths = tf.reshape(tf.shape(audio)[0], [1])
    return {AUDIO: audio, METADATA: _metadata(starts, lengths, sample_rate)}


def _views_result(
    audio: tf.Tensor,
    sample_rate: tf.Tensor | int,
    starts: tf.Tensor,
    target_samples: tf.Tensor,
    pad_mode: str,
    pad_position: str,
    seed: tf.Tensor | int | None,
) -> dict:
    starts = tf.cast(starts, tf.int32)

    def _view(start: tf.Tensor) -> tf.Tensor:
        return _crop_or_pad(audio, start, target_samples, pad_mode, pad_position, seed)

    views = tf.map_fn(_view, starts, fn_output_signature=audio.dtype)
    lengths = tf.fill(tf.shape(starts), target_samples)
    return {AUDIO: views, METADATA: _metadata(starts, lengths, sample_rate)}


def _single_result(
    audio: tf.Tensor,
    sample_rate: tf.Tensor | int,
    start: tf.Tensor,
    target_samples: tf.Tensor,
    pad_mode: str,
    pad_position: str,
    seed: tf.Tensor | int | None,
) -> dict:
    view = _crop_or_pad(audio, start, target_samples, pad_mode, pad_position, seed)
    starts = tf.reshape(tf.cast(start, tf.int32), [1])
    lengths = tf.reshape(target_samples, [1])
    return {AUDIO: view, METADATA: _metadata(starts, lengths, sample_rate)}


@register_audio_segment_strategy("random_crop")
@register_audio_segment_strategy("center_crop")
@register_audio_segment_strategy("full")
@register_audio_segment_strategy("sliding")
@register_audio_segment_strategy("pad_or_crop")
def segment_waveform(
    audio: tf.Tensor,
    sample_rate: tf.Tensor | int,
    config: SegmentStrategyConfig,
    *,
    is_training: bool,
    seed: tf.Tensor | int | None,
) -> dict:
    config = SegmentStrategyConfig.from_dict(config)
    audio = tf.convert_to_tensor(audio)
    sample_rate = tf.cast(sample_rate, tf.int32)
    target_samples = _target_samples(sample_rate, config)
    t = tf.shape(audio)[0]

    mode = config.train_mode if is_training else config.eval_mode
    pad_mode = config.pad_mode

    if config.duration_policy == "keep_1s":
        return _full_result(audio, sample_rate)
    if config.duration_policy == "pad_to_model_duration":
        mode = "pad_or_crop"
        pad_mode = "zero"
    elif config.duration_policy in {
        "tile_to_model_duration",
        "repeat_pad_to_model_duration",
    }:
        mode = "pad_or_crop"
        pad_mode = "repeat"
    elif config.duration_policy == "sliding_windows":
        mode = "sliding"

    if is_training and mode == "sliding" and not config.allow_train_sliding:
        raise ValueError(
            "sliding segmentation is not allowed during training unless "
            "allow_train_sliding=True"
        )

    if mode == "full":
        return _full_result(audio, sample_rate)
    if mode == "random_crop":
        return _single_result(
            audio,
            sample_rate,
            _random_start(t, target_samples, seed),
            target_samples,
            pad_mode,
            config.pad_position,
            seed,
        )
    if mode in {"center_crop", "pad_or_crop"}:
        return _single_result(
            audio,
            sample_rate,
            _center_start(t, target_samples),
            target_samples,
            pad_mode,
            config.pad_position,
            seed,
        )
    if mode == "multi_crop":
        starts = _multi_crop_starts(t, target_samples, config.num_views)
        return _views_result(
            audio,
            sample_rate,
            starts,
            target_samples,
            pad_mode,
            config.pad_position,
            seed,
        )
    if mode == "sliding":
        starts = _sliding_starts(t, target_samples, sample_rate, config)
        return _views_result(
            audio,
            sample_rate,
            starts,
            target_samples,
            pad_mode,
            config.pad_position,
            seed,
        )

    raise ValueError(f"Unknown segment mode: {mode!r}")


__all__ = [
    "pad_waveform",
    "segment_waveform",
]

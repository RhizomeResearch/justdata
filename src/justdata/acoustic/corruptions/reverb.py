from __future__ import annotations

import tensorflow as tf

from justdata.acoustic.corruptions.registry import (
    _EPS,
    _convolve_waveform,
    _sample_rate,
    _severity_value,
    _split_seed,
    register_audio_corruption,
)


REVERB_RT60_SECONDS = {1: 0.15, 2: 0.3, 3: 0.5, 4: 0.8, 5: 1.2}


@register_audio_corruption("reverb_rir")
def reverb_rir(
    audio_or_features: tf.Tensor,
    severity: int,
    seed: tf.Tensor,
    config: dict | None = None,
) -> tf.Tensor:
    rir = None if config is None else config.get("rir")
    if rir is not None:
        ir = tf.reshape(tf.cast(rir, tf.float32), [-1])
    else:
        sample_rate = _sample_rate(config)
        rt60 = _severity_value(REVERB_RT60_SECONDS, severity)
        max_ir_samples = (
            4096 if config is None else int(config.get("max_ir_samples", 4096))
        )
        target_length = tf.cast(tf.round(rt60 * sample_rate), tf.int32)
        length = tf.minimum(tf.maximum(target_length, 2), max_ir_samples)
        noise_seed = tf.unstack(_split_seed(seed, 1))[0]
        times = tf.cast(tf.range(length), tf.float32) / sample_rate
        decay = tf.pow(tf.constant(10.0, dtype=tf.float32), -3.0 * times / rt60)
        tail = tf.random.stateless_normal([length - 1], seed=noise_seed, stddev=0.05)
        ir = tf.concat([tf.ones([1], dtype=tf.float32), tail * decay[1:]], axis=0)

    ir = ir / tf.maximum(tf.sqrt(tf.reduce_sum(tf.square(ir))), _EPS)
    return _convolve_waveform(audio_or_features, ir, compensate_delay=True)


__all__ = [
    "REVERB_RT60_SECONDS",
    "reverb_rir",
]

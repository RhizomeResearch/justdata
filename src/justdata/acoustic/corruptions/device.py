from __future__ import annotations

import tensorflow as tf

from justdata.acoustic.corruptions.registry import (
    _convolve_waveform,
    _severity_value,
    register_audio_corruption,
)


DEVICE_IR_SET = {
    1: (0.02, 0.08, 0.80, 0.08, 0.02),
    2: (0.04, 0.12, 0.68, 0.12, 0.04),
    3: (0.08, 0.16, 0.52, 0.16, 0.08),
    4: (0.12, 0.20, 0.36, 0.20, 0.12),
    5: (0.16, 0.22, 0.24, 0.22, 0.16),
}


@register_audio_corruption("device_ir")
def device_ir(
    audio_or_features: tf.Tensor,
    severity: int,
    seed: tf.Tensor,
    config: dict | None = None,
) -> tf.Tensor:
    del seed
    if config is not None and "ir" in config:
        ir = tf.reshape(tf.cast(config["ir"], tf.float32), [-1])
    else:
        ir = _severity_value(DEVICE_IR_SET, severity)
    return _convolve_waveform(audio_or_features, ir, compensate_delay=False)


__all__ = [
    "DEVICE_IR_SET",
    "device_ir",
]

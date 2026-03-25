import numpy as np
import tensorflow as tf

from justdata.acoustic.configs import (
    FrontendConfig,
    LogCompressionConfig,
    MelConfig,
    STFTConfig,
)
from justdata.acoustic.frontends.log import compress_log, logmel


def _config(log_config=None):
    return FrontendConfig(
        name="logmel",
        stft=STFTConfig(
            sample_rate=16000,
            n_fft=400,
            win_length=400,
            hop_length=160,
        ),
        mel=MelConfig(n_mels=32),
        log=log_config or LogCompressionConfig(kind="log"),
    )


def test_logmel_log_offset():
    result = compress_log(
        tf.constant([0.0, 1.0]),
        LogCompressionConfig(kind="log", amin=1e-10, log_offset=1.0),
    )

    np.testing.assert_allclose(result.numpy(), np.log([1.0, 2.0]), rtol=1e-6)


def test_logmel_no_nan_on_zero_input():
    result = logmel(tf.zeros([16000, 1]), _config())

    assert bool(tf.reduce_all(tf.math.is_finite(result)))


def test_db_top_db_clamps():
    result = compress_log(
        tf.constant([1.0, 1e-6]),
        LogCompressionConfig(kind="db", ref="max", top_db=20.0),
    )

    assert np.isclose(float(tf.reduce_max(result)), 0.0)
    assert float(tf.reduce_min(result)) >= -20.0

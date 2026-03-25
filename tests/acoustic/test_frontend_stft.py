import tensorflow as tf

from justdata.acoustic.configs import FrontendConfig, STFTConfig
from justdata.acoustic.frontends.stft import stft_magnitude


def _config(*, center=True, window_periodic=True):
    return FrontendConfig(
        name="stft_magnitude",
        stft=STFTConfig(
            sample_rate=16000,
            n_fft=400,
            win_length=400,
            hop_length=160,
            center=center,
            window_periodic=window_periodic,
            power=1.0,
        ),
    )


def test_stft_shape_onesided():
    features = stft_magnitude(tf.ones([16000, 1]), _config(center=False))

    assert features.shape == (98, 201, 1)


def test_stft_center_padding_changes_frame_count():
    centered = stft_magnitude(tf.ones([16000, 1]), _config(center=True))
    uncentered = stft_magnitude(tf.ones([16000, 1]), _config(center=False))

    assert centered.shape[0] == 101
    assert uncentered.shape[0] == 98


def test_stft_hann_window_periodic_false_available():
    features = stft_magnitude(tf.ones([16000, 1]), _config(window_periodic=False))

    assert features.shape == (101, 201, 1)

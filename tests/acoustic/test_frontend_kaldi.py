import tensorflow as tf

from justdata.acoustic.configs import FrontendConfig, MelConfig, STFTConfig
from justdata.acoustic.frontends.kaldi import kaldi_fbank


def test_kaldi_fbank_shape_and_finite():
    config = FrontendConfig(
        name="kaldi_fbank",
        stft=STFTConfig(
            sample_rate=16000,
            n_fft=400,
            win_length=400,
            hop_length=160,
        ),
        mel=MelConfig(n_mels=40, filterbank_impl="kaldi_compatible"),
    )

    features = kaldi_fbank(tf.ones([16000, 1]), config)

    assert features.shape == (101, 40, 1)
    assert bool(tf.reduce_all(tf.math.is_finite(features)))

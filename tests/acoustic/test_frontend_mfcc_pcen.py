import tensorflow as tf

from justdata.acoustic.configs import (
    FrontendConfig,
    LogCompressionConfig,
    MelConfig,
    STFTConfig,
)
from justdata.acoustic.frontends.mfcc import mfcc
from justdata.acoustic.frontends.pcen import pcen_mel


def _base(name: str):
    return FrontendConfig(
        name=name,
        stft=STFTConfig(
            sample_rate=16000,
            n_fft=400,
            win_length=400,
            hop_length=160,
        ),
        mel=MelConfig(n_mels=40),
        log=LogCompressionConfig(kind="pcen" if name == "pcen_mel" else "log"),
        n_mfcc=13,
    )


def test_mfcc_shape():
    features = mfcc(tf.ones([16000, 1]), _base("mfcc"))

    assert features.shape == (101, 13, 1)


def test_pcen_shape_and_nonnegative():
    features = pcen_mel(tf.ones([16000, 1]), _base("pcen_mel"))

    assert features.shape == (101, 40, 1)
    assert float(tf.reduce_min(features)) >= 0.0

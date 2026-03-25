import tensorflow as tf

from justdata.acoustic.configs import FrontendConfig, MelConfig, STFTConfig
from justdata.acoustic.frontends.log import logmel
from justdata.acoustic.frontends.stft import raw_waveform
from justdata.acoustic.layouts import convert_audio_layout


def _logmel_config():
    return FrontendConfig(
        name="logmel",
        stft=STFTConfig(
            sample_rate=16000,
            n_fft=400,
            win_length=400,
            hop_length=160,
        ),
        mel=MelConfig(n_mels=32),
    )


def test_raw_waveform_layout_bt():
    waveform = raw_waveform(tf.ones([8000, 1]), FrontendConfig(name="raw_waveform"))
    result = convert_audio_layout(waveform, "bt", output_kind="waveform")

    assert result.shape == (8000,)


def test_logmel_layout_btf():
    features = logmel(tf.ones([16000, 1]), _logmel_config())
    result = convert_audio_layout(features, "btf", output_kind="features")

    assert result.shape == (101, 32)


def test_logmel_layout_bft():
    features = logmel(tf.ones([16000, 1]), _logmel_config())
    result = convert_audio_layout(features, "bft", output_kind="features")

    assert result.shape == (32, 101)


def test_logmel_layout_bcft():
    features = logmel(tf.ones([16000, 1]), _logmel_config())
    result = convert_audio_layout(features, "bcft", output_kind="features")

    assert result.shape == (1, 32, 101)


def test_multichannel_layout_btfc():
    features = logmel(tf.ones([16000, 2]), _logmel_config())
    result = convert_audio_layout(features, "btfc", output_kind="features")

    assert result.shape == (101, 32, 2)

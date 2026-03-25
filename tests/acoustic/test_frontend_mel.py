import pytest

from justdata.acoustic.configs import FrontendConfig, MelConfig, STFTConfig
from justdata.acoustic.frontends.mel import mel_weight_matrix


def _frontend(mel: MelConfig):
    return FrontendConfig(
        name="mel_power",
        stft=STFTConfig(
            sample_rate=16000,
            n_fft=400,
            win_length=400,
            hop_length=160,
        ),
        mel=mel,
    )


def test_mel_tf_matrix_shape():
    matrix = mel_weight_matrix(_frontend(MelConfig(n_mels=64)))

    assert matrix.shape == (201, 64)


def test_mel_slaney_matrix_shape():
    matrix = mel_weight_matrix(
        _frontend(
            MelConfig(
                n_mels=40,
                mel_scale="slaney",
                mel_norm="slaney",
                filterbank_impl="librosa",
            )
        )
    )

    assert matrix.shape == (201, 40)


def test_mel_invalid_fmax_above_nyquist_raises():
    with pytest.raises(ValueError, match="Nyquist"):
        mel_weight_matrix(_frontend(MelConfig(n_mels=64, f_max=9000.0)))

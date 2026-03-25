import pytest


pytestmark = pytest.mark.golden


def test_same_waveform_same_ced_features_against_hf_extractor():
    pytest.skip("CED Hugging Face feature-extractor fixture is not checked in.")


def test_same_waveform_same_ced_features_against_onnx_kaldi_path():
    pytest.skip("CED ONNX/Kaldi reference fixture is not checked into tests/acoustic/golden.")


def test_same_frontend_tensor_same_equimo_logits_after_checkpoint_conversion():
    pytest.skip("Equimo CED checkpoint-conversion fixture is not checked in.")

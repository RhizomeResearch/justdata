import pytest


pytestmark = pytest.mark.golden


def test_same_waveform_same_efficientat_logmel_against_torch():
    pytest.skip("EfficientAT torch frontend fixture is not checked into tests/acoustic/golden.")


def test_same_frontend_tensor_same_equimo_logits_after_checkpoint_conversion():
    pytest.skip("Equimo EfficientAT checkpoint-conversion fixture is not checked in.")

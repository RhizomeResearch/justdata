import pytest


pytestmark = pytest.mark.golden


def test_same_waveform_same_panns_logmel_against_reference():
    pytest.skip(
        "PANNs reference frontend fixture is not checked into tests/acoustic/golden."
    )

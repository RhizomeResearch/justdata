import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


pytestmark = pytest.mark.golden

GOLDEN_ROOT = Path(__file__).parent / "golden"
FAMILIES = ("ast", "efficientat", "passt", "ced", "panns")
MAX_FIXTURE_BYTES = 2 * 1024 * 1024


@pytest.mark.parametrize("family", FAMILIES)
def test_certified_family_metadata_and_fixture_budget(family):
    family_root = GOLDEN_ROOT / family
    metadata = json.loads((family_root / "metadata.json").read_text())

    assert metadata["schema_version"] == 1
    assert metadata["certification"] == "frontend_golden"
    assert metadata["source"]["repository"].startswith("https://")
    assert len(metadata["source"]["revision"]) == 40
    assert metadata["source"]["license"]
    assert metadata["tolerance"]["rationale"]
    assert "timestamp" not in metadata
    for fixture in family_root.glob("*.npz"):
        assert fixture.stat().st_size <= MAX_FIXTURE_BYTES


def test_shared_waveform_corpus_matches_deterministic_recipe():
    metadata = json.loads((GOLDEN_ROOT / "corpus.json").read_text())
    corpus = np.load(GOLDEN_ROOT / "corpus.npz")
    recipe = metadata["recipe"]

    for sample_rate in recipe["sample_rates"]:
        samples = sample_rate * recipe["duration_seconds"]
        time = np.arange(samples, dtype=np.float32) / np.float32(sample_rate)
        regenerated = np.zeros(samples, dtype=np.float32)
        for amplitude, frequency in zip(
            recipe["amplitudes"], recipe["frequencies_hz"], strict=True
        ):
            regenerated += np.float32(amplitude) * np.sin(
                np.float32(2.0 * np.pi * frequency) * time
            )
        name = f"waveform_{sample_rate}"
        np.testing.assert_array_equal(corpus[name], regenerated)
        assert (
            hashlib.sha256(regenerated.tobytes()).hexdigest()
            == metadata["sha256"][name]
        )

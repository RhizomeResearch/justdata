from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from corpus import RECIPE, array_sha256, synthetic_waveform


ROOT = Path(__file__).resolve().parent


def main() -> None:
    waveforms = {
        f"waveform_{sample_rate}": synthetic_waveform(sample_rate)
        for sample_rate in RECIPE["sample_rates"]
    }
    np.savez_compressed(ROOT / "corpus.npz", **waveforms)
    metadata = {
        "schema_version": 1,
        "license": "CC0-1.0 (deterministic synthetic signal; no source audio)",
        "recipe": RECIPE,
        "sha256": {
            name: array_sha256(waveform) for name, waveform in waveforms.items()
        },
    }
    (ROOT / "corpus.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torchaudio


ROOT = Path(__file__).resolve().parent
GOLDEN_ROOT = ROOT.parent
sys.path.insert(0, str(GOLDEN_ROOT))
from corpus import array_sha256  # noqa: E402


REFERENCE_REVISION = "2a5c818afcc2a215b2a1aaf1ed8be71f89d43201"


class _Ingredient:
    def __init__(self, _name: str):
        pass

    def command(self, value):
        return value


def _load_frontend(reference_root: Path):
    revision = subprocess.check_output(
        ["git", "-C", str(reference_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REFERENCE_REVISION:
        raise ValueError(f"Expected PaSST {REFERENCE_REVISION}, got {revision}")

    ba3l = types.ModuleType("ba3l")
    ingredients = types.ModuleType("ba3l.ingredients")
    ingredient = types.ModuleType("ba3l.ingredients.ingredient")
    ingredient.Ingredient = _Ingredient
    sys.modules.update(
        {"ba3l": ba3l, "ba3l.ingredients": ingredients, "ba3l.ingredients.ingredient": ingredient}
    )
    source = reference_root / "models" / "preprocess.py"
    spec = importlib.util.spec_from_file_location("passt_preprocess", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.AugmentMelSTFT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, required=True)
    args = parser.parse_args()

    waveform = np.load(GOLDEN_ROOT / "corpus.npz")["waveform_32000"]
    frontend_type = _load_frontend(args.reference_root)
    frontend = frontend_type(
        n_mels=128,
        sr=32000,
        win_length=800,
        hopsize=320,
        n_fft=1024,
        fmin=0.0,
        fmax=15000.0,
        fmin_aug_range=1,
        fmax_aug_range=1,
    ).eval()
    with torch.no_grad():
        features = frontend(torch.from_numpy(waveform)[None]).numpy()[0]
    np.savez_compressed(ROOT / "passt_32k_1s.npz", features=features)
    metadata = {
        "schema_version": 1,
        "family": "passt",
        "certification": "frontend_golden",
        "source": {
            "repository": "https://github.com/kkoutini/PaSST",
            "revision": REFERENCE_REVISION,
            "path": "models/preprocess.py:AugmentMelSTFT",
            "license": "Apache-2.0",
        },
        "dependencies": {"torch": torch.__version__, "torchaudio": torchaudio.__version__},
        "fixture": {
            "file": "passt_32k_1s.npz",
            "waveform": "../corpus.npz:waveform_32000",
            "features_layout": "frequency,time",
            "features_sha256": array_sha256(features),
            "patchout": False,
        },
        "tolerance": {"rtol": 0.003, "atol": 0.003, "rationale": "TensorFlow and Torch FFT kernel drift"},
        "logits": {"certification": "declared", "reason": "No licensed compact checkpoint-derived oracle is redistributed"},
    }
    (ROOT / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

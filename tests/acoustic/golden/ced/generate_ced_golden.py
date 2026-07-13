from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio


ROOT = Path(__file__).resolve().parent
GOLDEN_ROOT = ROOT.parent
sys.path.insert(0, str(GOLDEN_ROOT))
from corpus import array_sha256  # noqa: E402


HF_REVISION = "db3e14a8db4c21b56b165261c39649741a900e7f"


def main() -> None:
    waveform = np.load(GOLDEN_ROOT / "corpus.npz")["waveform_16000"]
    frontend = torch.nn.Sequential(
        torchaudio.transforms.MelSpectrogram(
            f_min=0.0,
            sample_rate=16000,
            win_length=512,
            center=True,
            n_fft=512,
            f_max=None,
            hop_length=160,
            n_mels=64,
        ),
        torchaudio.transforms.AmplitudeToDB(top_db=120),
    )
    with torch.no_grad():
        features = frontend(torch.from_numpy(waveform)[None]).numpy()[0]
    np.savez_compressed(ROOT / "ced_16k_1s.npz", features=features)
    metadata = {
        "schema_version": 1,
        "family": "ced",
        "certification": "frontend_golden",
        "source": {
            "repository": "https://huggingface.co/mispeech/ced-base",
            "revision": HF_REVISION,
            "path": "feature_extraction_ced.py:CedFeatureExtractor",
            "license": "Apache-2.0",
        },
        "dependencies": {
            "torch": torch.__version__,
            "torchaudio": torchaudio.__version__,
        },
        "fixture": {
            "file": "ced_16k_1s.npz",
            "waveform": "../corpus.npz:waveform_16000",
            "features_layout": "frequency,time",
            "features_sha256": array_sha256(features),
        },
        "tolerance": {
            "rtol": 0.005,
            "atol": 0.25,
            "rationale": "dB-scale drift in near-floor FFT bins across TensorFlow and Torch",
        },
        "onnx_kaldi": {
            "certification": "declared",
            "reason": "The upstream helper uses center=False and differs from the authoritative Hugging Face extractor",
        },
        "logits": {
            "certification": "declared",
            "reason": "No checkpoint-derived output is redistributed",
        },
    }
    (ROOT / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()

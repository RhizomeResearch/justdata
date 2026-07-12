from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchlibrosa
from torchlibrosa.stft import LogmelFilterBank, Spectrogram


ROOT = Path(__file__).resolve().parent
GOLDEN_ROOT = ROOT.parent
sys.path.insert(0, str(GOLDEN_ROOT))
from corpus import array_sha256  # noqa: E402


REFERENCE_REVISION = "d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4"
RECIPES = {
    "panns_32k_1s.npz": (32000, 1024, 320, 14000.0, 1),
    "panns_32k_10s_zero_pad.npz": (32000, 1024, 320, 14000.0, 10),
    "panns_16k_1s.npz": (16000, 512, 160, 8000.0, 1),
    "panns_16k_10s_zero_pad.npz": (16000, 512, 160, 8000.0, 10),
}


def main() -> None:
    corpus = np.load(GOLDEN_ROOT / "corpus.npz")
    fixture_metadata = {}
    for filename, (sample_rate, window_size, hop_size, f_max, duration) in RECIPES.items():
        waveform = corpus[f"waveform_{sample_rate}"]
        waveform = np.pad(waveform, (0, sample_rate * (duration - 1)))
        spectrogram = Spectrogram(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window="hann",
            center=True,
            pad_mode="reflect",
            freeze_parameters=True,
        )
        logmel = LogmelFilterBank(
            sr=sample_rate,
            n_fft=window_size,
            n_mels=64,
            fmin=50.0,
            fmax=f_max,
            ref=1.0,
            amin=1e-10,
            top_db=None,
            freeze_parameters=True,
        )
        with torch.no_grad():
            features = logmel(spectrogram(torch.from_numpy(waveform)[None])).numpy()[0, 0]
        np.savez_compressed(ROOT / filename, features=features)
        fixture_metadata[filename] = {
            "waveform": f"../corpus.npz:waveform_{sample_rate}",
            "zero_padded_duration_seconds": duration,
            "features_layout": "time,frequency",
            "features_sha256": array_sha256(features),
        }

    metadata = {
        "schema_version": 1,
        "family": "panns",
        "certification": "frontend_golden",
        "source": {
            "repository": "https://github.com/qiuqiangkong/audioset_tagging_cnn",
            "revision": REFERENCE_REVISION,
            "path": "pytorch/models.py:Cnn14 Spectrogram/LogmelFilterBank",
            "license": "MIT",
        },
        "dependencies": {"torch": torch.__version__, "torchlibrosa": torchlibrosa.__version__},
        "fixtures": fixture_metadata,
        "tolerance": {"rtol": 0.001, "atol": 0.06, "rationale": "dB-scale FFT kernel drift across TensorFlow and TorchLibrosa"},
        "logits": {"certification": "declared", "reason": "No checkpoint-derived output is redistributed"},
    }
    (ROOT / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

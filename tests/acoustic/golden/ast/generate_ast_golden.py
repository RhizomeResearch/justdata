from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torchaudio


ROOT = Path(__file__).resolve().parent

RECIPES = {
    "ast_audioset_eval.npz": {
        "samples": 160000,
        "target_length": 1024,
        "mean": -4.2677393,
        "std": 4.5689974,
    },
    "ast_esc50_eval.npz": {
        "samples": 80000,
        "target_length": 512,
        "mean": -6.6268077,
        "std": 5.358466,
    },
    "ast_speechcommands_eval.npz": {
        "samples": 16000,
        "target_length": 128,
        "mean": -6.845978,
        "std": 5.5654526,
    },
}


def _synthetic_waveform(num_samples: int) -> np.ndarray:
    t = np.arange(num_samples, dtype=np.float32) / 16000.0
    waveform = 0.35 * np.sin(2.0 * np.pi * 440.0 * t)
    waveform += 0.15 * np.sin(2.0 * np.pi * 1370.0 * t)
    return waveform.astype(np.float32)


def _reference_features(
    waveform_np: np.ndarray,
    *,
    target_length: int,
    mean: float,
    std: float,
) -> tuple[np.ndarray, np.ndarray]:
    waveform = torch.from_numpy(waveform_np).unsqueeze(0)
    waveform = waveform - waveform.mean()
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=16000,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=128,
        dither=0.0,
        frame_shift=10,
    )
    pad = target_length - fbank.shape[0]
    if pad > 0:
        fbank = torch.nn.ZeroPad2d((0, 0, 0, pad))(fbank)
    elif pad < 0:
        fbank = fbank[:target_length, :]
    fbank = (fbank - mean) / (std * 2)
    return waveform.transpose(0, 1).numpy(), fbank.numpy()


def main() -> None:
    metadata = {
        "torch": torch.__version__,
        "torchaudio": torchaudio.__version__,
        "recipes": RECIPES,
    }
    for filename, recipe in RECIPES.items():
        source = _synthetic_waveform(recipe["samples"])
        waveform, features = _reference_features(
            source,
            target_length=recipe["target_length"],
            mean=recipe["mean"],
            std=recipe["std"],
        )
        np.savez_compressed(ROOT / filename, waveform=waveform, features=features)

    (ROOT / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

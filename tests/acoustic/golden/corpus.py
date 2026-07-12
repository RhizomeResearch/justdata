from __future__ import annotations

import hashlib

import numpy as np


RECIPE = {
    "dtype": "float32",
    "duration_seconds": 1,
    "frequencies_hz": [440.0, 1370.0],
    "amplitudes": [0.35, 0.15],
    "sample_rates": [16000, 32000],
}


def synthetic_waveform(sample_rate: int) -> np.ndarray:
    samples = sample_rate * RECIPE["duration_seconds"]
    time = np.arange(samples, dtype=np.float32) / np.float32(sample_rate)
    waveform = np.zeros(samples, dtype=np.float32)
    for amplitude, frequency in zip(
        RECIPE["amplitudes"], RECIPE["frequencies_hz"], strict=True
    ):
        waveform += np.float32(amplitude) * np.sin(
            np.float32(2.0 * np.pi * frequency) * time
        )
    return waveform.astype(np.float32)


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()

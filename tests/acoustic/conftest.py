import wave
from pathlib import Path

import numpy as np
import pytest


def _write_wav(path: Path, data, sample_rate: int = 16000) -> Path:
    samples = np.asarray(data, dtype=np.int16)
    if samples.ndim == 1:
        channels = 1
    elif samples.ndim == 2:
        channels = samples.shape[1]
    else:
        raise ValueError("WAV fixture data must be rank 1 or 2")

    with wave.open(str(path), "wb") as f:
        f.setnchannels(channels)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(samples.tobytes())
    return path


@pytest.fixture
def write_wav_file():
    return _write_wav

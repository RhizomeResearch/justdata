from typing import Any, NotRequired, TypedDict


WAVEFORM = "waveform"
SAMPLE_RATE = "sample_rate"
LABEL = "label"
FEATURES = "features"
DURATION = "duration"
METADATA = "metadata"


class AudioSample(TypedDict):
    waveform: Any
    sample_rate: int
    label: NotRequired[Any]
    features: NotRequired[Any]
    duration: NotRequired[float]
    metadata: NotRequired[dict[str, Any]]


__all__ = [
    "DURATION",
    "FEATURES",
    "LABEL",
    "METADATA",
    "SAMPLE_RATE",
    "WAVEFORM",
    "AudioSample",
]

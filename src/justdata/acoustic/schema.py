from typing import Any, NotRequired, TypedDict


AUDIO = "audio"
WAVEFORM = "waveform"
SAMPLE_RATE = "sample_rate"
ORIGINAL_SAMPLE_RATE = "original_sample_rate"
LABEL = "label"
FEATURES = "features"
DURATION = "duration"
METADATA = "metadata"
PATH = "path"
FILENAME = "filename"
DATASET = "dataset"
SPLIT = "split"
EXAMPLE_ID = "example_id"
CLIP_ID = "clip_id"
SOURCE_ID = "source_id"
START_TIME = "start_time"
END_TIME = "end_time"


class AudioSample(TypedDict):
    waveform: Any
    sample_rate: int
    label: NotRequired[Any]
    features: NotRequired[Any]
    duration: NotRequired[float]
    metadata: NotRequired[dict[str, Any]]


__all__ = [
    "AUDIO",
    "CLIP_ID",
    "DATASET",
    "DURATION",
    "END_TIME",
    "EXAMPLE_ID",
    "FEATURES",
    "FILENAME",
    "LABEL",
    "METADATA",
    "ORIGINAL_SAMPLE_RATE",
    "PATH",
    "SAMPLE_RATE",
    "SOURCE_ID",
    "SPLIT",
    "START_TIME",
    "WAVEFORM",
    "AudioSample",
]

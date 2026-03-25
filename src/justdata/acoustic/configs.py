from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar


ConfigT = TypeVar("ConfigT", bound="_SerializableConfig")


def _canonical_json(config: Any) -> str:
    return json.dumps(dataclasses.asdict(config), sort_keys=True, separators=(",", ":"))


def _short_hash(config: Any) -> str:
    return hashlib.sha256(_canonical_json(config).encode("utf-8")).hexdigest()[:16]


def _ensure_literal(field_name: str, value: Any, allowed: set[Any]) -> None:
    if value not in allowed:
        choices = ", ".join(repr(v) for v in sorted(allowed, key=str))
        raise ValueError(f"{field_name} must be one of {choices}; got {value!r}")


def _ensure_positive(field_name: str, value: int | float | None) -> None:
    if (
        value is None
        or isinstance(value, bool)
        or not isinstance(value, (int, float))
        or value <= 0
    ):
        raise ValueError(f"{field_name} must be positive; got {value!r}")


def _tuple_value(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(value)
    return value


class _SerializableConfig:
    def __post_init__(self) -> None:
        self.validate()

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return _canonical_json(self)

    @classmethod
    def from_dict(cls: type[ConfigT], data: Mapping[str, Any] | ConfigT) -> ConfigT:
        if isinstance(data, cls):
            return data
        return cls(**dict(data))

    @classmethod
    def from_json(cls: type[ConfigT], data: str) -> ConfigT:
        return cls.from_dict(json.loads(data))

    def hash(self) -> str:
        return _short_hash(self)

    def validate(self) -> _SerializableConfig:
        return self


@dataclass(frozen=True)
class AudioPreprocessConfig(_SerializableConfig):
    target_sample_rate: int
    dtype: Literal["float32"] = "float32"
    waveform_range: Literal["[-1,1]"] = "[-1,1]"
    channel_strategy: Literal["mono_mean", "mono_left", "mono_right", "keep"] = "mono_mean"
    resampler: Literal["tensorflow", "soxr", "kaiser_best", "hf_audio", "identity"] = (
        "tensorflow"
    )
    normalize_waveform: Literal["none", "peak", "rms", "lufs"] = "none"
    remove_dc_offset: bool = False
    clip_value: float | None = None
    cache_after_resample: bool = True

    def validate(self) -> AudioPreprocessConfig:
        _ensure_positive("target_sample_rate", self.target_sample_rate)
        _ensure_literal("dtype", self.dtype, {"float32"})
        _ensure_literal("waveform_range", self.waveform_range, {"[-1,1]"})
        _ensure_literal(
            "channel_strategy",
            self.channel_strategy,
            {"mono_mean", "mono_left", "mono_right", "keep"},
        )
        _ensure_literal(
            "resampler",
            self.resampler,
            {"tensorflow", "soxr", "kaiser_best", "hf_audio", "identity"},
        )
        _ensure_literal(
            "normalize_waveform", self.normalize_waveform, {"none", "peak", "rms", "lufs"}
        )
        if self.clip_value is not None and self.clip_value <= 0:
            raise ValueError("clip_value must be positive when set")
        return self


@dataclass(frozen=True)
class SegmentStrategyConfig(_SerializableConfig):
    clip_duration: float
    train_mode: Literal["random_crop", "center_crop", "full", "sliding", "pad_or_crop"]
    eval_mode: Literal["center_crop", "full", "sliding", "multi_crop"]
    pad_mode: Literal["zero", "repeat", "reflect"]
    pad_position: Literal["right", "center", "random"]
    num_views: int = 1
    sliding_hop_duration: float | None = None
    allow_train_sliding: bool = False
    drop_short: bool = False
    min_duration: float | None = None
    duration_policy: Literal[
        "keep_1s",
        "pad_to_model_duration",
        "tile_to_model_duration",
        "repeat_pad_to_model_duration",
        "sliding_windows",
        "none",
    ] = "none"

    def validate(self) -> SegmentStrategyConfig:
        _ensure_positive("clip_duration", self.clip_duration)
        _ensure_literal(
            "train_mode",
            self.train_mode,
            {"random_crop", "center_crop", "full", "sliding", "pad_or_crop"},
        )
        _ensure_literal("eval_mode", self.eval_mode, {"center_crop", "full", "sliding", "multi_crop"})
        _ensure_literal("pad_mode", self.pad_mode, {"zero", "repeat", "reflect"})
        _ensure_literal("pad_position", self.pad_position, {"right", "center", "random"})
        _ensure_positive("num_views", self.num_views)
        if self.sliding_hop_duration is not None:
            _ensure_positive("sliding_hop_duration", self.sliding_hop_duration)
        _ensure_literal(
            "duration_policy",
            self.duration_policy,
            {
                "keep_1s",
                "pad_to_model_duration",
                "tile_to_model_duration",
                "repeat_pad_to_model_duration",
                "sliding_windows",
                "none",
            },
        )
        if self.min_duration is not None:
            _ensure_positive("min_duration", self.min_duration)
        return self


@dataclass(frozen=True)
class STFTConfig(_SerializableConfig):
    sample_rate: int
    n_fft: int
    win_length: int
    hop_length: int
    window: Literal["hann", "hamming", "povey", "rectangular"] = "hann"
    center: bool = True
    pad_mode: Literal["reflect", "constant"] = "reflect"
    power: float = 2.0
    normalized: bool = False
    onesided: bool = True
    eps: float = 1e-10

    def validate(self) -> STFTConfig:
        _ensure_positive("sample_rate", self.sample_rate)
        _ensure_positive("n_fft", self.n_fft)
        _ensure_positive("win_length", self.win_length)
        _ensure_positive("hop_length", self.hop_length)
        if self.win_length > self.n_fft:
            raise ValueError("win_length must be <= n_fft")
        _ensure_literal("window", self.window, {"hann", "hamming", "povey", "rectangular"})
        _ensure_literal("pad_mode", self.pad_mode, {"reflect", "constant"})
        _ensure_positive("power", self.power)
        _ensure_positive("eps", self.eps)
        return self


@dataclass(frozen=True)
class MelConfig(_SerializableConfig):
    n_mels: int
    f_min: float = 0.0
    f_max: float | None = None
    mel_scale: Literal["htk", "slaney"] = "htk"
    mel_norm: Literal["none", "slaney"] = "none"
    filterbank_impl: Literal["tf", "librosa", "torchaudio", "kaldi_compatible"] = "tf"

    def validate(self) -> MelConfig:
        _ensure_positive("n_mels", self.n_mels)
        if self.f_min < 0:
            raise ValueError("f_min must be non-negative")
        if self.f_max is not None and self.f_max <= self.f_min:
            raise ValueError("f_max must be greater than f_min")
        _ensure_literal("mel_scale", self.mel_scale, {"htk", "slaney"})
        _ensure_literal("mel_norm", self.mel_norm, {"none", "slaney"})
        _ensure_literal(
            "filterbank_impl", self.filterbank_impl, {"tf", "librosa", "torchaudio", "kaldi_compatible"}
        )
        return self


@dataclass(frozen=True)
class LogCompressionConfig(_SerializableConfig):
    kind: Literal["log", "log10", "db", "pcen"] = "log"
    amin: float = 1e-10
    ref: float | Literal["max"] = 1.0
    top_db: float | None = None
    log_offset: float = 0.0

    def validate(self) -> LogCompressionConfig:
        _ensure_literal("kind", self.kind, {"log", "log10", "db", "pcen"})
        _ensure_positive("amin", self.amin)
        if self.ref != "max":
            _ensure_positive("ref", self.ref)
        if self.top_db is not None:
            _ensure_positive("top_db", self.top_db)
        if self.log_offset < 0:
            raise ValueError("log_offset must be non-negative")
        return self


@dataclass(frozen=True)
class FeatureNormConfig(_SerializableConfig):
    kind: Literal[
        "none",
        "dataset_mean_std",
        "per_clip_mean_std",
        "per_frequency_mean_std",
        "kaldi_cmvn",
        "checkpoint_mean_std",
        "affine",
    ] = "none"
    mean: tuple[float, ...] | str | None = None
    std: tuple[float, ...] | str | None = None
    axes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "mean", _tuple_value(self.mean))
        object.__setattr__(self, "std", _tuple_value(self.std))
        object.__setattr__(self, "axes", _tuple_value(self.axes))
        self.validate()

    def validate(self) -> FeatureNormConfig:
        _ensure_literal(
            "kind",
            self.kind,
            {
                "none",
                "dataset_mean_std",
                "per_clip_mean_std",
                "per_frequency_mean_std",
                "kaldi_cmvn",
                "checkpoint_mean_std",
                "affine",
            },
        )
        for field_name, value in (("mean", self.mean), ("std", self.std)):
            if value is not None and not isinstance(value, (tuple, str)):
                raise ValueError(f"{field_name} must be a tuple, string, or None")
        if not isinstance(self.axes, tuple):
            raise ValueError("axes must be a tuple")
        return self


@dataclass(frozen=True)
class FrontendConfig(_SerializableConfig):
    name: Literal[
        "raw_waveform",
        "stft_magnitude",
        "mel_power",
        "logmel",
        "kaldi_fbank",
        "mfcc",
        "pcen_mel",
    ]
    stft: STFTConfig | None = None
    mel: MelConfig | None = None
    log: LogCompressionConfig | None = None
    norm: FeatureNormConfig = FeatureNormConfig()

    def __post_init__(self) -> None:
        if isinstance(self.stft, Mapping):
            object.__setattr__(self, "stft", STFTConfig.from_dict(self.stft))
        if isinstance(self.mel, Mapping):
            object.__setattr__(self, "mel", MelConfig.from_dict(self.mel))
        if isinstance(self.log, Mapping):
            object.__setattr__(self, "log", LogCompressionConfig.from_dict(self.log))
        if isinstance(self.norm, Mapping):
            object.__setattr__(self, "norm", FeatureNormConfig.from_dict(self.norm))
        self.validate()

    def validate(self) -> FrontendConfig:
        _ensure_literal(
            "name",
            self.name,
            {
                "raw_waveform",
                "stft_magnitude",
                "mel_power",
                "logmel",
                "kaldi_fbank",
                "mfcc",
                "pcen_mel",
            },
        )
        if self.stft is not None and not isinstance(self.stft, STFTConfig):
            raise ValueError("stft must be an STFTConfig or None")
        if self.mel is not None and not isinstance(self.mel, MelConfig):
            raise ValueError("mel must be a MelConfig or None")
        if self.log is not None and not isinstance(self.log, LogCompressionConfig):
            raise ValueError("log must be a LogCompressionConfig or None")
        if not isinstance(self.norm, FeatureNormConfig):
            raise ValueError("norm must be a FeatureNormConfig")
        return self


@dataclass(frozen=True)
class LabelTransformConfig(_SerializableConfig):
    mode: Literal["index", "one_hot", "multi_hot", "text", "event_frames"]
    num_classes: int | None = None
    class_names: tuple[str, ...] | None = None
    smoothing: float = 0.0
    keep_hard_label_in_metadata: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "class_names", _tuple_value(self.class_names))
        self.validate()

    def validate(self) -> LabelTransformConfig:
        _ensure_literal("mode", self.mode, {"index", "one_hot", "multi_hot", "text", "event_frames"})
        if self.num_classes is not None:
            _ensure_positive("num_classes", self.num_classes)
        if self.class_names is not None and not isinstance(self.class_names, tuple):
            raise ValueError("class_names must be a tuple or None")
        if not 0.0 <= self.smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1)")
        return self


@dataclass(frozen=True)
class AudioPreset(_SerializableConfig):
    name: str
    input_duration: float | None
    target_sample_rate: int
    preprocess: AudioPreprocessConfig
    segment: SegmentStrategyConfig
    frontend: FrontendConfig
    label_transform: LabelTransformConfig
    layout: Literal["bt", "btc", "btf", "bft", "bcft", "btfc"]
    dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    train_augment: dict[str, Any] = field(default_factory=dict)
    eval_views: dict[str, Any] = field(default_factory=dict)
    metadata_mode: Literal["full", "numeric_only", "none"] = "numeric_only"

    def __post_init__(self) -> None:
        if isinstance(self.preprocess, Mapping):
            object.__setattr__(self, "preprocess", AudioPreprocessConfig.from_dict(self.preprocess))
        if isinstance(self.segment, Mapping):
            object.__setattr__(self, "segment", SegmentStrategyConfig.from_dict(self.segment))
        if isinstance(self.frontend, Mapping):
            object.__setattr__(self, "frontend", FrontendConfig.from_dict(self.frontend))
        if isinstance(self.label_transform, Mapping):
            object.__setattr__(
                self,
                "label_transform",
                LabelTransformConfig.from_dict(self.label_transform),
            )
        object.__setattr__(self, "train_augment", dict(self.train_augment))
        object.__setattr__(self, "eval_views", dict(self.eval_views))
        self.validate()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | AudioPreset) -> AudioPreset:
        if isinstance(data, cls):
            return data
        if "target_sample_rate" not in data:
            raise ValueError("AudioPreset requires target_sample_rate")
        return cls(**dict(data))

    def validate(self) -> AudioPreset:
        if not self.name:
            raise ValueError("name must be non-empty")
        _ensure_positive("target_sample_rate", self.target_sample_rate)
        if self.input_duration is not None:
            _ensure_positive("input_duration", self.input_duration)
        if not isinstance(self.preprocess, AudioPreprocessConfig):
            raise ValueError("preprocess must be an AudioPreprocessConfig")
        if not isinstance(self.segment, SegmentStrategyConfig):
            raise ValueError("segment must be a SegmentStrategyConfig")
        if not isinstance(self.frontend, FrontendConfig):
            raise ValueError("frontend must be a FrontendConfig")
        if not isinstance(self.label_transform, LabelTransformConfig):
            raise ValueError("label_transform must be a LabelTransformConfig")
        if self.preprocess.target_sample_rate != self.target_sample_rate:
            raise ValueError("target_sample_rate must match preprocess.target_sample_rate")
        _ensure_literal("layout", self.layout, {"bt", "btc", "btf", "bft", "bcft", "btfc"})
        _ensure_literal("dtype", self.dtype, {"float32", "float16", "bfloat16"})
        _ensure_literal("metadata_mode", self.metadata_mode, {"full", "numeric_only", "none"})
        return self

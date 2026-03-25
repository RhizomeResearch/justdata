from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


def _ensure_probability(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]; got {value!r}")


def _ensure_order(name_min: str, min_value: float, name_max: str, max_value: float) -> None:
    if min_value > max_value:
        raise ValueError(f"{name_min} must be <= {name_max}")


@dataclass(frozen=True)
class RandomGainConfig:
    prob: float = 0.5
    min_db: float = -12.0
    max_db: float = 12.0

    def __post_init__(self) -> None:
        _ensure_probability("prob", self.prob)
        _ensure_order("min_db", self.min_db, "max_db", self.max_db)


@dataclass(frozen=True)
class RandomTimeShiftConfig:
    prob: float = 0.5
    max_shift_seconds: float = 0.5
    mode: Literal["roll", "zero"] = "roll"

    def __post_init__(self) -> None:
        _ensure_probability("prob", self.prob)
        if self.max_shift_seconds < 0:
            raise ValueError("max_shift_seconds must be non-negative")
        if self.mode not in {"roll", "zero"}:
            raise ValueError("mode must be one of 'roll' or 'zero'")


@dataclass(frozen=True)
class AdditiveNoiseConfig:
    prob: float = 0.5
    snr_db_min: float = 5.0
    snr_db_max: float = 30.0
    noise_kind: Literal["white", "pink", "brown"] = "white"

    def __post_init__(self) -> None:
        _ensure_probability("prob", self.prob)
        _ensure_order("snr_db_min", self.snr_db_min, "snr_db_max", self.snr_db_max)
        if self.noise_kind not in {"white", "pink", "brown"}:
            raise ValueError("noise_kind must be one of 'white', 'pink', or 'brown'")


@dataclass(frozen=True)
class RandomClippingConfig:
    prob: float = 0.2
    min_threshold: float = 0.2
    max_threshold: float = 1.0

    def __post_init__(self) -> None:
        _ensure_probability("prob", self.prob)
        if self.min_threshold <= 0 or self.max_threshold <= 0:
            raise ValueError("clipping thresholds must be positive")
        _ensure_order(
            "min_threshold",
            self.min_threshold,
            "max_threshold",
            self.max_threshold,
        )


@dataclass(frozen=True)
class DynamicRangeCompressionConfig:
    prob: float = 0.3
    threshold_db_min: float = -30.0
    threshold_db_max: float = -6.0
    ratio_min: float = 2.0
    ratio_max: float = 8.0

    def __post_init__(self) -> None:
        _ensure_probability("prob", self.prob)
        _ensure_order(
            "threshold_db_min",
            self.threshold_db_min,
            "threshold_db_max",
            self.threshold_db_max,
        )
        if self.ratio_min < 1.0 or self.ratio_max < 1.0:
            raise ValueError("compression ratios must be >= 1")
        _ensure_order("ratio_min", self.ratio_min, "ratio_max", self.ratio_max)


@dataclass(frozen=True)
class RIRConvolutionConfig:
    prob: float = 0.3
    rir_dataset: str | None = None
    normalize_ir: bool = True
    compensate_delay: bool = True

    def __post_init__(self) -> None:
        _ensure_probability("prob", self.prob)


@dataclass(frozen=True)
class SpeedPerturbConfig:
    prob: float = 0.3
    rates: tuple[float, ...] = (0.9, 1.0, 1.1)

    def __post_init__(self) -> None:
        _ensure_probability("prob", self.prob)
        object.__setattr__(self, "rates", tuple(self.rates))
        if not self.rates:
            raise ValueError("rates must be non-empty")
        if any(rate <= 0 for rate in self.rates):
            raise ValueError("all speed perturb rates must be positive")


@dataclass(frozen=True)
class CodecSimulationConfig:
    prob: float = 0.2
    codec: Literal["mulaw", "alaw", "mp3_proxy"] = "mulaw"
    bitrate: int | None = None

    def __post_init__(self) -> None:
        _ensure_probability("prob", self.prob)
        if self.codec not in {"mulaw", "alaw", "mp3_proxy"}:
            raise ValueError("codec must be one of 'mulaw', 'alaw', or 'mp3_proxy'")
        if self.bitrate is not None and self.bitrate <= 0:
            raise ValueError("bitrate must be positive when set")


__all__ = [
    "AdditiveNoiseConfig",
    "CodecSimulationConfig",
    "DynamicRangeCompressionConfig",
    "RIRConvolutionConfig",
    "RandomClippingConfig",
    "RandomGainConfig",
    "RandomTimeShiftConfig",
    "SpeedPerturbConfig",
]

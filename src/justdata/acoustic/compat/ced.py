from __future__ import annotations

from justdata.acoustic.compat.efficientat import DCASE_CLASSES
from justdata.acoustic.configs import (
    AudioPreprocessConfig,
    AudioPreset,
    FrontendConfig,
    LabelTransformConfig,
    LogCompressionConfig,
    MelConfig,
    STFTConfig,
    SegmentStrategyConfig,
)


def ced_frontend() -> FrontendConfig:
    return FrontendConfig(
        name="kaldi_fbank",
        stft=STFTConfig(
            sample_rate=16000,
            n_fft=400,
            win_length=400,
            hop_length=160,
            window="povey",
            center=False,
            power=2.0,
        ),
        mel=MelConfig(
            n_mels=64,
            f_min=0.0,
            f_max=8000.0,
            mel_scale="htk",
            mel_norm="none",
            filterbank_impl="kaldi_compatible",
        ),
        log=LogCompressionConfig(kind="log"),
    )


def _preprocess() -> AudioPreprocessConfig:
    return AudioPreprocessConfig(
        target_sample_rate=16000,
        channel_strategy="mono_mean",
        normalize_waveform="none",
    )


def _metadata(model_size: str, *, source_duration: float | None = None) -> dict:
    metadata = {
        "preset_version": 1,
        "model_family": "ced",
        "model_size": model_size,
        "frontend_contract": "ced-kaldi-fbank-v1",
        "compatibility_status": "frontend_declared_golden_pending",
    }
    if source_duration is not None:
        metadata["source_duration"] = source_duration
    return metadata


def _audioset_label_transform() -> LabelTransformConfig:
    return LabelTransformConfig(mode="multi_hot", num_classes=527)


def _dcase_label_transform() -> LabelTransformConfig:
    return LabelTransformConfig(
        mode="index",
        num_classes=len(DCASE_CLASSES),
        class_names=DCASE_CLASSES,
    )


def _preset(
    *,
    name: str,
    model_size: str,
    input_duration: float,
    segment: SegmentStrategyConfig,
    label_transform: LabelTransformConfig,
    metadata: dict | None = None,
) -> AudioPreset:
    return AudioPreset(
        name=name,
        input_duration=input_duration,
        target_sample_rate=16000,
        preprocess=_preprocess(),
        segment=segment,
        frontend=ced_frontend(),
        layout="btf",
        label_transform=label_transform,
        metadata=metadata or _metadata(model_size),
    )


def _audioset_preset(name: str, model_size: str) -> AudioPreset:
    return _preset(
        name=name,
        model_size=model_size,
        input_duration=10.0,
        segment=SegmentStrategyConfig(
            clip_duration=10.0,
            train_mode="random_crop",
            eval_mode="center_crop",
            pad_mode="zero",
            pad_position="right",
        ),
        label_transform=_audioset_label_transform(),
    )


def ced_tiny_16k_logmel64() -> AudioPreset:
    return _audioset_preset("ced_tiny_16k_logmel64", "tiny")


def ced_mini_16k_logmel64() -> AudioPreset:
    return _audioset_preset("ced_mini_16k_logmel64", "mini")


def ced_small_16k_logmel64() -> AudioPreset:
    return _audioset_preset("ced_small_16k_logmel64", "small")


def ced_base_16k_logmel64() -> AudioPreset:
    return _audioset_preset("ced_base_16k_logmel64", "base")


def dcase2025_task1_ced_16k_1s() -> AudioPreset:
    return _preset(
        name="dcase2025_task1_ced_16k_1s",
        model_size="base",
        input_duration=1.0,
        segment=SegmentStrategyConfig(
            clip_duration=1.0,
            train_mode="random_crop",
            eval_mode="center_crop",
            pad_mode="zero",
            pad_position="right",
            duration_policy="keep_1s",
        ),
        label_transform=_dcase_label_transform(),
        metadata=_metadata("base", source_duration=1.0),
    )


__all__ = [
    "ced_base_16k_logmel64",
    "ced_frontend",
    "ced_mini_16k_logmel64",
    "ced_small_16k_logmel64",
    "ced_tiny_16k_logmel64",
    "dcase2025_task1_ced_16k_1s",
]

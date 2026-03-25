from __future__ import annotations

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


def panns_frontend(
    *,
    sample_rate: int,
    window_size: int,
    hop_size: int,
    mel_bins: int,
    fmin: float,
    fmax: float,
) -> FrontendConfig:
    return FrontendConfig(
        name="logmel",
        stft=STFTConfig(
            sample_rate=sample_rate,
            n_fft=window_size,
            win_length=window_size,
            hop_length=hop_size,
            center=True,
            power=2.0,
        ),
        mel=MelConfig(
            n_mels=mel_bins,
            f_min=fmin,
            f_max=fmax,
            mel_scale="slaney",
            mel_norm="slaney",
            filterbank_impl="librosa",
        ),
        log=LogCompressionConfig(kind="log"),
    )


def _preset(
    *,
    name: str,
    sample_rate: int,
    frontend: FrontendConfig,
) -> AudioPreset:
    return AudioPreset(
        name=name,
        input_duration=10.0,
        target_sample_rate=sample_rate,
        preprocess=AudioPreprocessConfig(
            target_sample_rate=sample_rate,
            channel_strategy="mono_mean",
            normalize_waveform="none",
        ),
        segment=SegmentStrategyConfig(
            clip_duration=10.0,
            train_mode="random_crop",
            eval_mode="center_crop",
            pad_mode="zero",
            pad_position="right",
        ),
        frontend=frontend,
        layout="btf",
        label_transform=LabelTransformConfig(mode="multi_hot", num_classes=527),
        metadata={
            "preset_version": 1,
            "model_family": "panns",
            "model_name": "cnn14",
            "frontend_contract": "panns-cnn14-logmel-v1",
            "compatibility_status": "reference_frontend_declared_golden_pending",
        },
    )


def panns_cnn14_32k_10s_logmel64() -> AudioPreset:
    return _preset(
        name="panns_cnn14_32k_10s_logmel64",
        sample_rate=32000,
        frontend=panns_frontend(
            sample_rate=32000,
            window_size=1024,
            hop_size=320,
            mel_bins=64,
            fmin=50.0,
            fmax=14000.0,
        ),
    )


def panns_cnn14_16k_10s_logmel64() -> AudioPreset:
    return _preset(
        name="panns_cnn14_16k_10s_logmel64",
        sample_rate=16000,
        frontend=panns_frontend(
            sample_rate=16000,
            window_size=512,
            hop_size=160,
            mel_bins=64,
            fmin=50.0,
            fmax=8000.0,
        ),
    )


__all__ = [
    "panns_cnn14_16k_10s_logmel64",
    "panns_cnn14_32k_10s_logmel64",
    "panns_frontend",
]

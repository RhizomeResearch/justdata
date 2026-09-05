from __future__ import annotations

from justdata.acoustic._dcase import DCASE_CLASSES as DCASE_CLASSES
from justdata.acoustic._dcase import _dcase_label_transform
from justdata.acoustic.configs import (
    AudioPreprocessConfig,
    AudioPreset,
    FeatureNormConfig,
    FrontendConfig,
    LabelTransformConfig,
    LogCompressionConfig,
    MelConfig,
    STFTConfig,
    SegmentStrategyConfig,
)


PATCH_GRID = {
    "input_fdim": 128,
    "input_tdim": 998,
    "fstride": 10,
    "tstride": 10,
}

PATCHOUT = {
    "structured_frequency": 4,
    "structured_time": 40,
    "unstructured": 0,
}


def passt_frontend() -> FrontendConfig:
    return FrontendConfig(
        name="efficientat_logmel",
        stft=STFTConfig(
            sample_rate=32000,
            n_fft=1024,
            win_length=800,
            hop_length=320,
            window_periodic=False,
            center=True,
            power=2.0,
        ),
        mel=MelConfig(
            n_mels=128,
            f_min=0.0,
            f_max=15000.0,
            mel_scale="htk",
            mel_norm="none",
            filterbank_impl="efficientat_kaldi",
        ),
        log=LogCompressionConfig(kind="log", log_offset=1e-5),
        norm=FeatureNormConfig(kind="affine", scale=0.2, bias=0.9),
    )


def _preprocess() -> AudioPreprocessConfig:
    return AudioPreprocessConfig(
        target_sample_rate=32000,
        channel_strategy="mono_mean",
        normalize_waveform="none",
    )


def _metadata(*, source_duration: float | None = None) -> dict:
    metadata = {
        "preset_version": 2,
        "model_family": "passt",
        "frontend_contract": "passt-logmel-v1",
        "compatibility_status": "frontend_golden",
        "patch_grid": PATCH_GRID,
    }
    if source_duration is not None:
        metadata["source_duration"] = source_duration
    return metadata


def _audioset_label_transform() -> LabelTransformConfig:
    return LabelTransformConfig(mode="multi_hot", num_classes=527)


def _preset(
    *,
    name: str,
    input_duration: float,
    segment: SegmentStrategyConfig,
    label_transform: LabelTransformConfig,
    train_augment: dict | None = None,
    metadata: dict | None = None,
) -> AudioPreset:
    return AudioPreset(
        name=name,
        input_duration=input_duration,
        target_sample_rate=32000,
        preprocess=_preprocess(),
        segment=segment,
        frontend=passt_frontend(),
        layout="bcft",
        static_shape=(1, 128, round(input_duration * 100)),
        label_transform=label_transform,
        train_augment=train_augment or {},
        metadata=metadata or _metadata(),
    )


def passt_32k_10s_logmel128() -> AudioPreset:
    return _preset(
        name="passt_32k_10s_logmel128",
        input_duration=10.0,
        segment=SegmentStrategyConfig(
            clip_duration=10.0,
            train_mode="random_crop",
            eval_mode="center_crop",
            pad_mode="zero",
            pad_position="right",
        ),
        label_transform=_audioset_label_transform(),
        train_augment={"patchout": PATCHOUT},
    )


def dcase2025_task1_passt_32k_1s() -> AudioPreset:
    return _preset(
        name="dcase2025_task1_passt_32k_1s",
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
        train_augment={"patchout": PATCHOUT},
        metadata=_metadata(source_duration=1.0),
    )


def _dcase_10s(name: str, *, duration_policy: str, pad_mode: str) -> AudioPreset:
    metadata = _metadata(source_duration=1.0)
    metadata["model_input_duration"] = 10.0
    return _preset(
        name=name,
        input_duration=10.0,
        segment=SegmentStrategyConfig(
            clip_duration=10.0,
            train_mode="pad_or_crop",
            eval_mode="center_crop",
            pad_mode=pad_mode,
            pad_position="right",
            duration_policy=duration_policy,
        ),
        label_transform=_dcase_label_transform(),
        train_augment={"patchout": PATCHOUT},
        metadata=metadata,
    )


def dcase2025_task1_passt_32k_1s_zero_pad_to_10s() -> AudioPreset:
    return _dcase_10s(
        "dcase2025_task1_passt_32k_1s_zero_pad_to_10s",
        duration_policy="pad_to_model_duration",
        pad_mode="zero",
    )


def dcase2025_task1_passt_32k_1s_repeat_to_10s() -> AudioPreset:
    return _dcase_10s(
        "dcase2025_task1_passt_32k_1s_repeat_to_10s",
        duration_policy="repeat_pad_to_model_duration",
        pad_mode="repeat",
    )


__all__ = [
    "PATCH_GRID",
    "PATCHOUT",
    "dcase2025_task1_passt_32k_1s",
    "dcase2025_task1_passt_32k_1s_repeat_to_10s",
    "dcase2025_task1_passt_32k_1s_zero_pad_to_10s",
    "passt_32k_10s_logmel128",
    "passt_frontend",
]

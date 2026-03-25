from typing import Any, Dict

from justdata.acoustic.compat.ast import (
    ast_audioset_16k_10s_fbank128,
    ast_esc50_16k_5s_fbank128,
    ast_speechcommands_16k_1s_fbank128,
)
from justdata.acoustic.compat.ced import (
    ced_base_16k_logmel64,
    ced_mini_16k_logmel64,
    ced_small_16k_logmel64,
    ced_tiny_16k_logmel64,
    dcase2025_task1_ced_16k_1s,
)
from justdata.acoustic.compat.efficientat import (
    dcase2025_task1_dymn_32k_1s,
    dcase2025_task1_efficientat_32k_1s,
    dcase2025_task1_efficientat_32k_1s_repeat_to_10s,
    dcase2025_task1_efficientat_32k_1s_zero_pad_to_10s,
    dymn_32k_10s_logmel128,
    efficientat_32k_10s_logmel128,
)
from justdata.acoustic.compat.panns import (
    panns_cnn14_16k_10s_logmel64,
    panns_cnn14_32k_10s_logmel64,
)
from justdata.acoustic.compat.passt import (
    dcase2025_task1_passt_32k_1s,
    dcase2025_task1_passt_32k_1s_repeat_to_10s,
    dcase2025_task1_passt_32k_1s_zero_pad_to_10s,
    passt_32k_10s_logmel128,
)
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
from justdata.core.presets import (
    ResolvedPreset,
    get_dataset_presets as _get_dataset_presets,
    get_resolved_preset as _get_resolved_preset,
    merge_with_presets as _merge_with_presets,
    register_preset as _register_preset,
)


def register_preset(dataset: str, config: Dict[str, Any] | AudioPreset):
    if isinstance(config, AudioPreset):
        config = config.to_dict()
    _register_preset(dataset, config, modality="acoustic")


def get_dataset_presets(dataset: str) -> Dict[str, Any]:
    return _get_dataset_presets(dataset, modality="acoustic")


def get_resolved_preset(dataset: str) -> ResolvedPreset:
    return _get_resolved_preset(dataset, modality="acoustic")


def merge_with_presets(dataset: str, user_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return _merge_with_presets(dataset, user_kwargs, modality="acoustic")


def _segment(duration: float) -> SegmentStrategyConfig:
    return SegmentStrategyConfig(
        clip_duration=duration,
        train_mode="random_crop",
        eval_mode="center_crop",
        pad_mode="zero",
        pad_position="right",
    )


def _preprocess(sample_rate: int) -> AudioPreprocessConfig:
    return AudioPreprocessConfig(
        target_sample_rate=sample_rate,
        channel_strategy="mono_mean",
        normalize_waveform="none",
    )


def _label(mode: str, num_classes: int | None = None) -> LabelTransformConfig:
    return LabelTransformConfig(mode=mode, num_classes=num_classes)


def _logmel_frontend(
    *,
    sample_rate: int,
    n_fft: int,
    win_length: int,
    hop_length: int,
    n_mels: int,
    f_max: float,
) -> FrontendConfig:
    return FrontendConfig(
        name="logmel",
        stft=STFTConfig(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=True,
            power=2.0,
        ),
        mel=MelConfig(
            n_mels=n_mels,
            f_min=0.0,
            f_max=f_max,
            mel_scale="htk",
            mel_norm="none",
            filterbank_impl="kaldi_compatible",
        ),
        log=LogCompressionConfig(kind="log"),
    )


def _audio_default_16k_waveform() -> AudioPreset:
    return AudioPreset(
        name="audio_default_16k_waveform",
        input_duration=1.0,
        target_sample_rate=16000,
        preprocess=_preprocess(16000),
        segment=_segment(1.0),
        frontend=FrontendConfig(name="raw_waveform"),
        label_transform=_label("index"),
        layout="bt",
        metadata={
            "preset_version": 1,
            "model_family": "generic",
            "frontend_contract": "raw-waveform-16k-v1",
        },
    )


def _audio_default_32k_logmel(n_mels: int) -> AudioPreset:
    name = f"audio_default_32k_logmel{n_mels}"
    return AudioPreset(
        name=name,
        input_duration=1.0,
        target_sample_rate=32000,
        preprocess=_preprocess(32000),
        segment=_segment(1.0),
        frontend=_logmel_frontend(
            sample_rate=32000,
            n_fft=1024,
            win_length=800,
            hop_length=320,
            n_mels=n_mels,
            f_max=15000.0,
        ),
        label_transform=_label("index"),
        layout="btf",
        metadata={
            "preset_version": 1,
            "model_family": "generic",
            "frontend_contract": f"logmel-32k-{n_mels}-v1",
        },
    )


def _audioset_logmel(
    *,
    name: str,
    sample_rate: int,
    n_fft: int,
    win_length: int,
    hop_length: int,
    n_mels: int,
    f_max: float,
) -> AudioPreset:
    return AudioPreset(
        name=name,
        input_duration=10.0,
        target_sample_rate=sample_rate,
        preprocess=_preprocess(sample_rate),
        segment=_segment(10.0),
        frontend=_logmel_frontend(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_max=f_max,
        ),
        label_transform=_label("multi_hot", 527),
        layout="btf",
        metadata={
            "preset_version": 1,
            "model_family": "generic",
            "frontend_contract": f"audioset-logmel-{sample_rate}-{n_mels}-v1",
        },
    )


def _register_builtin_presets() -> None:
    for preset in (
        _audio_default_16k_waveform(),
        _audio_default_32k_logmel(64),
        _audio_default_32k_logmel(128),
        _audioset_logmel(
            name="audioset_32k_10s_logmel128",
            sample_rate=32000,
            n_fft=1024,
            win_length=800,
            hop_length=320,
            n_mels=128,
            f_max=15000.0,
        ),
        _audioset_logmel(
            name="audioset_32k_10s_logmel64",
            sample_rate=32000,
            n_fft=1024,
            win_length=800,
            hop_length=320,
            n_mels=64,
            f_max=15000.0,
        ),
        _audioset_logmel(
            name="audioset_16k_10s_logmel64",
            sample_rate=16000,
            n_fft=400,
            win_length=400,
            hop_length=160,
            n_mels=64,
            f_max=8000.0,
        ),
        efficientat_32k_10s_logmel128(),
        dymn_32k_10s_logmel128(),
        dcase2025_task1_efficientat_32k_1s(),
        dcase2025_task1_dymn_32k_1s(),
        dcase2025_task1_efficientat_32k_1s_zero_pad_to_10s(),
        dcase2025_task1_efficientat_32k_1s_repeat_to_10s(),
        passt_32k_10s_logmel128(),
        dcase2025_task1_passt_32k_1s(),
        dcase2025_task1_passt_32k_1s_zero_pad_to_10s(),
        dcase2025_task1_passt_32k_1s_repeat_to_10s(),
        ced_tiny_16k_logmel64(),
        ced_mini_16k_logmel64(),
        ced_small_16k_logmel64(),
        ced_base_16k_logmel64(),
        dcase2025_task1_ced_16k_1s(),
        panns_cnn14_32k_10s_logmel64(),
        panns_cnn14_16k_10s_logmel64(),
        ast_audioset_16k_10s_fbank128(),
        ast_esc50_16k_5s_fbank128(),
        ast_speechcommands_16k_1s_fbank128(),
    ):
        register_preset(preset.name, preset)


_register_builtin_presets()

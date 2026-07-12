from __future__ import annotations

import tensorflow as tf

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
from justdata.acoustic.frontends.log import compress_log
from justdata.acoustic.frontends.mel import mel_weight_matrix
from justdata.acoustic.frontends.stft import stft_power_spectrogram
from justdata.acoustic.normalization import apply_feature_normalization
from justdata.acoustic.registry import register_audio_frontend


DCASE_CLASSES = (
    "airport",
    "shopping_mall",
    "metro_station",
    "street_pedestrian",
    "public_square",
    "street_traffic",
    "tram",
    "bus",
    "metro",
    "park",
)


def efficientat_frontend() -> FrontendConfig:
    return FrontendConfig(
        name="efficientat_logmel",
        stft=STFTConfig(
            sample_rate=32000,
            n_fft=1024,
            win_length=800,
            hop_length=320,
            window="hann",
            window_periodic=False,
            center=True,
            power=2.0,
            normalized=False,
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


@register_audio_frontend("efficientat_logmel")
def efficientat_logmel(
    audio: tf.Tensor, config: FrontendConfig | dict
) -> tf.Tensor:
    config = FrontendConfig.from_dict(config)
    if config.stft is None or config.mel is None:
        raise ValueError("EfficientAT frontend requires STFT and mel configs")
    spectrogram = stft_power_spectrogram(
        audio,
        config.stft,
        preemphasis=0.97,
        frame_length=config.stft.n_fft,
    )
    mel = tf.einsum("tfc,fm->tmc", spectrogram, mel_weight_matrix(config))
    features = compress_log(mel, config.log)
    return apply_feature_normalization(features, config.norm)


def _preprocess() -> AudioPreprocessConfig:
    return AudioPreprocessConfig(
        target_sample_rate=32000,
        channel_strategy="mono_mean",
        normalize_waveform="none",
    )


def _metadata(model_family: str, *, source_duration: float | None = None) -> dict:
    metadata = {
        "preset_version": 2,
        "model_family": model_family,
        "frontend_contract": "efficientat-logmel-v1",
        "compatibility_status": "frontend_golden",
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
    model_family: str,
    input_duration: float,
    segment: SegmentStrategyConfig,
    label_transform: LabelTransformConfig,
    metadata: dict | None = None,
) -> AudioPreset:
    return AudioPreset(
        name=name,
        input_duration=input_duration,
        target_sample_rate=32000,
        preprocess=_preprocess(),
        segment=segment,
        frontend=efficientat_frontend(),
        layout="bcft",
        static_shape=(1, 128, round(input_duration * 100)),
        label_transform=label_transform,
        metadata=metadata or _metadata(model_family),
    )


def _default_segment(duration: float) -> SegmentStrategyConfig:
    return SegmentStrategyConfig(
        clip_duration=duration,
        train_mode="random_crop",
        eval_mode="center_crop",
        pad_mode="zero",
        pad_position="right",
    )


def efficientat_32k_10s_logmel128() -> AudioPreset:
    return _preset(
        name="efficientat_32k_10s_logmel128",
        model_family="efficientat",
        input_duration=10.0,
        segment=_default_segment(10.0),
        label_transform=_audioset_label_transform(),
    )


def dymn_32k_10s_logmel128() -> AudioPreset:
    return _preset(
        name="dymn_32k_10s_logmel128",
        model_family="dymn",
        input_duration=10.0,
        segment=_default_segment(10.0),
        label_transform=_audioset_label_transform(),
    )


def _dcase_keep_1s(name: str, model_family: str) -> AudioPreset:
    return _preset(
        name=name,
        model_family=model_family,
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
        metadata=_metadata(model_family, source_duration=1.0),
    )


def dcase2025_task1_efficientat_32k_1s() -> AudioPreset:
    return _dcase_keep_1s(
        "dcase2025_task1_efficientat_32k_1s",
        "efficientat",
    )


def dcase2025_task1_dymn_32k_1s() -> AudioPreset:
    return _dcase_keep_1s(
        "dcase2025_task1_dymn_32k_1s",
        "dymn",
    )


def _dcase_10s(name: str, *, duration_policy: str, pad_mode: str) -> AudioPreset:
    metadata = _metadata("efficientat", source_duration=1.0)
    metadata["model_input_duration"] = 10.0
    return _preset(
        name=name,
        model_family="efficientat",
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
        metadata=metadata,
    )


def dcase2025_task1_efficientat_32k_1s_zero_pad_to_10s() -> AudioPreset:
    return _dcase_10s(
        "dcase2025_task1_efficientat_32k_1s_zero_pad_to_10s",
        duration_policy="pad_to_model_duration",
        pad_mode="zero",
    )


def dcase2025_task1_efficientat_32k_1s_repeat_to_10s() -> AudioPreset:
    return _dcase_10s(
        "dcase2025_task1_efficientat_32k_1s_repeat_to_10s",
        duration_policy="repeat_pad_to_model_duration",
        pad_mode="repeat",
    )


__all__ = [
    "DCASE_CLASSES",
    "dcase2025_task1_dymn_32k_1s",
    "dcase2025_task1_efficientat_32k_1s",
    "dcase2025_task1_efficientat_32k_1s_repeat_to_10s",
    "dcase2025_task1_efficientat_32k_1s_zero_pad_to_10s",
    "dymn_32k_10s_logmel128",
    "efficientat_32k_10s_logmel128",
    "efficientat_frontend",
    "efficientat_logmel",
]

from dataclasses import replace

import pytest

from justdata.acoustic.configs import (
    AudioPreset,
    AudioPreprocessConfig,
    FeatureNormConfig,
    FrontendConfig,
    LabelTransformConfig,
    LogCompressionConfig,
    MelConfig,
    STFTConfig,
    SegmentStrategyConfig,
)


def make_preset(**overrides):
    sample_rate = overrides.pop("target_sample_rate", 16000)
    preset = AudioPreset(
        name="unit_logmel",
        input_duration=1.0,
        target_sample_rate=sample_rate,
        preprocess=AudioPreprocessConfig(
            target_sample_rate=sample_rate,
            resampler="identity",
        ),
        segment=SegmentStrategyConfig(
            clip_duration=1.0,
            train_mode="random_crop",
            eval_mode="center_crop",
            pad_mode="zero",
            pad_position="right",
        ),
        frontend=FrontendConfig(
            name="logmel",
            stft=STFTConfig(
                sample_rate=sample_rate,
                n_fft=400,
                win_length=400,
                hop_length=160,
            ),
            mel=MelConfig(n_mels=64),
            log=LogCompressionConfig(kind="log"),
            norm=FeatureNormConfig(kind="per_clip_mean_std", axes=("time", "frequency")),
        ),
        label_transform=LabelTransformConfig(
            mode="index",
            num_classes=2,
            class_names=("negative", "positive"),
        ),
        layout="btf",
        train_augment={"gain_db": 3.0},
    )
    return replace(preset, **overrides)


def test_config_roundtrip_json():
    preset = make_preset()

    assert AudioPreset.from_json(preset.to_json()) == preset


def test_config_hash_stable_for_same_config():
    assert make_preset().hash() == make_preset().hash()


def test_config_hash_changes_when_frontend_changes():
    preset = make_preset()
    raw = replace(
        preset,
        frontend=FrontendConfig(name="raw_waveform", norm=FeatureNormConfig()),
        layout="bt",
    )

    assert raw.hash() != preset.hash()


def test_audio_preset_requires_target_sample_rate():
    data = make_preset().to_dict()
    data.pop("target_sample_rate")

    with pytest.raises(ValueError, match="target_sample_rate"):
        AudioPreset.from_dict(data)


def test_invalid_layout_rejected():
    with pytest.raises(ValueError, match="layout"):
        make_preset(layout="tb")


def test_invalid_log_compression_rejected():
    with pytest.raises(ValueError, match="kind"):
        LogCompressionConfig(kind="ln")

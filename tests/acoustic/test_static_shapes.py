from dataclasses import replace

import pytest

from justdata.acoustic.configs import (
    AudioPreset,
    AudioPreprocessConfig,
    FrontendConfig,
    LabelTransformConfig,
    MelConfig,
    STFTConfig,
    SegmentStrategyConfig,
)
from justdata.acoustic.postprocessing import expected_audio_static_shape


def _preset(sample_rate: int, duration: float, hop_length: int):
    return AudioPreset(
        name="unit_static",
        input_duration=duration,
        target_sample_rate=sample_rate,
        preprocess=AudioPreprocessConfig(
            target_sample_rate=sample_rate,
            resampler="identity",
        ),
        segment=SegmentStrategyConfig(
            clip_duration=duration,
            train_mode="pad_or_crop",
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
                hop_length=hop_length,
            ),
            mel=MelConfig(n_mels=64),
        ),
        label_transform=LabelTransformConfig(mode="index", num_classes=10),
        layout="btf",
    )


def test_static_shape_10s_32k_hop320():
    assert expected_audio_static_shape(_preset(32000, 10.0, 320)) == (1001, 64)


def test_static_shape_1s_16k_hop160():
    assert expected_audio_static_shape(_preset(16000, 1.0, 160)) == (101, 64)


def test_static_shape_override_used():
    preset = replace(_preset(16000, 1.0, 160), static_shape=(100, 64))

    assert expected_audio_static_shape(preset) == (100, 64)


def test_fixed_multi_crop_static_shape_has_leading_view_axis():
    preset = _preset(16000, 1.0, 160)
    preset = replace(
        preset,
        segment=replace(preset.segment, eval_mode="multi_crop", num_views=3),
    )

    assert expected_audio_static_shape(preset) == (3, 101, 64)


def test_sliding_static_shape_fails_instead_of_claiming_fixed_shape():
    preset = _preset(16000, 1.0, 160)
    preset = replace(preset, segment=replace(preset.segment, eval_mode="sliding"))

    with pytest.raises(ValueError, match="sliding"):
        expected_audio_static_shape(preset)

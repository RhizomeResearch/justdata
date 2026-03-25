import json
from uuid import uuid4

from justdata.acoustic.configs import (
    AudioPreset,
    AudioPreprocessConfig,
    FrontendConfig,
    LabelTransformConfig,
    SegmentStrategyConfig,
)
from justdata.acoustic.presets import get_resolved_preset, register_preset
from justdata.vision.presets import get_resolved_preset as get_vision_resolved_preset


def make_audio_preset(name: str) -> AudioPreset:
    return AudioPreset(
        name=name,
        input_duration=1.0,
        target_sample_rate=16000,
        preprocess=AudioPreprocessConfig(
            target_sample_rate=16000, resampler="identity"
        ),
        segment=SegmentStrategyConfig(
            clip_duration=1.0,
            train_mode="pad_or_crop",
            eval_mode="center_crop",
            pad_mode="zero",
            pad_position="right",
        ),
        frontend=FrontendConfig(name="raw_waveform"),
        label_transform=LabelTransformConfig(mode="index", num_classes=10),
        layout="bt",
        train_augment={"none": True},
    )


def test_acoustic_resolved_preset_serializes_and_hashes():
    name = f"unit_audio_{uuid4().hex}"
    register_preset(name, make_audio_preset(name))

    resolved = get_resolved_preset(name)
    data = json.loads(resolved.to_json())

    assert data["target_sample_rate"] == 16000
    assert data["frontend"]["name"] == "raw_waveform"
    assert resolved.hash() == get_resolved_preset(name).hash()


def test_vision_resolved_preset_serializes_and_hashes():
    resolved = get_vision_resolved_preset("cifar10")
    data = json.loads(resolved.to_json())

    assert data["postproc_kwargs"]["image_size"] == 32
    assert data["model_input"]["output_key"] == "image"
    assert data["model_input"]["layout"] == "bchw"
    assert data["model_input"]["dtype"] == "float32"
    assert data["model_input"]["static_shape"] == [3, 32, 32]
    assert data["model_input"]["normalization"]["kind"] == "mean_std"
    assert len(resolved.hash()) == 16
    assert resolved.hash() == get_vision_resolved_preset("cifar10").hash()

from dataclasses import replace

import pytest

import justdata.acoustic  # noqa: F401
from justdata.acoustic.compat.efficientat import DCASE_CLASSES
from justdata.acoustic.configs import (
    AudioPreset,
    FrontendConfig,
    MelConfig,
)
from justdata.acoustic.postprocessing import expected_audio_static_shape
from justdata.acoustic.presets import get_dataset_presets, get_resolved_preset
from justdata.core.registry import get_pipeline


EXPECTED_HASHES = {
    "audio_default_16k_waveform": "7124f208319ef89d",
    "audio_default_32k_logmel64": "16f8c92045d8d983",
    "audio_default_32k_logmel128": "6f49846d0426bcd5",
    "audioset_32k_10s_logmel128": "311f9c6e0359c382",
    "audioset_32k_10s_logmel64": "314867d1ea97bd98",
    "audioset_16k_10s_logmel64": "6059a52472ca9f8a",
    "efficientat_32k_10s_logmel128": "aeddd33e73d978f3",
    "dymn_32k_10s_logmel128": "dfd3ccd0c5b170df",
    "dcase2025_task1_efficientat_32k_1s": "bfceba1dab08df4b",
    "dcase2025_task1_dymn_32k_1s": "de75f196a56ca4e3",
    "dcase2025_task1_efficientat_32k_1s_zero_pad_to_10s": "6e8524b54b68ab81",
    "dcase2025_task1_efficientat_32k_1s_repeat_to_10s": "8912f36e4387b933",
    "passt_32k_10s_logmel128": "9a448c55fbf418ca",
    "dcase2025_task1_passt_32k_1s": "8bde9d5035463a7b",
    "dcase2025_task1_passt_32k_1s_zero_pad_to_10s": "0fe864be59ceb999",
    "dcase2025_task1_passt_32k_1s_repeat_to_10s": "0830dfb12b7b9477",
    "ced_tiny_16k_logmel64": "9775450f5f24486f",
    "ced_mini_16k_logmel64": "b8739cc1aed6526e",
    "ced_small_16k_logmel64": "dcdb52ab2b21fc40",
    "ced_base_16k_logmel64": "6422c438897b7eea",
    "dcase2025_task1_ced_16k_1s": "139c16c9629fef50",
    "panns_cnn14_32k_10s_logmel64": "c98250d06de4965b",
    "panns_cnn14_16k_10s_logmel64": "ec3411b8dfbe97b6",
    "ast_audioset_16k_10s_fbank128": "ba89920c62c4eebd",
    "ast_esc50_16k_5s_fbank128": "4bba2ece4b2b2bcc",
    "ast_speechcommands_16k_1s_fbank128": "b62c85e524041c0a",
}

EXPECTED_SHAPES = {
    "audio_default_16k_waveform": (16000,),
    "audio_default_32k_logmel64": (101, 64),
    "audio_default_32k_logmel128": (101, 128),
    "audioset_32k_10s_logmel128": (1001, 128),
    "audioset_32k_10s_logmel64": (1001, 64),
    "audioset_16k_10s_logmel64": (1001, 64),
    "efficientat_32k_10s_logmel128": (1, 128, 1001),
    "dymn_32k_10s_logmel128": (1, 128, 1001),
    "dcase2025_task1_efficientat_32k_1s": (1, 128, 101),
    "dcase2025_task1_dymn_32k_1s": (1, 128, 101),
    "dcase2025_task1_efficientat_32k_1s_zero_pad_to_10s": (1, 128, 1001),
    "dcase2025_task1_efficientat_32k_1s_repeat_to_10s": (1, 128, 1001),
    "passt_32k_10s_logmel128": (1, 128, 1001),
    "dcase2025_task1_passt_32k_1s": (1, 128, 101),
    "dcase2025_task1_passt_32k_1s_zero_pad_to_10s": (1, 128, 1001),
    "dcase2025_task1_passt_32k_1s_repeat_to_10s": (1, 128, 1001),
    "ced_tiny_16k_logmel64": (998, 64),
    "ced_mini_16k_logmel64": (998, 64),
    "ced_small_16k_logmel64": (998, 64),
    "ced_base_16k_logmel64": (998, 64),
    "dcase2025_task1_ced_16k_1s": (98, 64),
    "panns_cnn14_32k_10s_logmel64": (1001, 64),
    "panns_cnn14_16k_10s_logmel64": (1001, 64),
    "ast_audioset_16k_10s_fbank128": (1024, 128),
    "ast_esc50_16k_5s_fbank128": (512, 128),
    "ast_speechcommands_16k_1s_fbank128": (128, 128),
}


def _preset(name: str) -> AudioPreset:
    return AudioPreset.from_dict(get_dataset_presets(name))


@pytest.mark.parametrize("name", sorted(EXPECTED_HASHES))
def test_preset_exists_serializes_hashes_and_has_declared_shape(name):
    raw = get_dataset_presets(name)
    preset = AudioPreset.from_dict(raw)

    assert preset.name == name
    assert AudioPreset.from_json(preset.to_json()) == preset
    assert preset.hash() == EXPECTED_HASHES[name]
    assert get_resolved_preset(name).hash() == EXPECTED_HASHES[name]
    assert set(raw["frontend"]) <= set(FrontendConfig.__dataclass_fields__)
    assert expected_audio_static_shape(preset) == EXPECTED_SHAPES[name]


@pytest.mark.parametrize("name", sorted(EXPECTED_HASHES))
def test_layout_and_label_transform_match_contract(name):
    preset = _preset(name)

    if name.startswith("dcase2025_task1"):
        assert preset.label_transform.mode == "index"
        assert preset.label_transform.num_classes == 10
        assert preset.label_transform.class_names == DCASE_CLASSES
    elif name.startswith(
        ("audioset", "efficientat", "dymn", "passt", "ced", "panns", "ast")
    ):
        assert preset.label_transform.mode == "multi_hot"
        if name.startswith("ast_esc50"):
            assert preset.label_transform.num_classes == 50
        elif name.startswith("ast_speechcommands"):
            assert preset.label_transform.num_classes == 35
        else:
            assert preset.label_transform.num_classes == 527
    else:
        assert preset.label_transform.mode == "index"

    if name.startswith(
        (
            "efficientat",
            "dymn",
            "dcase2025_task1_efficientat",
            "dcase2025_task1_dymn",
            "passt",
            "dcase2025_task1_passt",
        )
    ):
        assert preset.layout == "bcft"
    elif name.startswith(
        ("ced", "dcase2025_task1_ced", "panns", "audioset", "audio_default_32k", "ast")
    ):
        assert preset.layout == "btf"
    else:
        assert preset.layout == "bt"


def test_efficientat_preset_fields():
    preset = _preset("efficientat_32k_10s_logmel128")

    assert preset.input_duration == 10.0
    assert preset.target_sample_rate == 32000
    assert preset.preprocess.channel_strategy == "mono_mean"
    assert preset.preprocess.normalize_waveform == "none"
    assert preset.frontend.stft.n_fft == 1024
    assert preset.frontend.stft.win_length == 800
    assert preset.frontend.stft.hop_length == 320
    assert preset.frontend.mel.n_mels == 128
    assert preset.frontend.mel.f_max == 15000.0
    assert preset.frontend.mel.mel_scale == "htk"
    assert preset.frontend.mel.filterbank_impl == "kaldi_compatible"
    assert preset.frontend.norm.kind == "affine"
    assert preset.frontend.norm.mean == (-4.5,)
    assert preset.frontend.norm.std == (5.0,)


def test_dymn_preset_aliases_efficientat_frontend():
    efficientat = _preset("efficientat_32k_10s_logmel128")
    dymn = _preset("dymn_32k_10s_logmel128")

    assert dymn.frontend.to_dict() == efficientat.frontend.to_dict()
    assert dymn.preprocess.to_dict() == efficientat.preprocess.to_dict()


def test_passt_preset_win_length_800_not_1024():
    preset = _preset("passt_32k_10s_logmel128")

    assert preset.frontend.stft.n_fft == 1024
    assert preset.frontend.stft.win_length == 800
    assert preset.frontend.stft.hop_length == 320
    assert preset.frontend.mel.mel_scale == "slaney"
    assert preset.frontend.mel.mel_norm == "slaney"
    assert preset.metadata["patch_grid"]["input_tdim"] == 998


def test_passt_patchout_not_enabled_eval():
    preset = _preset("passt_32k_10s_logmel128")

    assert preset.train_augment["patchout"] == {
        "structured_frequency": 4,
        "structured_time": 40,
        "unstructured": 0,
    }
    assert "patchout" not in preset.eval_views


def test_ced_preset_uses_16k_64mel():
    for name in (
        "ced_tiny_16k_logmel64",
        "ced_mini_16k_logmel64",
        "ced_small_16k_logmel64",
        "ced_base_16k_logmel64",
    ):
        preset = _preset(name)
        assert preset.target_sample_rate == 16000
        assert preset.frontend.name == "kaldi_fbank"
        assert preset.frontend.mel.n_mels == 64
        assert preset.layout == "btf"


def test_dcase_presets_keep_1s_by_default():
    for name in (
        "dcase2025_task1_efficientat_32k_1s",
        "dcase2025_task1_dymn_32k_1s",
        "dcase2025_task1_passt_32k_1s",
        "dcase2025_task1_ced_16k_1s",
    ):
        preset = _preset(name)
        assert preset.input_duration == 1.0
        assert preset.segment.clip_duration == 1.0
        assert preset.segment.duration_policy == "keep_1s"


def test_dcase_10s_policy_explicit_in_name():
    explicit = {
        "dcase2025_task1_efficientat_32k_1s_zero_pad_to_10s": "pad_to_model_duration",
        "dcase2025_task1_efficientat_32k_1s_repeat_to_10s": "repeat_pad_to_model_duration",
        "dcase2025_task1_passt_32k_1s_zero_pad_to_10s": "pad_to_model_duration",
        "dcase2025_task1_passt_32k_1s_repeat_to_10s": "repeat_pad_to_model_duration",
    }

    for name, policy in explicit.items():
        preset = _preset(name)
        assert "to_10s" in name
        assert preset.input_duration == 10.0
        assert preset.segment.clip_duration == 10.0
        assert preset.segment.duration_policy == policy


def test_preset_hash_includes_frontend():
    preset = _preset("efficientat_32k_10s_logmel128")
    changed = replace(
        preset,
        frontend=replace(
            preset.frontend, mel=replace(preset.frontend.mel, f_max=14000.0)
        ),
    )

    assert changed.hash() != preset.hash()


def test_preset_hash_includes_duration_policy():
    preset = _preset("dcase2025_task1_efficientat_32k_1s")
    changed = replace(
        preset,
        segment=replace(preset.segment, duration_policy="pad_to_model_duration"),
    )

    assert changed.hash() != preset.hash()


def test_model_presets_register_with_core_resolver():
    pipeline = get_pipeline(
        dataset="audioset",
        preset="efficientat_32k_10s_logmel128",
        apply_presets=True,
    )

    assert pipeline.modality == "acoustic"
    assert pipeline.kwargs["name"] == "efficientat_32k_10s_logmel128"
    assert pipeline.kwargs["frontend"]["stft"]["win_length"] == 800


def test_panns_reference_presets_use_documented_window_hop_and_mel_bins():
    p32 = _preset("panns_cnn14_32k_10s_logmel64")
    p16 = _preset("panns_cnn14_16k_10s_logmel64")

    assert p32.frontend.stft.sample_rate == 32000
    assert p32.frontend.stft.n_fft == 1024
    assert p32.frontend.stft.hop_length == 320
    assert p32.frontend.mel == MelConfig(
        n_mels=64,
        f_min=50.0,
        f_max=14000.0,
        mel_scale="slaney",
        mel_norm="slaney",
        filterbank_impl="librosa",
    )
    assert p16.frontend.stft.sample_rate == 16000
    assert p16.frontend.stft.n_fft == 512
    assert p16.frontend.stft.hop_length == 160
    assert p16.frontend.mel.f_max == 8000.0


def test_ast_presets_match_official_recipe_constants():
    audioset = _preset("ast_audioset_16k_10s_fbank128")
    esc50 = _preset("ast_esc50_16k_5s_fbank128")
    speech = _preset("ast_speechcommands_16k_1s_fbank128")

    assert audioset.frontend.name == "ast_kaldi_fbank"
    assert audioset.target_sample_rate == 16000
    assert audioset.frontend.stft.n_fft == 512
    assert audioset.frontend.stft.win_length == 400
    assert audioset.frontend.stft.hop_length == 160
    assert audioset.metadata["ast"]["target_length"] == 1024
    assert audioset.metadata["ast"]["freqm"] == 48
    assert audioset.metadata["ast"]["timem"] == 192
    assert audioset.metadata["ast"]["mixup"] == 0.5
    assert audioset.metadata["ast"]["mean"] == -4.2677393
    assert audioset.metadata["ast"]["std"] == 4.5689974

    assert esc50.metadata["ast"]["target_length"] == 512
    assert esc50.metadata["ast"]["freqm"] == 24
    assert esc50.metadata["ast"]["timem"] == 96
    assert esc50.label_transform.num_classes == 50

    assert speech.metadata["ast"]["target_length"] == 128
    assert speech.metadata["ast"]["noise"] is True
    assert speech.label_transform.num_classes == 35

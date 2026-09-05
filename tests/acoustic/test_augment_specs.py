import copy
from dataclasses import dataclass

import numpy as np
import pytest
import tensorflow as tf

from justdata.acoustic.augment.batch import audio_mixup, normalize_batch_augment_specs
from justdata.acoustic.augment.spectrogram import (
    frequency_mask,
    normalize_spectrogram_augment_specs,
)
from justdata.acoustic.augment.waveform import (
    normalize_waveform_augment_specs,
    random_gain,
)


NORMALIZERS = (
    normalize_waveform_augment_specs,
    normalize_spectrogram_augment_specs,
    normalize_batch_augment_specs,
)


@pytest.mark.parametrize("normalize", NORMALIZERS)
def test_spec_forms_preserve_order_precedence_and_input(normalize):
    source = [
        None,
        "first",
        {"name": 7, "prob": 0.5},
        {
            "disabled": False,
            "missing": None,
            "enabled": True,
            "outer": {"name": "inner", "prob": 0.25},
            "configured": 0,
        },
        [["last"]],
    ]
    original = copy.deepcopy(source)

    assert normalize(source) == [
        {"name": "first"},
        {"name": "7", "prob": 0.5},
        {"name": "enabled"},
        {"name": "inner", "prob": 0.25},
        {"name": "configured", "config": 0},
        {"name": "last"},
    ]
    assert source == original


@pytest.mark.parametrize("normalize", NORMALIZERS)
def test_explicit_spec_is_copied(normalize):
    source = {"name": "first", "prob": 0.5}
    normalized = normalize(source)
    normalized[0]["prob"] = 1.0
    assert source["prob"] == 0.5


@pytest.mark.parametrize("name", ["cutmix", "mixstyle", "patchout"])
def test_waveform_does_not_resolve_other_stage_aliases(name):
    for source in (name, {"name": name}, {name: True}):
        assert normalize_waveform_augment_specs(source) == [{"name": name}]


def test_patchout_alias_applies_only_to_mapping_keys():
    normalize = normalize_spectrogram_augment_specs
    assert normalize("patchout") == [{"name": "patchout"}]
    assert normalize({"name": "patchout"}) == [{"name": "patchout"}]
    assert normalize({"patchout": True}) == [{"name": "passt_patchout"}]
    assert normalize({"patchout": {"name": "patchout"}}) == [{"name": "patchout"}]


@pytest.mark.parametrize(
    "alias,canonical", [("cutmix", "cutmix_spec"), ("mixstyle", "batch_mixstyle")]
)
def test_batch_aliases_and_embedded_name_normalization(alias, canonical):
    normalize = normalize_batch_augment_specs
    for source in (alias, {"name": alias}, {alias: True}):
        assert normalize(source) == [{"name": canonical}]
    first_pass = normalize({"outer": {"name": alias}})
    assert first_pass == [{"name": alias}]
    assert normalize(first_pass) == [{"name": canonical}]


@dataclass
class GateConfig:
    prob: float = 0.0


@pytest.mark.parametrize("config", [{"prob": 0.0}, GateConfig()])
@pytest.mark.parametrize("kind", ["waveform", "spectrogram", "batch"])
def test_augmentation_overrides_win_over_mapping_and_dataclass_configs(kind, config):
    x = tf.reshape(tf.range(32, dtype=tf.float32), [4, 8])
    labels = tf.constant([0, 1, 0, 1])
    original = copy.deepcopy(config)

    def apply(config, prob):
        kwargs = {"config": config, "prob": prob, "seed": [7, 13]}
        if kind == "waveform":
            return random_gain(x, min_db=6.0, max_db=6.0, **kwargs)
        if kind == "spectrogram":
            return frequency_mask(x, max_width=8, **kwargs)
        return audio_mixup(x, labels, num_classes=2, **kwargs)

    disabled = tf.nest.flatten(apply(config, None))[0]
    np.testing.assert_array_equal(disabled, x)
    actual = tf.nest.flatten(apply(config, 1.0))
    expected = tf.nest.flatten(apply(None, 1.0))
    for value, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(value, reference)
    assert not np.array_equal(actual[0], x)
    assert config == original


@pytest.mark.parametrize(
    "augment,args,error",
    [
        (random_gain, (tf.ones([4, 1]),), "RandomGainConfig"),
        (frequency_mask, (tf.ones([4, 2]),), "dataclass"),
        (audio_mixup, (tf.ones([2, 4]), tf.constant([0, 1])), "AudioMixupConfig"),
    ],
)
def test_invalid_augmentation_configs_keep_their_error(augment, args, error):
    with pytest.raises(TypeError, match=f"config must be a {error}, mapping, or None"):
        augment(*args, config=object())

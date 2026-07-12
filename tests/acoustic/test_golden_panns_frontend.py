from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

import justdata.acoustic  # noqa: F401
from justdata.acoustic.compat.panns import panns_frontend
from justdata.acoustic.frontends.log import logmel
from justdata.core.registry import get_pipeline


pytestmark = pytest.mark.golden

GOLDEN_ROOT = Path(__file__).parent / "golden"

CASES = [
    (
        32000,
        1024,
        320,
        14000.0,
        "panns_32k_1s.npz",
        "panns_32k_10s_zero_pad.npz",
        "panns_cnn14_32k_10s_logmel64",
    ),
    (
        16000,
        512,
        160,
        8000.0,
        "panns_16k_1s.npz",
        "panns_16k_10s_zero_pad.npz",
        "panns_cnn14_16k_10s_logmel64",
    ),
]


@pytest.mark.parametrize(
    (
        "sample_rate",
        "window_size",
        "hop_size",
        "f_max",
        "fixture_name",
        "_pipeline_fixture",
        "_preset",
    ),
    CASES,
)
def test_panns_frontend_matches_torchlibrosa_fixture(
    sample_rate,
    window_size,
    hop_size,
    f_max,
    fixture_name,
    _pipeline_fixture,
    _preset,
):
    waveform = np.load(GOLDEN_ROOT / "corpus.npz")[f"waveform_{sample_rate}"]
    expected = np.load(GOLDEN_ROOT / "panns" / fixture_name)["features"]
    frontend = panns_frontend(
        sample_rate=sample_rate,
        window_size=window_size,
        hop_size=hop_size,
        mel_bins=64,
        fmin=50.0,
        fmax=f_max,
    )
    actual = logmel(tf.constant(waveform[:, None]), frontend)

    np.testing.assert_allclose(
        np.squeeze(actual.numpy(), axis=-1),
        expected,
        rtol=1e-3,
        atol=0.06,
    )


@pytest.mark.parametrize(
    (
        "sample_rate",
        "_window_size",
        "_hop_size",
        "_f_max",
        "_frontend_fixture",
        "fixture_name",
        "preset",
    ),
    CASES,
)
def test_panns_eval_pipeline_matches_torchlibrosa_fixture(
    sample_rate,
    _window_size,
    _hop_size,
    _f_max,
    _frontend_fixture,
    fixture_name,
    preset,
):
    waveform = np.load(GOLDEN_ROOT / "corpus.npz")[f"waveform_{sample_rate}"]
    expected = np.load(GOLDEN_ROOT / "panns" / fixture_name)["features"]
    pipeline = get_pipeline(
        dataset="audioset",
        preset=preset,
        pipeline_name="acoustic/classification",
    )
    preprocess, augment, _late_augment, postprocess = pipeline.build(
        is_training=False
    )
    sample = preprocess(
        {
            "waveform": tf.constant(waveform),
            "sample_rate": tf.constant(sample_rate, tf.int32),
            "label": tf.constant([0], tf.int64),
        }
    )
    actual = postprocess(augment(sample))["features"].numpy()

    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=0.06)

from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

import justdata.acoustic  # noqa: F401
from justdata.acoustic.compat.ced import ced_frontend
from justdata.acoustic.frontends.log import logmel
from justdata.core.registry import get_pipeline


pytestmark = pytest.mark.golden

GOLDEN_ROOT = Path(__file__).parent / "golden"


def _inputs():
    waveform = np.load(GOLDEN_ROOT / "corpus.npz")["waveform_16000"]
    expected = np.load(GOLDEN_ROOT / "ced" / "ced_16k_1s.npz")["features"]
    return waveform, expected


def test_ced_frontend_matches_hugging_face_extractor_fixture():
    waveform, expected = _inputs()
    actual = logmel(tf.constant(waveform[:, None]), ced_frontend())

    np.testing.assert_allclose(
        np.squeeze(actual.numpy(), axis=-1).T,
        expected,
        rtol=5e-3,
        atol=0.25,
    )


def test_ced_eval_pipeline_matches_hugging_face_extractor_fixture():
    waveform, expected = _inputs()
    pipeline = get_pipeline(
        dataset="dcase2025_task1",
        preset="dcase2025_task1_ced_16k_1s",
        pipeline_name="acoustic/classification",
    )
    preprocess, augment, _late_augment, postprocess = pipeline.build(
        is_training=False
    )
    sample = preprocess(
        {
            "waveform": tf.constant(waveform),
            "sample_rate": tf.constant(16000, tf.int32),
            "label": tf.constant(0, tf.int64),
        }
    )
    actual = postprocess(augment(sample))["features"].numpy()

    np.testing.assert_allclose(actual.T, expected, rtol=5e-3, atol=0.25)

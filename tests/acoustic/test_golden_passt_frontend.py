from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

import justdata.acoustic  # noqa: F401
from justdata.acoustic.compat.efficientat import efficientat_logmel
from justdata.acoustic.compat.passt import passt_frontend
from justdata.core.registry import get_pipeline


pytestmark = pytest.mark.golden

GOLDEN_ROOT = Path(__file__).parent / "golden"


def _inputs():
    waveform = np.load(GOLDEN_ROOT / "corpus.npz")["waveform_32000"]
    expected = np.load(GOLDEN_ROOT / "passt" / "passt_32k_1s.npz")[
        "features"
    ]
    return waveform, expected


def test_passt_eval_frontend_matches_upstream_fixture_without_patchout():
    waveform, expected = _inputs()
    actual = efficientat_logmel(tf.constant(waveform[:, None]), passt_frontend())

    np.testing.assert_allclose(
        np.squeeze(actual.numpy(), axis=-1).T,
        expected,
        rtol=3e-3,
        atol=3e-3,
    )


def test_passt_eval_pipeline_matches_upstream_fixture_without_patchout():
    waveform, expected = _inputs()
    pipeline = get_pipeline(
        dataset="dcase2025_task1",
        preset="dcase2025_task1_passt_32k_1s",
        pipeline_name="acoustic/classification",
    )
    preprocess, augment, late_augment, postprocess = pipeline.build(is_training=False)
    sample = preprocess(
        {
            "waveform": tf.constant(waveform),
            "sample_rate": tf.constant(32000, tf.int32),
            "label": tf.constant(0, tf.int64),
        }
    )
    features = postprocess(augment(sample))["features"]
    batch = {"features": features[None, ...]}
    actual = late_augment(batch)["features"].numpy()[0, 0]

    np.testing.assert_allclose(actual, expected, rtol=3e-3, atol=3e-3)

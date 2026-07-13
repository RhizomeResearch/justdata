from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

import justdata.acoustic  # noqa: F401
from justdata.acoustic.compat.efficientat import (
    efficientat_frontend,
    efficientat_logmel,
)
from justdata.core.registry import get_pipeline


pytestmark = pytest.mark.golden

GOLDEN_ROOT = Path(__file__).parent / "golden"


def _inputs():
    waveform = np.load(GOLDEN_ROOT / "corpus.npz")["waveform_32000"]
    expected = np.load(GOLDEN_ROOT / "efficientat" / "efficientat_32k_1s.npz")[
        "features"
    ]
    return waveform, expected


def test_efficientat_eval_frontend_matches_upstream_fixture():
    waveform, expected = _inputs()
    actual = efficientat_logmel(tf.constant(waveform[:, None]), efficientat_frontend())

    np.testing.assert_allclose(
        np.squeeze(actual.numpy(), axis=-1).T,
        expected,
        rtol=3e-3,
        atol=3e-3,
    )


@pytest.mark.parametrize(
    "preset",
    ["dcase2025_task1_efficientat_32k_1s", "dcase2025_task1_dymn_32k_1s"],
)
def test_efficientat_and_dymn_eval_pipelines_match_upstream_fixture(preset):
    waveform, expected = _inputs()
    pipeline = get_pipeline(
        dataset="dcase2025_task1",
        preset=preset,
        pipeline_name="acoustic/classification",
    )
    preprocess, augment, _late_augment, postprocess = pipeline.build(is_training=False)
    sample = preprocess(
        {
            "waveform": tf.constant(waveform),
            "sample_rate": tf.constant(32000, tf.int32),
            "label": tf.constant(0, tf.int64),
        }
    )
    actual = postprocess(augment(sample))["features"].numpy()

    np.testing.assert_allclose(actual[0], expected, rtol=3e-3, atol=3e-3)

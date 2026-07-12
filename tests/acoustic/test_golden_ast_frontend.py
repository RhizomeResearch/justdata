from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

import justdata.acoustic  # noqa: F401
from justdata.acoustic.compat.ast import (
    ast_frontend,
    ast_normalize_fbank,
    ast_pad_or_crop_fbank,
)
from justdata.acoustic.frontends.ast import ast_kaldi_fbank
from justdata.core.registry import get_pipeline


pytestmark = pytest.mark.golden


GOLDEN_DIR = Path(__file__).parent / "golden" / "ast"


@pytest.mark.parametrize(
    ("fixture_name", "target_length", "mean", "std"),
    [
        ("ast_audioset_eval.npz", 1024, -4.2677393, 4.5689974),
        ("ast_esc50_eval.npz", 512, -6.6268077, 5.358466),
        ("ast_speechcommands_eval.npz", 128, -6.845978, 5.5654526),
    ],
)
def test_ast_eval_frontend_matches_torchaudio_ast_fixture(
    fixture_name,
    target_length,
    mean,
    std,
):
    fixture_path = GOLDEN_DIR / fixture_name
    fixture = np.load(fixture_path)
    waveform = tf.constant(fixture["waveform"], dtype=tf.float32)
    expected = fixture["features"]

    actual = ast_kaldi_fbank(waveform, ast_frontend())
    actual = ast_pad_or_crop_fbank(actual, target_length)
    actual = ast_normalize_fbank(actual, mean, std)

    # TensorFlow and Torch use different FFT/filterbank kernels; this keeps the
    # comparison tight while allowing sub-milliscale floating-point drift.
    np.testing.assert_allclose(
        np.squeeze(actual.numpy(), axis=-1), expected, rtol=5e-4, atol=5e-4
    )


@pytest.mark.parametrize(
    ("preset", "fixture_name"),
    [
        ("ast_audioset_16k_10s_fbank128", "ast_audioset_eval.npz"),
        ("ast_esc50_16k_5s_fbank128", "ast_esc50_eval.npz"),
        ("ast_speechcommands_16k_1s_fbank128", "ast_speechcommands_eval.npz"),
    ],
)
def test_ast_eval_pipeline_matches_torchaudio_ast_fixture(preset, fixture_name):
    fixture_path = GOLDEN_DIR / fixture_name
    fixture = np.load(fixture_path)
    pipeline = get_pipeline(
        dataset="audioset",
        preset=preset,
        pipeline_name="acoustic/ast_classification",
    )
    preprocess, _augment, _late_augment, postprocess = pipeline.build(is_training=False)

    sample = preprocess(
        {
            "waveform": tf.constant(fixture["waveform"], dtype=tf.float32),
            "sample_rate": tf.constant(16000, dtype=tf.int32),
            "label": tf.constant([0], dtype=tf.int64),
        }
    )
    result = postprocess(sample)

    np.testing.assert_allclose(
        result["features"].numpy(), fixture["features"], rtol=5e-4, atol=5e-4
    )

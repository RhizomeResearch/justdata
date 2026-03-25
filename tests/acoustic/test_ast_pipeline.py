import numpy as np
import tensorflow as tf

import justdata.acoustic  # noqa: F401
from justdata.acoustic.compat.ast import (
    AST_MIX_LABEL,
    AST_MIX_LAMBDA,
    AST_MIX_WAVEFORM,
)
from justdata.core.registry import get_pipeline, has_pipeline


def _waveform(num_samples=16000):
    return tf.linspace(-0.5, 0.5, num_samples)[:, tf.newaxis]


def test_ast_pipeline_registered():
    assert has_pipeline("acoustic/ast_classification")


def test_ast_eval_pipeline_outputs_static_fbank_and_multihot_label():
    pipeline = get_pipeline(
        dataset="audioset",
        preset="ast_audioset_16k_10s_fbank128",
        pipeline_name="acoustic/ast_classification",
    )
    preprocess, _augment, _late_augment, postprocess = pipeline.build(is_training=False)
    sample = {
        "waveform": _waveform(),
        "sample_rate": tf.constant(16000, dtype=tf.int32),
        "label": tf.constant([1, 3], dtype=tf.int64),
    }

    result = postprocess(preprocess(sample))

    assert result["features"].shape == (1024, 128)
    assert result["features"].dtype == tf.float32
    assert result["label"].shape == (527,)
    assert result["label"].numpy()[1] == 1.0
    assert result["label"].numpy()[3] == 1.0


def test_ast_eval_pipeline_is_deterministic():
    pipeline = get_pipeline(
        dataset="audioset",
        preset="ast_audioset_16k_10s_fbank128",
        pipeline_name="acoustic/ast_classification",
    )
    preprocess, _augment, _late_augment, postprocess = pipeline.build(is_training=False)
    sample = preprocess(
        {
            "waveform": _waveform(),
            "sample_rate": tf.constant(16000, dtype=tf.int32),
            "label": tf.constant([1], dtype=tf.int64),
        }
    )

    first = postprocess(sample)
    second = postprocess(sample)

    np.testing.assert_allclose(first["features"].numpy(), second["features"].numpy())


def test_ast_train_pipeline_mixes_explicit_partner_waveform_and_labels():
    pipeline = get_pipeline(
        dataset="speech_commands",
        preset="ast_speechcommands_16k_1s_fbank128",
        pipeline_name="acoustic/ast_classification",
    )
    preprocess, augment, _late_augment, postprocess = pipeline.build(is_training=True)
    sample = {
        "waveform": _waveform(),
        "sample_rate": tf.constant(16000, dtype=tf.int32),
        "label": tf.constant(1, dtype=tf.int64),
        AST_MIX_WAVEFORM: -_waveform(),
        AST_MIX_LABEL: tf.constant(3, dtype=tf.int64),
        AST_MIX_LAMBDA: tf.constant(0.25, dtype=tf.float32),
    }

    result = postprocess(augment(preprocess(sample), seed=[7, 0]))

    assert result["features"].shape == (128, 128)
    assert result["label"].shape == (35,)
    np.testing.assert_allclose(result["label"].numpy()[1], 0.25, rtol=1e-6)
    np.testing.assert_allclose(result["label"].numpy()[3], 0.75, rtol=1e-6)

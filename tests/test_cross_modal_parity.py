import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import tensorflow as tf

import justdata.acoustic  # noqa: F401
import justdata.vision  # noqa: F401
from justdata.acoustic.corruptions.datasets import create_audio_corruption_datasets
from justdata.acoustic.corruptions.registry import list_audio_corruptions
from justdata.acoustic.presets import get_resolved_preset as get_acoustic_preset
from justdata.acoustic.registry import (
    list_audio_frontends,
    list_audio_waveform_augments,
)
from justdata.core.loader import load_ds
from justdata.core.registry import get_pipeline, list_pipelines
from justdata.vision.augmentations import list_augment_strategies, list_crop_strategies
from justdata.vision.corruptions import list_corruptions
from justdata.vision.minic import create_minic_datasets
from justdata.vision.presets import get_resolved_preset as get_vision_preset


def _identity(sample, *args, **kwargs):
    return sample


def _vision_dataset(num_examples=3):
    images = np.arange(num_examples * 4 * 4 * 3, dtype=np.uint8).reshape(
        num_examples,
        4,
        4,
        3,
    )
    return tf.data.Dataset.from_tensor_slices(
        {
            "image": images,
            "label": np.arange(num_examples, dtype=np.int64),
            "metadata": {
                "device": np.asarray([f"cam-{i}" for i in range(num_examples)]),
                "quality": np.arange(num_examples, dtype=np.float32),
            },
        }
    ).apply(tf.data.experimental.assert_cardinality(num_examples))


def _acoustic_dataset(num_examples=3):
    waveform = np.ones((num_examples, 16, 1), dtype=np.float32)
    return tf.data.Dataset.from_tensor_slices(
        {
            "waveform": waveform,
            "sample_rate": np.full((num_examples,), 16000, dtype=np.int32),
            "label": np.arange(num_examples, dtype=np.int64),
            "metadata": {
                "device": np.asarray([f"mic-{i}" for i in range(num_examples)]),
                "quality": np.arange(num_examples, dtype=np.float32),
            },
        }
    ).apply(tf.data.experimental.assert_cardinality(num_examples))


def _string_label_dataset():
    return tf.data.Dataset.from_tensor_slices(
        {
            "x": np.arange(3, dtype=np.int32),
            "label": np.asarray(["alpha", "beta", "gamma"]),
            "metadata": {
                "source": np.asarray(["a", "b", "c"]),
                "quality": np.arange(3, dtype=np.float32),
            },
        }
    ).apply(tf.data.experimental.assert_cardinality(3))


def _load_with_mocked_fetch(raw_ds, **kwargs):
    with patch("justdata.core.loader.fetch_ds", return_value=raw_ds):
        return load_ds(
            dataset_names_arg="mock",
            splits_arg="validation",
            dataset_type="validation",
            batch_size=kwargs.pop("batch_size", 2),
            seed=0,
            preprocess_fn=_identity,
            augment_fn=_identity,
            late_augment_fn=_identity,
            postprocess_fn=_identity,
            cache_dataset=False,
            **kwargs,
        )


def _corruption_postprocess(sample, num_classes=None):
    del num_classes
    return sample


def test_both_modalities_have_hashable_presets():
    presets = [
        get_vision_preset("cifar10"),
        get_acoustic_preset("dcase2025_task1_efficientat_32k_1s"),
    ]

    for preset in presets:
        data = json.loads(preset.to_json())
        assert data
        assert isinstance(preset.hash(), str)
        assert len(preset.hash()) == 16
        assert preset.hash() == preset.hash()


def test_both_modalities_have_metadata_mode():
    for raw_ds in (_vision_dataset(), _acoustic_dataset()):
        ds, _n = _load_with_mocked_fetch(raw_ds, metadata_mode="numeric_only")
        batch = next(iter(ds))

        assert "metadata" in batch
        assert "quality" in batch["metadata"]
        assert "device" not in batch["metadata"]


class _IdentityPipeline:
    def __init__(self, kwargs):
        self.kwargs = kwargs

    def build(self, is_training):
        del is_training
        return _identity, _identity, _identity, _identity


@pytest.mark.parametrize(
    ("metadata_mode", "expected_keys"),
    [
        (None, {"quality"}),
        ("full", {"device", "quality"}),
        ("none", set()),
    ],
)
def test_acoustic_preset_metadata_mode_reaches_loader(
    metadata_mode,
    expected_keys,
):
    preset_pipeline = get_pipeline(dataset="speech_commands")
    assert preset_pipeline.kwargs["metadata_mode"] == "numeric_only"
    pipeline = _IdentityPipeline(preset_pipeline.kwargs)

    with patch(
        "justdata.core.loader.fetch_ds",
        return_value=_acoustic_dataset(),
    ):
        ds, _n = load_ds(
            dataset_names_arg="mock",
            splits_arg="validation",
            dataset_type="validation",
            batch_size=2,
            seed=0,
            pipeline=pipeline,
            cache_dataset=False,
            metadata_mode=metadata_mode,
        )

    batch = next(iter(ds))
    if metadata_mode == "none":
        assert "metadata" not in batch
    else:
        assert set(batch["metadata"]) == expected_keys


def test_both_modalities_have_deterministic_eval():
    vision_pipeline = get_pipeline(
        pipeline_name="vision/classification",
        preset="cifar10",
        modality="vision",
    )
    vision_preproc, _vision_aug, _vision_late, vision_postproc = vision_pipeline.build(
        is_training=False
    )
    vision_sample = {
        "image": tf.reshape(
            tf.cast(tf.range(40 * 48 * 3) % 256, tf.uint8), [40, 48, 3]
        ),
        "label": tf.constant(3, dtype=tf.int64),
    }

    first_vision = vision_postproc(vision_preproc(vision_sample), num_classes=10)
    second_vision = vision_postproc(vision_preproc(vision_sample), num_classes=10)
    np.testing.assert_allclose(
        first_vision["image"].numpy(), second_vision["image"].numpy()
    )

    acoustic_pipeline = get_pipeline(dataset="speech_commands")
    acoustic_preproc, _acoustic_aug, _acoustic_late, acoustic_postproc = (
        acoustic_pipeline.build(is_training=False)
    )
    acoustic_sample = {
        "waveform": tf.reshape(tf.linspace(-1.0, 1.0, 16000), [16000, 1]),
        "sample_rate": tf.constant(16000, dtype=tf.int32),
        "label": tf.constant(3, dtype=tf.int64),
    }

    first_acoustic = acoustic_postproc(
        acoustic_preproc(acoustic_sample), num_classes=35
    )
    second_acoustic = acoustic_postproc(
        acoustic_preproc(acoustic_sample), num_classes=35
    )
    np.testing.assert_allclose(
        first_acoustic["waveform"].numpy(),
        second_acoustic["waveform"].numpy(),
    )


def test_both_modalities_have_registry_list_functions():
    pipelines = list_pipelines()
    assert "vision/classification" in pipelines
    assert "acoustic/classification" in pipelines

    assert "random_resized" in list_crop_strategies()
    assert "rand_augment" in list_augment_strategies()
    assert "noise" in list_corruptions()

    assert "raw_waveform" in list_audio_frontends()
    assert "none" in list_audio_waveform_augments()
    assert "additive_white_noise" in list_audio_corruptions()


def test_both_modalities_have_corruption_dataset_api():
    with patch(
        "justdata.core.loader.fetch_ds",
        return_value=_vision_dataset(),
    ):
        vision_ds, vision_n = create_minic_datasets(
            corruption_types="noise",
            severity=1,
            dataset_names_arg="mock_vision",
            splits_arg="validation",
            dataset_type="validation",
            batch_size=2,
            seed=0,
            preprocess_fn=_identity,
            augment_fn=_identity,
            late_augment_fn=_identity,
            postprocess_fn=_corruption_postprocess,
            cache_dataset=False,
        )

    vision_batch = next(iter(vision_ds))
    assert vision_n == 2
    assert "padding_mask" in vision_batch
    assert vision_batch["metadata"]["corruption"].numpy()[0] == b"noise"

    with patch(
        "justdata.core.loader.fetch_ds",
        return_value=_acoustic_dataset(),
    ):
        acoustic_ds, acoustic_n = create_audio_corruption_datasets(
            corruption_types="identity",
            severity=1,
            base_dataset="mock_audio",
            preset=None,
            split="validation",
            batch_size=2,
            seed=0,
            preprocess_fn=_identity,
            augment_fn=_identity,
            late_augment_fn=_identity,
            postprocess_fn=_corruption_postprocess,
            cache_dataset=False,
        )

    acoustic_batch = next(iter(acoustic_ds))
    assert acoustic_n == 2
    assert "padding_mask" in acoustic_batch
    assert acoustic_batch["metadata"]["corruption"].numpy()[0] == b"identity"


@pytest.mark.parametrize("metadata_mode", ["numeric_only", "none"])
def test_minic_uses_shared_metadata_and_numpy_finalization(metadata_mode):
    with patch(
        "justdata.core.loader.fetch_ds",
        return_value=_vision_dataset(),
    ):
        iterator, n_batches = create_minic_datasets(
            corruption_types="noise",
            severity=1,
            dataset_names_arg="mock_vision",
            splits_arg="validation",
            dataset_type="validation",
            batch_size=2,
            seed=0,
            preprocess_fn=_identity,
            augment_fn=_identity,
            late_augment_fn=_identity,
            postprocess_fn=_corruption_postprocess,
            cache_dataset=False,
            metadata_mode=metadata_mode,
            as_numpy=True,
        )

    final = list(iterator)[-1]
    assert n_batches == 2
    assert isinstance(final["image"], np.ndarray)
    np.testing.assert_array_equal(final["padding_mask"], [True, False])
    if metadata_mode == "none":
        assert "metadata" not in final
    else:
        assert set(final["metadata"]) == {
            "corruption_identity_hash",
            "quality",
            "severity",
        }
        assert final["metadata"]["corruption_identity_hash"].dtype == np.int64


def test_both_modalities_preserve_padding_mask():
    for raw_ds in (_vision_dataset(), _acoustic_dataset()):
        ds, _n = _load_with_mocked_fetch(raw_ds, batch_size=2)
        final_batch = list(ds.take(2))[-1]

        np.testing.assert_array_equal(
            final_batch["padding_mask"].numpy(),
            [True, False],
        )


def test_both_modalities_support_as_numpy():
    for raw_ds, key in (
        (_vision_dataset(), "image"),
        (_acoustic_dataset(), "waveform"),
    ):
        iterator, _n = _load_with_mocked_fetch(raw_ds, as_numpy=True)
        batch = next(iter(iterator))

        assert isinstance(batch[key], np.ndarray)
        assert isinstance(batch["padding_mask"], np.ndarray)


def test_as_numpy_preserves_top_level_string_labels_with_numeric_metadata():
    iterator, _n = _load_with_mocked_fetch(
        _string_label_dataset(),
        as_numpy=True,
        metadata_mode="numeric_only",
    )
    first, final = list(iterator)

    assert isinstance(first["label"], np.ndarray)
    np.testing.assert_array_equal(first["label"], [b"alpha", b"beta"])
    np.testing.assert_array_equal(final["label"], [b"gamma", b""])
    assert "source" not in final["metadata"]
    assert np.issubdtype(final["metadata"]["quality"].dtype, np.number)


def test_docs_and_examples_cover_both_modalities():
    root = Path(__file__).parents[1]

    for path in (
        "docs/vision.md",
        "docs/acoustic.md",
        "docs/presets.md",
        "docs/golden_tests.md",
    ):
        assert (root / path).is_file()

    example_pairs = (
        (
            "examples/vision/cifar10_classification.py",
            "examples/acoustic/dcase2025_efficientat.py",
        ),
        (
            "examples/vision/hf_vision_dataset.py",
            "examples/acoustic/hf_audio_dataset.py",
        ),
        (
            "examples/vision/compute_cifar_channel_stats.py",
            "examples/acoustic/compute_dcase_source_stats.py",
        ),
        (
            "examples/vision/create_minic_corruptions.py",
            "examples/acoustic/create_audio_corruptions.py",
        ),
    )
    for vision_example, acoustic_example in example_pairs:
        assert (root / vision_example).is_file()
        assert (root / acoustic_example).is_file()

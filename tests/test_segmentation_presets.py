"""The graded policies work for both dense target formats."""

import numpy as np
import pytest
import tensorflow as tf

import justdata.vision  # noqa: F401
from justdata.core.registry import get_pipeline
from justdata.vision.augmentations.color import photometric_distortion
from justdata.vision.geometry import (
    DenseGeometryConfig,
    PanopticGeometryConfig,
    replay_dense_geometry,
    replay_panoptic_geometry,
    sample_dense_geometry,
    sample_panoptic_geometry,
)
from justdata.vision.presets import get_resolved_preset


PRESETS = {
    "segmentation_a3": ((0.8, 1.25), False),
    "segmentation_a2": ((0.5, 2.0), True),
    "segmentation_a1": ((0.1, 2.0), True),
}
HASHES = {
    "segmentation_a1": "4cff36cc2d2d91e1",
    "segmentation_a2": "163f97e3087e4026",
    "segmentation_a3": "cb747beb2f23cdec",
}


def _semantic():
    image = tf.reshape(tf.cast(tf.range(4 * 6 * 3) * 3, tf.uint8), [4, 6, 3])
    mask = tf.constant(
        [
            [0, 0, 1, 1, 2, 2],
            [0, 0, 1, 1, 2, 2],
            [0, 0, 1, 1, 2, 2],
            [255, 0, 1, 1, 2, 2],
        ],
        tf.int32,
    )
    return {"image": image, "mask": mask}


def _panoptic():
    sample = _semantic()
    return {
        "image": sample["image"],
        "panoptic_mask": tf.where(sample["mask"] == 2, 16777217, 3),
        "segments": {
            "segment_ids": tf.constant([3, 16777217], tf.int64),
            "category_ids": tf.constant([1, 2], tf.int32),
            "is_crowd": tf.constant([False, False]),
            "valid_mask": tf.constant([True, True]),
        },
    }


def test_fit_scale_has_explicit_anchor_and_preserves_large_ids():
    semantic = _semantic()
    panoptic = _panoptic()
    for factor, expected in ((1.0, [3, 4]), (2.0, [5, 8])):
        common = {
            "train_crop_size": 4,
            "train_resize_range": None,
            "train_scale_range": (factor, factor),
            "horizontal_flip_probability": 0,
            "patch_size": 1,
        }
        semantic_record = sample_dense_geometry(
            [4, 6],
            DenseGeometryConfig(class_values=(0, 1, 2), **common),
            is_training=True,
            seed=[1, 2],
        )
        panoptic_record = sample_panoptic_geometry(
            [4, 6],
            PanopticGeometryConfig(**common),
            void_value=0,
            is_training=True,
            seed=[1, 2],
        )
        np.testing.assert_array_equal(semantic_record["resized_size"], expected)
        np.testing.assert_array_equal(panoptic_record["resized_size"], expected)
        dense = replay_dense_geometry(semantic, semantic_record)
        pan = replay_panoptic_geometry(
            panoptic,
            panoptic_record,
            class_values=(1, 2),
            thing_class_values=(2,),
            void_value=0,
            max_segments=2,
        )
        assert dense["mask"].shape == pan["panoptic_mask"].shape == (4, 4)
        assert set(np.unique(pan["panoptic_mask"])) <= {0, 3, 16777217}
        for row in pan["segments"]["segment_ids"][pan["segments"]["valid_mask"]]:
            assert row.numpy() in (3, 16777217)


@pytest.mark.parametrize(
    "config",
    [
        {"train_resize_range": (1, 2), "train_scale_range": (0.5, 2)},
        {"train_resize_range": None, "train_scale_range": None},
        {"train_resize_range": None, "train_scale_range": (float("nan"), 2)},
        {"train_resize_range": None, "train_scale_range": (-1, 2)},
    ],
)
def test_scale_policy_rejects_ambiguous_or_invalid_ranges(config):
    with pytest.raises(ValueError):
        DenseGeometryConfig(class_values=(0,), **config)
    with pytest.raises(ValueError):
        PanopticGeometryConfig(**config)


def test_photometry_is_float_stateless_and_keeps_disabled_identity():
    image = tf.constant([[[1.25, 2.5, 3.75], [10.5, 20.25, 30.75]]])
    seed = tf.constant([23, 47])
    disabled = photometric_distortion(image, seed, probability=0)
    np.testing.assert_array_equal(disabled, image)
    kwargs = {
        "brightness": 0.125,
        "contrast": 0.5,
        "saturation": 0.5,
        "hue": 0.05,
        "probability": 1.0,
    }
    first = photometric_distortion(image, seed, **kwargs)
    traced = tf.function(photometric_distortion)(image, seed, **kwargs)
    np.testing.assert_array_equal(first, traced)
    assert first.dtype == tf.float32
    assert np.any(np.abs(first.numpy() - np.floor(first.numpy())) > 1e-5)
    assert not np.array_equal(first, photometric_distortion(image, [23, 48], **kwargs))


def test_brightness_and_contrast_match_float_reference():
    image = tf.constant([[[1.25, 2.5, 3.75], [10.5, 20.25, 30.75]]])
    seed = tf.constant([11, 13])
    seeds = tf.random.experimental.stateless_split(seed, 9)
    bright_factor = tf.random.stateless_uniform(
        [], seeds[4], minval=0.875, maxval=1.125
    )
    bright = photometric_distortion(
        image, seed, brightness=0.125, contrast=0, saturation=0, hue=0, probability=1
    )
    np.testing.assert_allclose(bright, image * bright_factor, rtol=1e-6)
    contrast_factor = tf.random.stateless_uniform([], seeds[5], minval=0.5, maxval=1.5)
    gray = np.sum(image.numpy() * [0.2989, 0.5870, 0.1140], axis=-1)
    mean = gray.mean()
    contrast = photometric_distortion(
        image, seed, brightness=0, contrast=0.5, saturation=0, hue=0, probability=1
    )
    expected = np.clip(mean + contrast_factor.numpy() * (image.numpy() - mean), 0, 255)
    np.testing.assert_allclose(contrast, expected, rtol=1e-6)


@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize("kind", ["semantic", "panoptic"])
def test_preset_resolves_and_preserves_dense_targets(preset, kind):
    scale_range, photo = PRESETS[preset]
    name = (
        "vision/segmentation" if kind == "semantic" else "vision/panoptic_segmentation"
    )
    options = (
        {
            "geometry_kwargs": {"class_values": (0, 1, 2), "ignore_value": 255},
            "postproc_kwargs": {"emit_semantic_targets": True},
        }
        if kind == "semantic"
        else {
            "panoptic_kwargs": {
                "class_values": (1, 2),
                "thing_class_values": (2,),
                "max_segments": 2,
            },
            "postproc_kwargs": {"emit_panoptic_targets": True},
        }
    )
    pipeline = get_pipeline(pipeline_name=name, preset=preset, overrides=options)
    resolved = pipeline.resolve_config(True)
    assert (
        tuple(resolved["configuration"]["geometry_kwargs"]["train_scale_range"])
        == scale_range
    )
    assert (resolved["configuration"]["photometric_kwargs"] is not None) == photo
    assert (
        resolved["stages"]["augment"]["geometry"]["resize_policy"]
        == "uniform_fit_scale"
    )
    assert resolved["model_input"]["static_shape"] == [3, 512, 512]
    pre, augment, _, post = pipeline.build(True)
    sample = pre(_semantic() if kind == "semantic" else _panoptic())
    first = post(augment(sample, tf.constant([5, 6])))
    second = post(augment(sample, tf.constant([5, 6])))
    np.testing.assert_array_equal(first["image"], second["image"])
    np.testing.assert_array_equal(first["pixel_valid_mask"], second["pixel_valid_mask"])
    assert first["image"].shape == (3, 512, 512)
    assert first["targets"]["masks"].shape[-2:] == (512, 512)
    if kind == "semantic":
        assert set(np.unique(first["mask"])) <= {0, 1, 2, 255}
        np.testing.assert_array_equal(first["mask"], second["mask"])
    else:
        assert set(np.unique(first["panoptic_mask"])) <= {0, 3, 16777217}
        np.testing.assert_array_equal(first["panoptic_mask"], second["panoptic_mask"])
    eval_record = pipeline.resolve_config(False)
    assert eval_record["stages"]["augment"]["active"] is False
    assert (
        eval_record["stages"]["postprocess"]["geometry"]["resize_policy"]
        == "long_side_cap"
    )
    eval_pre, _, _, eval_post = pipeline.build(False)
    eval_view = eval_post(eval_pre(_semantic() if kind == "semantic" else _panoptic()))
    assert eval_view["image"].shape == (3, 16, 16)
    assert not bool(eval_view["geometry"]["horizontal_flip"])


@pytest.mark.parametrize("name", PRESETS)
def test_preset_hash_and_override(name):
    preset = get_resolved_preset(name)
    assert preset.name == name
    assert preset.hash() == HASHES[name]
    pipeline = get_pipeline(
        pipeline_name="vision/segmentation",
        preset=name,
        overrides={
            "geometry_kwargs": {
                "class_values": (0, 1),
                "train_scale_range": None,
                "train_resize_range": (256, 512),
            }
        },
    )
    assert (
        pipeline.resolve_config(True)["stages"]["augment"]["geometry"]["resize_policy"]
        == "uniform_integer_short_side"
    )


def test_photometric_configuration_fails_before_source_access():
    base = {"geometry_kwargs": {"class_values": (0,)}}
    for extra in (
        {"photometric_kwargs": {"hue": float("inf")}},
        {"color_jitter_kwargs": {}, "photometric_kwargs": {}},
    ):
        pipeline = get_pipeline(
            pipeline_name="vision/segmentation",
            apply_presets=False,
            overrides=base | extra,
        )
        with pytest.raises(ValueError):
            pipeline.resolve_config(True)


def test_photometry_precedes_constant_padding():
    pipeline = get_pipeline(
        pipeline_name="vision/segmentation",
        preset="segmentation_a2",
        overrides={
            "geometry_kwargs": {
                "class_values": (0, 1, 2),
                "train_crop_size": 8,
                "train_scale_range": (0.1, 0.1),
                "patch_size": 1,
            },
            "postproc_kwargs": {"normalize_image": False},
        },
    )
    pre, augment, _, post = pipeline.build(True)
    view = post(augment(pre(_semantic()), tf.constant([1, 2])))
    image = np.moveaxis(view["image"].numpy(), 0, -1)
    support = view["source_valid_mask"].numpy()
    assert np.any(support)
    assert np.any(~support)
    np.testing.assert_array_equal(image[~support], 0)

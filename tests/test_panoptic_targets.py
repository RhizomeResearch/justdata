"""Panoptic instances, crowd validity, geometry, and loader transport."""

import numpy as np
import pytest
import tensorflow as tf

import justdata.vision  # noqa: F401
from justdata.core import (
    DataPipeline,
    InventoryRecord,
    InventorySource,
    admit_inventory,
    load_replay_epoch,
)
from justdata.vision.encodings import (
    decode_panoptic_rgb,
    panoptic_map_to_semantic,
    panoptic_map_to_targets,
    validate_panoptic_sample,
)
from justdata.vision.geometry import (
    PanopticGeometryConfig,
    restore_dense_scores,
    sample_panoptic_geometry,
)
from justdata.vision.panoptic_geometry import replay_panoptic_geometry


CONFIG = dict(
    class_values=(1, 3, 5, 11),
    thing_class_values=(11,),
    void_value=0,
    max_segments=6,
)


def _sample(*, crowd=False):
    mask = tf.constant(
        [
            [1, 3, 5, 11, 16777217],
            [1, 3, 5, 11, 16777217],
            [1, 3, 5, 0, 1],
            [1, 3, 5, 0, 1],
        ],
        tf.int64,
    )
    segments = {
        "segment_ids": tf.constant([16777217, 5, 11, 3, 1], tf.int64),
        "category_ids": tf.constant([11, 5, 11, 3, 1], tf.int32),
        "is_crowd": tf.constant([crowd, False, False, False, False]),
        "valid_mask": tf.constant([True] * 5),
    }
    return {
        "image": tf.zeros([4, 5, 3], tf.uint8),
        "panoptic_mask": mask,
        "segments": segments,
    }


def _pipeline(*, emit=True):
    return DataPipeline(
        "vision/panoptic_segmentation",
        geometry_kwargs={
            "train_crop_size": 4,
            "train_resize_range": (4, 4),
            "horizontal_flip_probability": 0.0,
            "eval_long_side": 5,
            "patch_size": 1,
        },
        panoptic_kwargs=CONFIG,
        postproc_kwargs={"normalize_image": False, "emit_panoptic_targets": emit},
    )


def test_rgb_decode_and_distinct_same_class_instances():
    rgb = tf.constant([[[11, 0, 0], [1, 0, 1]]], tf.uint8)
    np.testing.assert_array_equal(decode_panoptic_rgb(rgb), [[11, 65537]])
    sample = validate_panoptic_sample(_sample(), **CONFIG)
    record = sample_panoptic_geometry(
        [4, 5],
        PanopticGeometryConfig(eval_long_side=5, patch_size=1),
        void_value=0,
        is_training=False,
    )
    view = replay_panoptic_geometry(sample, record, **CONFIG)
    targets = panoptic_map_to_targets(
        view["panoptic_mask"],
        view["segments"],
        pixel_valid_mask=view["pixel_valid_mask"],
        **CONFIG,
    )
    assert int(targets["num_targets"]) == 5
    np.testing.assert_array_equal(targets["class_ids"][:5], [1, 3, 5, 11, 11])
    np.testing.assert_array_equal(targets["segment_ids"][3:5], [11, 16777217])
    assert np.logical_and(targets["masks"][3], targets["masks"][4]).sum() == 0
    assert int(view["segments"]["area"][0]) == 6
    np.testing.assert_array_equal(view["segments"]["bbox"][0], [0, 0, 5, 4])
    restored = restore_dense_scores(tf.ones([4, 5, 2]), record, from_model_input=True)
    assert restored.shape == (4, 5, 2)


def test_crop_removes_instance_and_preserves_remaining_ids():
    sample = validate_panoptic_sample(_sample(), **CONFIG)
    geometry = PanopticGeometryConfig(
        train_crop_size=4,
        train_resize_range=(4, 4),
        horizontal_flip_probability=0.0,
        patch_size=1,
    )
    record = sample_panoptic_geometry(
        [4, 5], geometry, void_value=0, is_training=True, seed=[4, 5]
    )
    record["crop_box"] = tf.constant([0, 0, 4, 4], tf.int32)
    view = replay_panoptic_geometry(sample, record, **CONFIG)
    np.testing.assert_array_equal(
        tf.boolean_mask(
            view["segments"]["segment_ids"], view["segments"]["valid_mask"]
        ),
        [1, 3, 5, 11],
    )
    assert int(view["segments"]["area"][3]) == 2
    np.testing.assert_array_equal(view["segments"]["bbox"][3], [3, 0, 1, 2])


def test_crowd_excluded_from_panoptic_supervision_but_semantic_projection_keeps_it():
    sample = validate_panoptic_sample(_sample(crowd=True), **CONFIG)
    geometry = PanopticGeometryConfig(eval_long_side=5, patch_size=1)
    record = sample_panoptic_geometry([4, 5], geometry, void_value=0, is_training=False)
    view = replay_panoptic_geometry(sample, record, **CONFIG)
    assert not view["pixel_valid_mask"].numpy()[:, 4].all()
    targets = panoptic_map_to_targets(
        view["panoptic_mask"],
        view["segments"],
        pixel_valid_mask=view["pixel_valid_mask"],
        **CONFIG,
    )
    assert int(targets["num_targets"]) == 4
    assert 16777217 not in targets["segment_ids"].numpy()
    projected = panoptic_map_to_semantic(
        sample["panoptic_mask"],
        sample["segments"],
        semantic_values=(0, 1, 2, 0),
        **CONFIG,
    )
    assert int(projected[0, 4]) == 0
    excluded = panoptic_map_to_semantic(
        sample["panoptic_mask"],
        sample["segments"],
        semantic_values=(0, 1, 2, 0),
        include_crowd=False,
        **CONFIG,
    )
    assert int(excluded[0, 4]) == 255


def test_direct_targets_enforce_crowd_and_void_even_with_full_input_validity():
    sample = validate_panoptic_sample(_sample(crowd=True), **CONFIG)
    targets = panoptic_map_to_targets(
        sample["panoptic_mask"],
        sample["segments"],
        pixel_valid_mask=tf.ones([4, 5], tf.bool),
        **CONFIG,
    )
    assert int(targets["num_targets"]) == 4
    assert not bool(targets["pixel_valid_mask"][0, 4])
    assert not bool(targets["pixel_valid_mask"][2, 3])
    assert 16777217 not in targets["segment_ids"].numpy()


@pytest.mark.parametrize("mode", ["all_void", "all_crowd", "stuff_only"])
def test_supervision_state(mode):
    original = _sample(crowd=True)
    if mode == "all_void":
        original["panoptic_mask"] = tf.zeros([4, 5], tf.int64)
        original["segments"] = {
            "segment_ids": tf.constant([], tf.int64),
            "category_ids": tf.constant([], tf.int32),
            "is_crowd": tf.constant([], tf.bool),
            "valid_mask": tf.constant([], tf.bool),
        }
    elif mode == "all_crowd":
        original["panoptic_mask"] = tf.fill([4, 5], tf.constant(11, tf.int64))
        original["segments"] = {
            "segment_ids": tf.constant([11], tf.int64),
            "category_ids": tf.constant([11], tf.int32),
            "is_crowd": tf.constant([True]),
            "valid_mask": tf.constant([True]),
        }
    else:
        original["panoptic_mask"] = tf.fill([4, 5], tf.constant(3, tf.int64))
        original["segments"] = {
            "segment_ids": tf.constant([3], tf.int64),
            "category_ids": tf.constant([3], tf.int32),
            "is_crowd": tf.constant([False]),
            "valid_mask": tf.constant([True]),
        }
    pre, _, _, post = _pipeline().build(False)
    result = post(pre(original))
    targets = result["targets"]
    expected = int(mode == "stuff_only")
    assert int(targets["num_targets"]) == expected
    assert bool(targets["supervision_valid"]) == bool(expected)


@pytest.mark.parametrize("problem", ["missing", "duplicate", "category", "capacity"])
def test_invalid_annotations_are_rejected(problem):
    sample = _sample()
    segments = dict(sample["segments"])
    if problem == "missing":
        sample["panoptic_mask"] = tf.tensor_scatter_nd_update(
            sample["panoptic_mask"], [[0, 0]], [99]
        )
    elif problem == "duplicate":
        segments["segment_ids"] = tf.constant([11, 5, 11, 3, 1], tf.int64)
    elif problem == "category":
        segments["category_ids"] = tf.constant([12, 5, 11, 3, 1], tf.int32)
    else:
        segments["segment_ids"] = tf.concat([segments["segment_ids"], [25, 26]], 0)
        segments["category_ids"] = tf.concat([segments["category_ids"], [11, 11]], 0)
        segments["is_crowd"] = tf.concat([segments["is_crowd"], [False, False]], 0)
        segments["valid_mask"] = tf.concat([segments["valid_mask"], [True, True]], 0)
    sample["segments"] = segments
    with pytest.raises((ValueError, tf.errors.InvalidArgumentError)):
        validate_panoptic_sample(sample, **CONFIG)


def test_replay_epoch_transports_targets_and_partial_batch(tmp_path):
    def reader(_payloads, metadata):
        return _sample(crowd=bool(metadata["index"] % 2))

    admitted = admit_inventory(
        {
            "local": InventorySource(
                {
                    "train": [
                        InventoryRecord(f"sample-{index}", metadata={"index": index})
                        for index in range(3)
                    ]
                },
                reader,
            )
        },
        {"local": ["train"]},
        inventory_id="panoptic-fixture",
        snapshot_dir=tmp_path / "snapshot",
        output_signature={
            "image": tf.TensorSpec([4, 5, 3], tf.uint8),
            "panoptic_mask": tf.TensorSpec([4, 5], tf.int64),
            "segments": {
                "segment_ids": tf.TensorSpec([5], tf.int64),
                "category_ids": tf.TensorSpec([5], tf.int32),
                "is_crowd": tf.TensorSpec([5], tf.bool),
                "valid_mask": tf.TensorSpec([5], tf.bool),
            },
        },
    )
    epoch = load_replay_epoch(
        admitted,
        "train",
        2,
        19,
        pipeline=_pipeline(),
        epoch=3,
        callbacks_are_stateless=True,
        as_numpy=True,
        shuffle_buffer=1,
    )
    rows = list(epoch.batches)
    assert len(rows) == 2
    assert rows[-1]["padding_mask"].tolist() == [True, False]
    assert rows[-1]["targets"]["num_targets"][1] == 0
    assert not rows[-1]["targets"]["target_valid_mask"][1].any()
    assert epoch.config is not None and epoch.state is not None

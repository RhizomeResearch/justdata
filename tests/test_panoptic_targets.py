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
    DenseGeometryConfig,
    PanopticGeometryConfig,
    replay_dense_geometry,
    replay_panoptic_geometry,
    restore_dense_scores,
    sample_dense_geometry,
    sample_panoptic_geometry,
)


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


def _pipeline(*, emit=True, **postprocess):
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
        postproc_kwargs={"normalize_image": False, "emit_panoptic_targets": emit}
        | postprocess,
    )


CROWDED = dict(class_values=(1, 11, 12), thing_class_values=(11, 12), void_value=0)


def _crowded():
    # Stuff 1 around category-11 things 101-104 of 10, 9, 8 and 7 pixels and a
    # one-pixel category-12 thing 201.
    mask = np.ones((6, 10), np.int64)
    for row, segment in enumerate((101, 102, 103, 104)):
        mask[row, : 10 - row] = segment
    mask[5, 9] = 201
    segments = {
        "segment_ids": tf.constant([1, 101, 102, 103, 104, 201], tf.int64),
        "category_ids": tf.constant([1, 11, 11, 11, 11, 12], tf.int32),
        "is_crowd": tf.zeros([6], tf.bool),
        "valid_mask": tf.ones([6], tf.bool),
    }
    return tf.constant(mask), segments


def _crowded_targets(max_segments, overflow_policy):
    mask, segments = _crowded()
    return panoptic_map_to_targets(
        mask,
        segments,
        max_segments=max_segments,
        pixel_valid_mask=tf.ones([6, 10], tf.bool),
        overflow_policy=overflow_policy,
        **CROWDED,
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


def test_geometry_records_reject_the_wrong_label_contract():
    sample = _sample()
    panoptic_record = sample_panoptic_geometry(
        [4, 5],
        PanopticGeometryConfig(eval_long_side=5, patch_size=1),
        void_value=0,
        is_training=False,
    )
    semantic_record = sample_dense_geometry(
        [4, 5],
        DenseGeometryConfig(class_values=(1,), eval_long_side=5, patch_size=1),
        is_training=False,
    )
    with pytest.raises(
        tf.errors.InvalidArgumentError, match="semantic geometry requires"
    ):
        replay_dense_geometry({"image": sample["image"]}, panoptic_record)
    with pytest.raises(
        tf.errors.InvalidArgumentError, match="panoptic geometry must not"
    ):
        replay_panoptic_geometry(sample, semantic_record, **CONFIG)
    with pytest.raises(tf.errors.InvalidArgumentError, match="void value mismatch"):
        replay_panoptic_geometry(
            sample,
            panoptic_record | {"mask_fill_value": tf.constant(1, tf.int64)},
            **CONFIG,
        )
    with pytest.raises(
        tf.errors.InvalidArgumentError, match="unsupported geometry version"
    ):
        replay_panoptic_geometry(
            sample,
            panoptic_record | {"version": tf.constant(3, tf.int32)},
            **CONFIG,
        )


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


def test_panoptic_replay_carries_categorical_fields_and_commutes_with_projection():
    sample = validate_panoptic_sample(_sample(), **CONFIG)
    projection = dict(semantic_values=(0, 1, 2, 0), **CONFIG)
    semantic = panoptic_map_to_semantic(
        sample["panoptic_mask"], sample["segments"], **projection
    )
    spatial = dict(train_crop_size=4, train_resize_range=(6, 6), patch_size=1)
    for seed in range(4):
        panoptic_record = sample_panoptic_geometry(
            [4, 5],
            PanopticGeometryConfig(**spatial),
            void_value=0,
            is_training=True,
            seed=[seed, 1],
        )
        view = replay_panoptic_geometry(
            sample | {"semantic": semantic},
            panoptic_record,
            categorical_fields={"semantic": 255},
            **CONFIG,
        )
        dense = replay_dense_geometry(
            {"image": sample["image"], "mask": semantic},
            sample_dense_geometry(
                [4, 5],
                DenseGeometryConfig(class_values=(0, 1, 2), **spatial),
                is_training=True,
                seed=[seed, 1],
            ),
        )
        np.testing.assert_array_equal(view["semantic"], dense["mask"])
        np.testing.assert_array_equal(
            panoptic_map_to_semantic(
                view["panoptic_mask"], view["segments"], **projection
            ),
            dense["mask"],
        )


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


@pytest.mark.parametrize("policy", ["exclude_supervision", "merge_by_category"])
def test_overflow_policies_leave_views_within_capacity_unchanged(policy):
    expected = _crowded_targets(6, "error")
    result = _crowded_targets(6, policy)
    assert result.keys() == expected.keys()
    for key in expected:
        np.testing.assert_array_equal(result[key], expected[key])
    assert int(result["num_overflow_things"]) == 0
    with pytest.raises(
        tf.errors.InvalidArgumentError, match="panoptic target capacity exceeded"
    ):
        _crowded_targets(5, "error")


def test_excluding_overflow_keeps_the_largest_things_and_drops_the_rest_from_validity():
    targets = _crowded_targets(5, "exclude_supervision")
    assert int(targets["num_targets"]) == 5
    np.testing.assert_array_equal(targets["segment_ids"], [0, 101, 102, 103, 104])
    assert int(targets["num_overflow_things"]) == 1
    valid = targets["pixel_valid_mask"].numpy()
    assert not valid[5, 9]
    assert valid.sum() == 59


def test_merging_overflow_keeps_the_most_instances_and_every_class_mask():
    mask, segments = _crowded()
    targets = _crowded_targets(5, "merge_by_category")
    # Keeping 101 and 102 leaves room for one target per remaining category;
    # 201 is alone in its category, so it keeps that target as an instance.
    assert int(targets["num_targets"]) == 5
    np.testing.assert_array_equal(targets["class_ids"], [1, 11, 11, 12, 11])
    np.testing.assert_array_equal(targets["segment_ids"], [0, 101, 102, 201, 0])
    np.testing.assert_array_equal(targets["is_thing"], [False, True, True, True, False])
    assert int(targets["num_overflow_things"]) == 2
    np.testing.assert_array_equal(targets["masks"][4], np.isin(mask, [103, 104]))
    assert targets["pixel_valid_mask"].numpy().all()
    semantic = panoptic_map_to_semantic(
        mask, segments, semantic_values=(1, 11, 12), max_segments=6, **CROWDED
    )
    for value in (1, 11, 12):
        of_class = targets["class_ids"].numpy() == value
        np.testing.assert_array_equal(
            targets["masks"].numpy()[of_class].any(axis=0), semantic.numpy() == value
        )
    with pytest.raises(
        tf.errors.InvalidArgumentError, match="panoptic target capacity exceeded"
    ):
        _crowded_targets(2, "merge_by_category")


def test_target_capacity_ignores_crowd_rows_and_ties_keep_the_lower_segment_id():
    capacity = CONFIG | {"max_segments": 4}
    crowd = validate_panoptic_sample(_sample(crowd=True), **CONFIG)
    targets = panoptic_map_to_targets(
        crowd["panoptic_mask"],
        crowd["segments"],
        pixel_valid_mask=tf.ones([4, 5], tf.bool),
        **capacity,
    )
    assert int(targets["num_targets"]) == 4
    # Things 11 and 16777217 both cover two pixels.
    sample = validate_panoptic_sample(_sample(), **CONFIG)
    targets = panoptic_map_to_targets(
        sample["panoptic_mask"],
        sample["segments"],
        pixel_valid_mask=tf.ones([4, 5], tf.bool),
        overflow_policy="exclude_supervision",
        **capacity,
    )
    np.testing.assert_array_equal(targets["segment_ids"], [0, 0, 0, 11])
    assert not targets["pixel_valid_mask"].numpy()[:2, 4].any()
    with pytest.raises(ValueError, match="overflow_policy"):
        panoptic_map_to_targets(
            sample["panoptic_mask"],
            sample["segments"],
            pixel_valid_mask=tf.ones([4, 5], tf.bool),
            overflow_policy="drop",
            **capacity,
        )


def test_pipeline_bounds_targets_below_the_table_capacity():
    default = _pipeline().resolve_config(False)["stages"]["postprocess"]["target_set"]
    assert (default["version"], default["capacity"]) == (2, 6)
    assert default["overflow_policy"] == "error"
    pipeline = _pipeline(max_targets=4, overflow_policy="exclude_supervision")
    target_set = pipeline.resolve_config(False)["stages"]["postprocess"]["target_set"]
    assert (target_set["capacity"], target_set["overflow_policy"]) == (
        4,
        "exclude_supervision",
    )
    pre, _, _, post = pipeline.build(False)
    targets = post(pre(_sample()))["targets"]
    assert targets["masks"].shape[0] == 4
    assert int(targets["num_targets"]) == 4
    assert int(targets["num_overflow_things"]) == 1
    for options, match in [
        ({"max_targets": 0}, "max_targets"),
        ({"overflow_policy": "drop"}, "overflow_policy"),
    ]:
        with pytest.raises(ValueError, match=match):
            _pipeline(**options).resolve_config(False)


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

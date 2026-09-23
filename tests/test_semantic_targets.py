"""Independent fixtures for semantic class masks and their pipeline transport."""

import numpy as np
import pytest
import tensorflow as tf

import justdata.vision  # noqa: F401
from justdata.core import (
    DataPipeline,
    InventoryRecord,
    InventorySource,
    ReplayError,
    admit_inventory,
    load_replay_epoch,
    open_inventory,
)
from justdata.vision.encodings import semantic_map_to_targets, with_semantic_targets
from justdata.vision.geometry import replay_dense_geometry


def _pipeline(*, values=(0, 1, 2), emit=True):
    return DataPipeline(
        "vision/segmentation",
        geometry_kwargs={
            "class_values": values,
            "ignore_value": 255,
            "train_crop_size": 4,
            "train_resize_range": (2, 2),
            "horizontal_flip_probability": 0.5,
            "eval_long_side": 4,
            "patch_size": 2,
        },
        postproc_kwargs={"normalize_image": False, "emit_semantic_targets": emit},
    )


def _inventory(tmp_path):
    masks = (
        [[0, 255, 2, 0], [2, 1, 0, 1]],
        [[255, 255, 255, 255], [255, 255, 255, 255]],
        [[0, 0, 0, 0], [0, 0, 0, 0]],
    )

    def reader(_payloads, metadata):
        index = metadata["index"]
        return {
            "image": tf.fill([2, 4, 3], tf.cast(index * 30, tf.uint8)),
            "mask": tf.constant(masks[index], tf.int32),
            "annotation_valid_mask": tf.constant(
                [[True, True, True, True], [True, True, index != 0, True]]
            ),
            "metadata": {"instance_ref": tf.constant(index + 10, tf.int32)},
        }

    return admit_inventory(
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
        inventory_id="semantic-target-fixture",
        snapshot_dir=tmp_path / "snapshot",
        output_signature={
            "image": tf.TensorSpec([2, 4, 3], tf.uint8),
            "mask": tf.TensorSpec([2, 4], tf.int32),
            "annotation_valid_mask": tf.TensorSpec([2, 4], tf.bool),
            "metadata": {"instance_ref": tf.TensorSpec([], tf.int32)},
        },
    )


def _epoch(admitted, *, state=None, pipeline=None, metadata_mode="full"):
    return load_replay_epoch(
        admitted,
        "train",
        2,
        19,
        pipeline=pipeline or _pipeline(),
        epoch=3,
        state=state,
        callbacks_are_stateless=True,
        as_numpy=True,
        metadata_mode=metadata_mode,
        shuffle_buffer=1,
    )


def test_static_slots_round_trip_noncontiguous_classes_and_disconnected_pixels():
    labels = np.array(
        [[20, 255, 10, 20], [10, 20, 255, 10], [20, 20, 20, 20]],
        np.int32,
    )
    available = np.ones(labels.shape, bool)
    available[2, 2] = False
    result = semantic_map_to_targets(
        labels,
        class_values=(20, 10, 30),
        ignore_value=255,
        pixel_valid_mask=available,
    )
    np.testing.assert_array_equal(result["class_ids"], [20, 10, 30])
    np.testing.assert_array_equal(result["class_indices"], [0, 1, 2])
    np.testing.assert_array_equal(result["target_valid_mask"], [True, True, False])
    assert int(result["num_targets"]) == 2
    assert bool(result["supervision_valid"])
    masks = result["masks"].numpy()
    valid = available & (labels != 255)
    np.testing.assert_array_equal(result["pixel_valid_mask"], valid)
    np.testing.assert_array_equal(masks[0], (labels == 20) & valid)
    np.testing.assert_array_equal(masks[1], (labels == 10) & valid)
    assert masks[0, 0, 0] and masks[0, 2, 3]
    assert not masks[2].any()
    reconstructed = np.full(labels.shape, 255, np.int32)
    for slot, class_id in enumerate(result["class_ids"].numpy()):
        reconstructed[masks[slot]] = class_id
    np.testing.assert_array_equal(reconstructed[valid], labels[valid])
    assert not masks[:, ~valid].any()


@pytest.mark.parametrize("example_valid", [True, False])
def test_all_ignore_or_padded_examples_have_no_supervision(example_valid):
    labels = tf.constant([[255, 255], [255, 255]], tf.uint8)
    targets = semantic_map_to_targets(
        labels,
        class_values=(0, 1, 2),
        ignore_value=255,
        pixel_valid_mask=tf.ones([2, 2], tf.bool),
        example_valid=example_valid,
    )
    assert int(targets["num_targets"]) == 0
    assert not bool(targets["supervision_valid"])
    assert not targets["pixel_valid_mask"].numpy().any()
    assert not targets["masks"].numpy().any()


def test_missing_mask_and_explicit_row_validity():
    empty = semantic_map_to_targets(
        None,
        class_values=(0, 1, 2),
        ignore_value=255,
        pixel_valid_mask=tf.zeros([2, 3], tf.bool),
    )
    assert empty["masks"].shape == (3, 2, 3)
    assert not bool(empty["supervision_valid"])
    with pytest.raises(tf.errors.InvalidArgumentError, match="missing mask"):
        semantic_map_to_targets(
            None,
            class_values=(0,),
            ignore_value=255,
            pixel_valid_mask=tf.ones([2, 3], tf.bool),
        )
    padded = semantic_map_to_targets(
        tf.zeros([2, 3], tf.int32),
        class_values=(0,),
        ignore_value=255,
        pixel_valid_mask=tf.ones([2, 3], tf.bool),
        example_valid=False,
    )
    assert not bool(padded["supervision_valid"])
    assert not padded["pixel_valid_mask"].numpy().any()


def test_converter_rejects_invalid_mapping_labels_and_validity():
    for values, ignore in [((0, 0), 255), ((0, 255), 255), ((0,), 2**31)]:
        with pytest.raises(ValueError):
            semantic_map_to_targets(
                tf.zeros([2, 2], tf.int32),
                class_values=values,
                ignore_value=ignore,
                pixel_valid_mask=tf.ones([2, 2], tf.bool),
            )
    with pytest.raises(TypeError, match="boolean"):
        semantic_map_to_targets(
            tf.zeros([2, 2], tf.int32),
            class_values=(0,),
            ignore_value=255,
            pixel_valid_mask=tf.ones([2, 2], tf.int32),
        )
    with pytest.raises(tf.errors.InvalidArgumentError, match="undeclared"):
        semantic_map_to_targets(
            tf.constant([[0, 7]], tf.int32),
            class_values=(0,),
            ignore_value=255,
            pixel_valid_mask=tf.constant([[True, False]]),
        )
    with pytest.raises(tf.errors.InvalidArgumentError, match="size mismatch"):
        semantic_map_to_targets(
            tf.zeros([2, 2], tf.int32),
            class_values=(0,),
            ignore_value=255,
            pixel_valid_mask=tf.ones([3, 2], tf.bool),
        )


def test_traced_conversion_and_sample_wrapper_preserve_identity_and_geometry():
    @tf.function(input_signature=[tf.TensorSpec([None, None, 1], tf.int64)])
    def traced(mask):
        return semantic_map_to_targets(
            mask,
            class_values=(0, 16777217, 2),
            ignore_value=255,
            pixel_valid_mask=tf.ones(tf.shape(mask)[:2], tf.bool),
        )

    result = traced(tf.constant([[[0], [16777217]]], tf.int64))
    assert result["masks"].shape == (3, 1, 2)
    np.testing.assert_array_equal(result["target_valid_mask"], [True, True, False])

    sample = {
        "mask": tf.constant([[0, 2]]),
        "pixel_valid_mask": tf.constant([[True, True]]),
        "metadata": {"example_id": tf.constant("sample-a")},
        "geometry": {"original_size": tf.constant([1, 2])},
        "instance_ref": tf.constant(17),
    }
    wrapped = with_semantic_targets(sample, class_values=(0, 1, 2), ignore_value=255)
    for key in ("metadata", "geometry", "instance_ref"):
        tf.nest.map_structure(np.testing.assert_array_equal, wrapped[key], sample[key])
    assert int(wrapped["targets"]["num_targets"]) == 2


def test_pipeline_builds_targets_after_geometry_and_records_configuration():
    source = {
        "image": tf.zeros([2, 4, 3], tf.uint8),
        "mask": tf.constant([[0, 255, 2, 0], [2, 1, 0, 1]], tf.int32),
        "annotation_valid_mask": tf.constant(
            [[True, True, True, True], [True, True, False, True]]
        ),
    }
    for training in (False, True):
        pipeline = _pipeline()
        pre, augment, _, post = pipeline.build(training)
        prepared = pre(source)
        if training:
            prepared = augment(prepared, tf.constant([7, 11]))
        result = post(prepared)
        targets = result["targets"]
        labels = result["mask"].numpy()
        valid = result["pixel_valid_mask"].numpy()
        np.testing.assert_array_equal(targets["pixel_valid_mask"], valid)
        np.testing.assert_array_equal(
            targets["masks"],
            np.stack([(labels == value) & valid for value in (0, 1, 2)]),
        )
        assert int(targets["num_targets"]) == sum(
            np.any((labels == value) & valid) for value in (0, 1, 2)
        )
        assert int(result["geometry"]["mask_fill_value"]) == 255
        assert "targets" not in replay_dense_geometry(source, result["geometry"])
        snapshot = pipeline.resolve_config(training)
        assert snapshot["stages"]["postprocess"]["target_set"]["capacity"] == 3
    inactive = _pipeline(emit=False)
    pre, _, _, post = inactive.build(False)
    assert "targets" not in post(pre(source))
    assert inactive.resolve_config(False)["stages"]["postprocess"]["target_set"] is None


def test_training_crop_removes_a_class_from_valid_target_slots():
    source = {
        "image": tf.zeros([2, 8, 3], tf.uint8),
        "mask": tf.constant([[0, 0, 0, 0, 0, 0, 0, 2]] * 2, tf.int32),
    }
    pre, augment, _, post = _pipeline().build(True)
    assert np.any(source["mask"].numpy() == 2)
    for index in range(12):
        cropped = augment(pre(source), tf.constant([index, 7]))
        if int(cropped["geometry"]["crop_box"][1]) < 4:
            result = post(cropped)
            np.testing.assert_array_equal(
                result["targets"]["target_valid_mask"], [True, False, False]
            )
            assert int(result["targets"]["num_targets"]) == 1
            assert not np.any(result["mask"].numpy() == 2)
            break
    else:
        pytest.fail("fixture did not produce a crop excluding the right edge")


def test_local_inventory_partial_batch_and_committed_replay(tmp_path):
    admitted = _inventory(tmp_path)
    complete = _epoch(admitted)
    expected = list(complete.batches)
    assert len(expected) == 2
    assert expected[-1]["padding_mask"].tolist() == [True, False]
    padded = expected[-1]["targets"]
    assert not padded["supervision_valid"][1]
    assert padded["num_targets"][1] == 0
    assert not padded["pixel_valid_mask"][1].any()
    assert not padded["masks"][1].any()
    assert {
        int(value)
        for batch in expected
        for value in batch["metadata"]["instance_ref"]
        if value
    } == {10, 11, 12}
    assert sorted(
        value.decode()
        for batch in expected
        for value, valid in zip(
            batch["metadata"]["example_id"], batch["padding_mask"], strict=True
        )
        if valid
    ) == ["sample-0", "sample-1", "sample-2"]
    numeric = list(_epoch(admitted, metadata_mode="numeric_only").batches)
    assert "example_id" not in numeric[0]["metadata"]
    assert numeric[0]["targets"]["masks"].dtype == np.bool_
    assert numeric[0]["targets"]["class_ids"].dtype == np.int32
    supervision = {
        value.decode(): bool(valid)
        for batch in expected
        for value, valid, real in zip(
            batch["metadata"]["example_id"],
            batch["targets"]["supervision_valid"],
            batch["padding_mask"],
            strict=True,
        )
        if real
    }
    assert supervision == {"sample-0": True, "sample-1": False, "sample-2": True}

    cursor = complete.state.with_next_batch(1)
    resumed = _epoch(open_inventory(admitted.path), state=cursor)
    actual = list(resumed.batches)
    assert len(actual) == 1
    tf.nest.map_structure(np.testing.assert_array_equal, actual[0], expected[1])
    for changed in (_pipeline(values=(1, 0, 2)), _pipeline(emit=False)):
        with pytest.raises(ReplayError, match="config_digest") as error:
            _epoch(admitted, state=cursor, pipeline=changed)
        assert error.value.status == "incompatible_replay"


def test_pipeline_rejects_invalid_opt_in_before_build():
    with pytest.raises(ValueError, match="emit_semantic_targets must be boolean"):
        _pipeline(emit="yes").build(False)
    with pytest.raises(ValueError, match="requires geometry_kwargs"):
        DataPipeline(
            "vision/segmentation",
            postproc_kwargs={"emit_semantic_targets": True},
        ).build(False)

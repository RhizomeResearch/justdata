import numpy as np
import pytest
import tensorflow as tf

from justdata.vision.augmentations.auto import RAND_AUGMENT_OPS, rand_augment
from justdata.vision.tasks.segmentation import (
    make_augmentations,
    make_late_augmentations,
    make_postprocessing,
    make_preprocessing,
)


def _coordinate_sample(mask_rank):
    mask = tf.reshape(tf.range(64, dtype=tf.int32), (8, 8))
    image = tf.repeat(tf.cast(mask[..., None], tf.uint8), repeats=3, axis=-1)
    if mask_rank == 3:
        mask = mask[..., None]
    return {"image": image, "mask": mask}


def _binary_segmentation_sample(mask_rank, size=9):
    start = (size - 3) // 2
    mask = np.zeros((size, size), dtype=np.int32)
    mask[start : start + 3, start : start + 3] = 1
    image = np.repeat((mask[..., None] * 255).astype(np.uint8), repeats=3, axis=-1)
    if mask_rank == 3:
        mask = mask[..., None]
    return {"image": tf.constant(image), "mask": tf.constant(mask)}


@pytest.mark.parametrize("mask_rank", [2, 3])
@pytest.mark.parametrize(
    ("crop_type", "image_size"),
    [
        ("none", 8),
        ("random_hflip", 8),
        ("random_rot90_hflip", 8),
        ("random_pad", 6),
    ],
)
def test_segmentation_paired_crops_preserve_coordinate_alignment(
    mask_rank, crop_type, image_size
):
    augment = make_augmentations(
        image_size=image_size,
        crop_type=crop_type,
        augment_type="none",
    )
    sample = _coordinate_sample(mask_rank)
    seed = tf.constant([7, 11], dtype=tf.int32)

    result = augment(sample, seed)
    repeated = augment(sample, seed)
    result_mask = result["mask"] if mask_rank == 2 else result["mask"][..., 0]

    np.testing.assert_array_equal(result["image"][..., 0], result_mask)
    assert set(np.unique(result_mask.numpy())) <= set(range(64))
    assert result["mask"].dtype == tf.int32
    assert result["mask"].shape.rank == mask_rank
    np.testing.assert_array_equal(result["image"], repeated["image"])
    np.testing.assert_array_equal(result["mask"], repeated["mask"])


@pytest.mark.parametrize("mask_rank", [2, 3])
@pytest.mark.parametrize(
    "operation", ["Rotate", "ShearX", "ShearY", "TranslateX", "TranslateY"]
)
def test_segmentation_rand_augment_applies_geometric_ops_to_image_and_mask(
    mask_rank, operation
):
    exclude_ops = [op for op in RAND_AUGMENT_OPS if op != operation]
    ra_kwargs = {
        "exclude_ops": exclude_ops.copy(),
        "magnitude": 20.0,
        "num_layers": 1,
        "translate_const": 3.0,
    }
    augment = make_augmentations(
        image_size=9,
        crop_type="none",
        ra_kwargs=ra_kwargs,
        mask_fill_value=99,
    )
    sample = _binary_segmentation_sample(mask_rank)
    seed = tf.constant([3, 5])

    result = augment(sample, seed)
    repeated = augment(sample, seed)
    result_mask = result["mask"] if mask_rank == 2 else result["mask"][..., 0]
    valid = result_mask.numpy() != 99

    assert ra_kwargs["exclude_ops"] == exclude_ops
    assert set(np.unique(result_mask.numpy())) <= {0, 1, 99}
    assert result["mask"].dtype == tf.int32
    assert result["mask"].shape.rank == mask_rank
    assert not np.array_equal(result["mask"].numpy(), sample["mask"].numpy())
    np.testing.assert_array_equal(
        (result["image"][..., 0].numpy() >= 128)[valid],
        (result_mask.numpy() == 1)[valid],
    )
    np.testing.assert_array_equal(result["image"], repeated["image"])
    np.testing.assert_array_equal(result["mask"], repeated["mask"])


@pytest.mark.parametrize(
    ("augment_type", "policy_kwargs"),
    [
        (
            "rand_augment",
            {"magnitude": 20.0, "num_layers": 1, "translate_const": 3.0},
        ),
        ("trivial_augment", {"translate_const": 3.0}),
        ("trivial_augment_wide", {}),
    ],
)
def test_segmentation_builtin_auto_policies_return_paired_targets(
    augment_type, policy_kwargs
):
    exclude_ops = [op for op in RAND_AUGMENT_OPS if op != "TranslateX"]
    policy_kwargs = policy_kwargs | {"exclude_ops": exclude_ops}
    kwargs = {
        "ra_kwargs" if augment_type == "rand_augment" else "ta_kwargs": policy_kwargs
    }
    augment = make_augmentations(
        image_size=65,
        crop_type="none",
        augment_type=augment_type,
        **kwargs,
    )
    sample = _binary_segmentation_sample(mask_rank=2, size=65)

    result = augment(sample, tf.constant([1, 2]))

    assert result["image"].shape == sample["image"].shape
    assert result["mask"].shape == sample["mask"].shape
    assert result["mask"].dtype == sample["mask"].dtype
    assert set(np.unique(result["mask"].numpy())) <= {0, 1, 255}
    assert not np.array_equal(result["mask"].numpy(), sample["mask"].numpy())


def test_rand_augment_rejects_combined_bbox_and_segmentation_targets():
    sample = _binary_segmentation_sample(mask_rank=2)

    with pytest.raises(ValueError, match="cannot transform bboxes and a mask together"):
        rand_augment(
            sample["image"],
            tf.constant([3, 5]),
            bboxes=tf.constant([[0.0, 0.0, 1.0, 1.0]]),
            segmentation_mask=sample["mask"],
        )


def test_segmentation_paired_rand_augment_honors_probability_across_layers():
    exclude_ops = [op for op in RAND_AUGMENT_OPS if op != "TranslateY"]
    sample = _binary_segmentation_sample(mask_rank=2)
    common_kwargs = {
        "exclude_ops": exclude_ops,
        "magnitude": 20.0,
        "num_layers": 2,
        "translate_const": 2.0,
    }
    disabled = make_augmentations(
        image_size=9,
        crop_type="none",
        ra_kwargs=common_kwargs | {"prob_to_apply": 0.0},
    )
    enabled = make_augmentations(
        image_size=9,
        crop_type="none",
        ra_kwargs=common_kwargs | {"prob_to_apply": 1.0},
    )
    seed = tf.constant([19, 23])

    disabled_result = disabled(sample, seed)
    enabled_result = enabled(sample, seed)

    np.testing.assert_array_equal(disabled_result["image"], sample["image"])
    np.testing.assert_array_equal(disabled_result["mask"], sample["mask"])
    assert not np.array_equal(enabled_result["mask"].numpy(), sample["mask"].numpy())


def test_segmentation_rand_augment_is_photometric_and_does_not_mutate_kwargs():
    exclude_ops = [op for op in RAND_AUGMENT_OPS if op != "Brightness"]
    ra_kwargs = {
        "exclude_ops": exclude_ops.copy(),
        "magnitude": 30.0,
        "num_layers": 1,
    }
    augment = make_augmentations(
        image_size=8,
        crop_type="none",
        ra_kwargs=ra_kwargs,
    )
    sample = _coordinate_sample(mask_rank=3)
    seed = tf.constant([13, 17], dtype=tf.int32)

    result = augment(sample, seed)
    repeated = augment(sample, seed)

    assert ra_kwargs["exclude_ops"] == exclude_ops
    assert not np.array_equal(result["image"].numpy(), sample["image"].numpy())
    np.testing.assert_array_equal(result["mask"], sample["mask"])
    np.testing.assert_array_equal(result["image"], repeated["image"])
    np.testing.assert_array_equal(result["mask"], repeated["mask"])


def test_segmentation_pipeline():
    preproc = make_preprocessing()
    aug = make_augmentations(image_size=32, enable=True)
    laug = make_late_augmentations()
    postproc = make_postprocessing(image_size=32, is_training=True)

    image = tf.random.uniform((64, 64, 3), minval=0, maxval=255, dtype=tf.float32)
    mask = tf.random.uniform((64, 64, 1), minval=0, maxval=21, dtype=tf.int32)
    sample = {"image": image, "mask": mask}

    # Preprocess
    sample = preproc(sample)
    assert "image" in sample
    assert "mask" in sample

    # Augment
    seed = tf.constant([1, 2], dtype=tf.int32)
    sample = aug(sample, seed=seed)
    assert "image" in sample
    assert "mask" in sample

    # Check same spatial size
    assert sample["image"].shape[:2] == sample["mask"].shape[:2]

    # Postprocess
    sample = postproc(sample)
    assert "image" in sample
    assert "mask" in sample

    # Check mask rank (should be 2 according to postprocessing squeezed output if last dim is 1)
    assert len(sample["mask"].shape) == 2
    assert sample["mask"].shape == (32, 32)

    # Late Augment (no-op in segmentation currently)
    batch = {
        "image": tf.expand_dims(sample["image"], 0),
        "mask": tf.expand_dims(sample["mask"], 0),
    }

    batch = laug(batch)
    assert "image" in batch
    assert "mask" in batch

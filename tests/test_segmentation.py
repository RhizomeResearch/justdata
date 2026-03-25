import tensorflow as tf

from justdata.vision.tasks.segmentation import (
    make_augmentations,
    make_late_augmentations,
    make_postprocessing,
    make_preprocessing,
)


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

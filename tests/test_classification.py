import tensorflow as tf

from justdata.tasks.classification import (
    make_augmentations,
    make_late_augmentations,
    make_postprocessing,
    make_preprocessing,
)


def test_classification_pipeline():
    preproc = make_preprocessing()
    aug = make_augmentations(image_size=32, enable=True, mode="sl")
    laug = make_late_augmentations(enable=True, mixup_alpha=0.8, cutmix_alpha=1.0)
    postproc = make_postprocessing(image_size=32, num_classes=10, is_training=True)

    image = tf.random.uniform((64, 64, 3), minval=0, maxval=255, dtype=tf.float32)
    sample = {"image": image, "label": tf.constant(1, dtype=tf.int64)}

    # Preprocess
    sample = preproc(sample)
    assert "image" in sample

    # Augment
    seed = tf.constant([1, 2], dtype=tf.int32)
    sample = aug(sample, seed=seed)
    assert "image" in sample

    # Postprocess
    sample = postproc(sample)
    assert "image" in sample

    # Batching for late augmentations
    batch = {
        "image": tf.expand_dims(sample["image"], 0),
        "label": tf.expand_dims(sample["label"], 0),
    }

    # Late Augment
    seed = tf.constant([1, 2], dtype=tf.int32)
    batch = laug(batch, num_classes=10, seed=seed)

    assert "image" in batch
    assert "label" in batch
    # Check that image is float32
    assert batch["image"].dtype == tf.float32


def test_classification_pipeline_ssl():
    preproc = make_preprocessing()
    aug = make_augmentations(image_size=32, enable=True, mode="ssl")
    postproc = make_postprocessing(image_size=32, is_training=True)

    image = tf.random.uniform((64, 64, 3), minval=0, maxval=255, dtype=tf.float32)
    sample = {"image": image}

    sample = preproc(sample)
    seed = tf.constant([1, 2], dtype=tf.int32)
    sample = aug(sample, seed=seed)

    assert "global_crops" in sample
    assert "local_crops" in sample

    sample = postproc(sample)
    assert "global_crops" in sample
    assert "local_crops" in sample

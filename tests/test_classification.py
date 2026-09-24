import tensorflow as tf

from justdata.vision.tasks.classification import (
    make_augmentations,
    make_postprocessing,
    make_preprocessing,
)


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


def test_classification_eval_view_metadata():
    postproc = make_postprocessing(
        image_size=32,
        is_training=False,
        normalize_image=False,
        permute_image=False,
        eval_view_config={
            "image_size": 32,
            "mode": "center_crop",
            "include_flip": True,
        },
    )
    image = tf.zeros((64, 64, 3), dtype=tf.uint8)
    sample = {"image": image, "label": tf.constant(1, dtype=tf.int64)}

    result = postproc(sample)

    assert result["image"].shape == (2, 32, 32, 3)
    assert "view_metadata" in result
    assert result["view_metadata"]["flip"].numpy().tolist() == [False, True]

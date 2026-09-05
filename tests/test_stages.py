import numpy as np
import pytest
import tensorflow as tf

from justdata.vision.stages import (
    EvalViewConfig,
    _resize_for_eval_view,
    _vision_crop_boxes,
    apply_eval_views,
    normalize_image_format,
    resize_and_normalize,
)


class TestNormalizeImageFormat:
    def test_hwc_rgb_passthrough(self, classification_sample):
        result = normalize_image_format(classification_sample)
        assert result["image"].shape == (32, 32, 3)
        assert result["label"] == classification_sample["label"]

    def test_grayscale_to_rgb(self):
        sample = {"image": tf.zeros([32, 32, 1], dtype=tf.uint8)}
        result = normalize_image_format(sample)
        assert result["image"].shape[-1] == 3

    def test_2d_image_gets_channel_dim(self):
        sample = {"image": tf.zeros([32, 32], dtype=tf.uint8)}
        result = normalize_image_format(sample)
        assert len(result["image"].shape) == 3
        assert result["image"].shape[-1] == 3  # grayscale -> RGB

    def test_chw_to_hwc(self):
        # CHW: 3x32x32
        sample = {"image": tf.zeros([3, 32, 32], dtype=tf.uint8)}
        result = normalize_image_format(sample)
        assert result["image"].shape == (32, 32, 3)

    def test_rgba_to_rgb(self):
        sample = {"image": tf.zeros([32, 32, 4], dtype=tf.uint8)}
        result = normalize_image_format(sample)
        assert result["image"].shape[-1] == 3

    def test_custom_image_key(self):
        sample = {"photo": tf.zeros([32, 32, 3], dtype=tf.uint8)}
        result = normalize_image_format(sample, image_key="photo")
        assert result["photo"].shape == (32, 32, 3)

    def test_preserves_other_keys(self, classification_sample):
        result = normalize_image_format(classification_sample)
        assert "label" in result
        assert result["label"].numpy() == classification_sample["label"].numpy()


class TestResizeAndNormalize:
    def test_resize_and_normalize_basic(self, classification_sample):
        result = resize_and_normalize(
            classification_sample,
            image_keys=["image"],
            image_size=16,
            resize_size=None,
            normalize_image=True,
            normalization_params=((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            permute=False,
        )
        assert result["image"].shape == (16, 16, 3)
        assert result["image"].dtype == tf.float32

    def test_resize_with_permute(self, classification_sample):
        result = resize_and_normalize(
            classification_sample,
            image_keys=["image"],
            image_size=16,
            resize_size=None,
            normalize_image=True,
            normalization_params=((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            permute=True,
        )
        assert result["image"].shape == (3, 16, 16)

    def test_val_resize_with_short_side(self):
        sample = {"image": tf.zeros([100, 200, 3], dtype=tf.uint8)}
        result = resize_and_normalize(
            sample,
            image_keys=["image"],
            image_size=64,
            resize_size=128,
            normalize_image=True,
            normalization_params=((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            permute=False,
        )
        assert result["image"].shape == (64, 64, 3)

    def test_skips_missing_keys(self, classification_sample):
        result = resize_and_normalize(
            classification_sample,
            image_keys=["nonexistent"],
            image_size=16,
            resize_size=None,
            normalize_image=True,
            normalization_params=((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            permute=False,
        )
        # Original image unchanged
        assert "nonexistent" not in result

    def test_no_normalize(self, classification_sample):
        result = resize_and_normalize(
            classification_sample,
            image_keys=["image"],
            image_size=16,
            resize_size=None,
            normalize_image=False,
            normalization_params=None,
            permute=False,
        )
        # Should still be float (from resize) but not normalized
        assert result["image"].dtype == tf.float32
        assert tf.reduce_max(result["image"]) <= 255.0

    def test_per_image_channel_standardization(self):
        image = tf.reshape(
            tf.cast(tf.range(16 * 16 * 3), tf.uint8),
            [16, 16, 3],
        )
        result = resize_and_normalize(
            {"image": image},
            image_keys=["image"],
            image_size=16,
            resize_size=None,
            normalize_image=True,
            normalization_mode="per_image",
            normalization_params=None,
            permute=False,
        )

        means = tf.reduce_mean(result["image"], axis=[0, 1]).numpy()
        stds = tf.math.reduce_std(result["image"], axis=[0, 1]).numpy()
        np.testing.assert_allclose(means, np.zeros(3), atol=1e-6)
        np.testing.assert_allclose(stds, np.ones(3), atol=1e-6)

    def test_per_image_channel_standardization_handles_constant_channels(self):
        image = tf.ones([16, 16, 3], dtype=tf.uint8) * 7
        result = resize_and_normalize(
            {"image": image},
            image_keys=["image"],
            image_size=16,
            resize_size=None,
            normalize_image=True,
            normalization_mode="per_image",
            normalization_params=None,
            permute=False,
        )

        np.testing.assert_allclose(result["image"].numpy(), np.zeros([16, 16, 3]))


def test_vision_eval_views_add_metadata_and_flip_views():
    sample = {"image": tf.zeros([100, 120, 3], dtype=tf.uint8)}

    result = apply_eval_views(
        sample,
        config={
            "image_size": 32,
            "mode": "multi_crop",
            "num_crops": 5,
            "include_flip": True,
        },
    )

    assert result["image"].shape == (10, 32, 32, 3)
    assert result["view_metadata"]["crop_box"].shape == (10, 4)
    assert result["view_metadata"]["flip"].shape == (10,)
    assert result["view_metadata"]["view_index"].numpy().tolist() == list(range(10))


@pytest.mark.parametrize(
    "mode,count",
    [("center_crop", 3), ("resize_crop", 1), ("multi_crop", 1), ("multi_crop", 5)],
)
@pytest.mark.parametrize("include_flip", [False, True])
@pytest.mark.parametrize("shape", [(20, 28, 3), (45, 51, 3)])
def test_eval_views_exactly_match_mapped_crops(mode, count, include_flip, shape):
    config = EvalViewConfig(
        image_size=32, mode=mode, num_crops=count, include_flip=include_flip
    )
    image = tf.reshape(tf.cast(tf.range(np.prod(shape)), tf.uint8), shape)

    def reference(image):
        resized, scale = _resize_for_eval_view(image, config)
        boxes = _vision_crop_boxes(resized, config)
        crops = tf.map_fn(
            lambda box: tf.image.crop_to_bounding_box(resized, *tf.unstack(box)),
            boxes,
            fn_output_signature=resized.dtype,
        )
        flip = tf.zeros([tf.shape(crops)[0]], tf.bool)
        if include_flip:
            crops = tf.concat([crops, tf.image.flip_left_right(crops)], axis=0)
            boxes = tf.concat([boxes, boxes], axis=0)
            flip = tf.concat([flip, tf.ones_like(flip)], axis=0)
        views = tf.shape(crops)[0]
        return {
            "image": crops,
            "view_metadata": {
                "crop_box": boxes,
                "scale": tf.fill([views], scale),
                "flip": flip,
                "view_index": tf.range(views),
            },
        }

    def actual(image):
        return apply_eval_views({"image": image}, config=config)

    signature = [tf.TensorSpec([None, None, 3], tf.uint8)]
    for wrap in (lambda fn: fn, lambda fn: tf.function(fn, input_signature=signature)):
        expected = wrap(reference)(image)
        result = wrap(actual)(image)
        tf.nest.assert_same_structure(result, expected)
        for left, right in zip(tf.nest.flatten(result), tf.nest.flatten(expected)):
            assert left.shape == right.shape
            assert left.dtype == right.dtype
            assert left.numpy().tobytes() == right.numpy().tobytes()

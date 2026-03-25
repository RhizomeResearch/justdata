import tensorflow as tf

from justdata.vision.stages import normalize_image_format, resize_and_normalize


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

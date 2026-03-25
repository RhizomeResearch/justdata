import numpy as np
import tensorflow as tf

from justdata.transforms import (
    center_crop,
    nhwc_to_nchw,
    normalize,
    resize_image,
    resize_short_side,
)


class TestNormalize:
    def test_output_dtype_is_float32(self, rgb_image_uint8):
        result = normalize(rgb_image_uint8, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        assert result.dtype == tf.float32

    def test_zero_mean_unit_std_on_constant_image(self):
        # Image of all 128s => pixel/255 = ~0.502
        image = tf.fill([4, 4, 3], tf.constant(128, dtype=tf.uint8))
        mean = (128.0 / 255.0,) * 3
        std = (1.0,) * 3
        result = normalize(image, mean, std)
        np.testing.assert_allclose(result.numpy(), 0.0, atol=1e-5)

    def test_imagenet_normalization_range(self, rgb_image_uint8):
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        result = normalize(rgb_image_uint8, mean, std)
        # Normalized values should roughly be in [-3, 3]
        assert tf.reduce_min(result) > -5.0
        assert tf.reduce_max(result) < 5.0

    def test_batch_normalization(self, rgb_batch_uint8):
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        result = normalize(rgb_batch_uint8, mean, std)
        assert result.shape == (4, 32, 32, 3)
        assert result.dtype == tf.float32


class TestNhwcToNchw:
    def test_3d_permutation(self, rgb_image_uint8):
        result = nhwc_to_nchw(rgb_image_uint8)
        assert result.shape == (3, 32, 32)

    def test_4d_permutation(self, rgb_batch_uint8):
        result = nhwc_to_nchw(rgb_batch_uint8)
        assert result.shape == (4, 3, 32, 32)

    def test_values_preserved(self, rgb_image_uint8):
        result = nhwc_to_nchw(rgb_image_uint8)
        # Channel 0 of CHW should equal [:,:,0] of HWC
        np.testing.assert_array_equal(
            result[0].numpy(), rgb_image_uint8[:, :, 0].numpy()
        )


class TestResizeShortSide:
    def test_square_image(self, rgb_image_uint8):
        result = resize_short_side(rgb_image_uint8, 64)
        assert result.shape[0] == 64
        assert result.shape[1] == 64

    def test_rectangular_image(self):
        image = tf.zeros([100, 200, 3], dtype=tf.uint8)
        result = resize_short_side(image, 50)
        assert result.shape[0] == 50
        assert result.shape[1] == 100


class TestCenterCrop:
    def test_output_size(self):
        image = tf.zeros([64, 64, 3])
        result = center_crop(image, 32)
        assert result.shape == (32, 32, 3)

    def test_crop_is_centered(self):
        # Create image with known center
        image = tf.zeros([10, 10, 1])
        indices = tf.constant([[5, 5, 0]])
        updates = tf.constant([1.0])
        image = tf.tensor_scatter_nd_update(image, indices, updates)
        result = center_crop(image, 4)
        # The pixel at (5,5) in original => (5-3, 5-3) = (2,2) in cropped
        assert result[2, 2, 0].numpy() == 1.0


class TestResizeImage:
    def test_direct_resize_no_short_side(self, rgb_image_uint8):
        result = resize_image(rgb_image_uint8, image_size=16, resize_size=None)
        assert result.shape == (16, 16, 3)

    def test_resize_with_short_side_and_center_crop(self):
        image = tf.zeros([100, 200, 3], dtype=tf.uint8)
        result = resize_image(image, image_size=64, resize_size=128)
        assert result.shape == (64, 64, 3)

    def test_output_dtype_is_float(self, rgb_image_uint8):
        result = resize_image(rgb_image_uint8, image_size=16, resize_size=None)
        assert result.dtype == tf.float32

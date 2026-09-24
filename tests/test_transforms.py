import numpy as np
import tensorflow as tf

import pytest

from justdata.vision.transforms import (
    center_crop,
    nhwc_to_nchw,
    normalize,
    pad_to_patch_multiple,
    resize_image,
    resize_short_side,
)


class TestNormalize:
    def test_scales_to_unit_range_then_divides_by_std(self):
        image = tf.fill([2, 2, 3], tf.constant(255, dtype=tf.uint8))
        result = normalize(image, (0.5, 0.5, 0.5), (0.25, 0.5, 1.0))
        assert result.dtype == tf.float32
        np.testing.assert_allclose(result.numpy()[0, 0], [2.0, 1.0, 0.5])

    def test_zero_mean_unit_std_on_constant_image(self):
        # Image of all 128s => pixel/255 = ~0.502
        image = tf.fill([4, 4, 3], tf.constant(128, dtype=tf.uint8))
        mean = (128.0 / 255.0,) * 3
        std = (1.0,) * 3
        result = normalize(image, mean, std)
        np.testing.assert_allclose(result.numpy(), 0.0, atol=1e-5)

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


@pytest.mark.parametrize(
    ("shape", "patch_size", "expected"),
    [
        ((224, 224, 3), 14, (224, 224, 3)),
        ((225, 225, 3), 14, (238, 238, 3)),
        ((200, 300, 3), 16, (208, 304, 3)),
    ],
    ids=["aligned", "pad-to-238", "patch16-rectangular"],
)
def test_pad_to_patch_multiple_rounds_up_to_the_patch_size(shape, patch_size, expected):
    result = pad_to_patch_multiple(tf.zeros(shape), patch_size=patch_size)
    assert result.shape == expected


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

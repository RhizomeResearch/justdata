import numpy as np
import pytest
import tensorflow as tf

from justdata.vision.utils import gaussian_filter2d


@pytest.mark.parametrize("shape", [(9, 10), (9, 10, 3), (2, 9, 10, 3)])
@pytest.mark.parametrize("dtype", [tf.uint8, tf.float32, tf.float64])
@pytest.mark.parametrize("padding", ["REFLECT", "SYMMETRIC", "CONSTANT"])
def test_gaussian_preserves_constant_image_shape_and_dtype(shape, dtype, padding):
    image = tf.fill(shape, tf.cast(32, dtype))
    result = gaussian_filter2d(image, 4, 0.8, padding, constant_values=32)

    assert result.shape == image.shape
    assert result.dtype == dtype
    # Integer output truncates after convolution.
    np.testing.assert_allclose(result, image, atol=1 if dtype == tf.uint8 else 1e-5)


def test_gaussian_impulse_is_symmetric_and_preserves_total_weight():
    image = tf.scatter_nd([[4, 4]], [1.0], [9, 9])
    result = gaussian_filter2d(image, 3, 1.0, "CONSTANT").numpy()

    np.testing.assert_allclose(result.sum(), 1.0, atol=1e-6)
    np.testing.assert_array_equal(result, result.T)
    np.testing.assert_array_equal(result, result[::-1, ::-1])
    assert np.count_nonzero(result) == 9


def test_gaussian_zero_sigma_retains_nan_behavior():
    result = gaussian_filter2d(tf.ones([5, 5]), sigma=0.0)
    assert np.isnan(result.numpy()).all()


def test_gaussian_negative_sigma_keeps_validation_error():
    with pytest.raises(tf.errors.InvalidArgumentError, match="sigma_h must be >= 0"):
        gaussian_filter2d(tf.ones([5, 5]), sigma=-1.0)

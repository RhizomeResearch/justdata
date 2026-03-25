import tensorflow as tf

from justdata.corruptions.registry import _get_severity_index, register_corruption
from justdata.utils import _pad, from_4d, gaussian_filter2d, to_4d

DEFOCUS_RADIUS = (3.0, 4.0, 6.0, 8.0, 10.0)

DEFOCUS_ALIAS = (0.1, 0.5, 0.5, 0.5, 0.5)


@register_corruption("blur")
@tf.function
def defocus_blur(image: tf.Tensor, severity: int, seed: tf.Tensor) -> tf.Tensor:
    """
    Applies Defocus Blur using a disk kernel convolution.

    Note: Standard Gaussian blur mimics atmospheric scattering.
    Defocus blur (bokeh) is physically caused by a circular aperture,
    resulting in a disk-shaped Point Spread Function (PSF).
    """
    idx = _get_severity_index(severity)
    radius = tf.gather(tf.constant(DEFOCUS_RADIUS, dtype=tf.float32), idx)
    alias_blur = tf.gather(tf.constant(DEFOCUS_ALIAS, dtype=tf.float32), idx)

    # Kernel size must be large enough to hold the disk (approx 2*r + 1)
    k_size = tf.cast(tf.math.ceil(radius * 2.0) + 1, tf.int32)
    # Ensure odd
    k_size = tf.where(k_size % 2 == 0, k_size + 1, k_size)

    # Coordinate grid
    x = tf.range(tf.cast(k_size, tf.float32), dtype=tf.float32)
    x = x - (tf.cast(k_size, tf.float32) - 1.0) / 2.0
    xx, yy = tf.meshgrid(x, x)

    # Disk equation: x^2 + y^2 <= r^2
    dist_sq = xx**2 + yy**2
    mask = tf.cast(dist_sq <= radius**2, tf.float32)

    # Normalize kernel so energy is preserved
    kernel = mask / (tf.reduce_sum(mask) + 1e-8)

    kernel = kernel[:, :, tf.newaxis, tf.newaxis]
    channels = tf.shape(image)[-1]
    kernel = tf.tile(kernel, [1, 1, channels, 1])

    image_f = tf.cast(image, tf.float32)
    original_ndims = tf.rank(image_f)
    image_4d = to_4d(image_f)

    # Pad to preserve size
    image_padded = _pad(image_4d, [k_size, k_size], pad_mode="REFLECT")

    blurred = tf.nn.depthwise_conv2d(
        image_padded, kernel, strides=[1, 1, 1, 1], padding="VALID"
    )

    blurred = from_4d(blurred, original_ndims)

    # Slight Gaussian "aliasing" blur (simulates soft edges of aperture)
    return gaussian_filter2d(
        blurred, filter_shape=5, sigma=alias_blur, padding="REFLECT"
    )

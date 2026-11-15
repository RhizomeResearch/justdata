"""Float32 HWC filters shared by corruptions and compositing.

Every filter keeps the spatial size of its input and pads with reflect-101
borders, so results do not depend on an implicit zero frame.
"""

import tensorflow as tf


def reflect_indices(length: tf.Tensor, radius: tf.Tensor) -> tf.Tensor:
    """Gather indices that pad an axis by ``radius`` with reflect-101 borders.

    Radii larger than the axis keep reflecting, and a single-pixel axis repeats
    its only pixel.
    """
    length = tf.cast(length, tf.int32)
    radius = tf.cast(radius, tf.int32)

    def repeat_single_pixel() -> tf.Tensor:
        return tf.zeros([length + 2 * radius], dtype=tf.int32)

    def reflect_101() -> tf.Tensor:
        positions = tf.range(-radius, length + radius)
        period = 2 * (length - 1)
        reflected = tf.math.floormod(positions, period)
        return tf.where(reflected < length, reflected, period - reflected)

    return tf.cond(length > 1, reflect_101, repeat_single_pixel)


def reflect_pad_hwc(
    image: tf.Tensor,
    radius_y: tf.Tensor,
    radius_x: tf.Tensor,
) -> tf.Tensor:
    y_indices = reflect_indices(tf.shape(image)[0], radius_y)
    x_indices = reflect_indices(tf.shape(image)[1], radius_x)
    image = tf.gather(image, y_indices, axis=0)
    return tf.gather(image, x_indices, axis=1)


def gaussian_kernel_1d(sigma: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Return a float32 Gaussian kernel of radius ``ceil(3 * sigma)`` summing to one."""
    sigma = tf.cast(sigma, tf.float32)
    radius = tf.cast(tf.math.ceil(3.0 * sigma), tf.int32)
    positions = tf.cast(tf.range(-radius, radius + 1), tf.float32)
    kernel = tf.exp(-tf.square(positions) / (2.0 * tf.square(sigma)))
    kernel /= tf.reduce_sum(kernel)
    return kernel, radius


def depthwise_hwc(image: tf.Tensor, kernel_2d: tf.Tensor) -> tf.Tensor:
    """Apply one 2-D kernel to every channel without padding (VALID)."""
    channels = tf.shape(image)[-1]
    kernel = tf.tile(kernel_2d[:, :, tf.newaxis, tf.newaxis], [1, 1, channels, 1])
    return tf.nn.depthwise_conv2d(
        image[tf.newaxis, ...],
        kernel,
        strides=[1, 1, 1, 1],
        padding="VALID",
    )[0]


def gaussian_blur_hwc(
    image: tf.Tensor,
    sigma_y: tf.Tensor,
    sigma_x: tf.Tensor,
) -> tf.Tensor:
    """Separable, possibly anisotropic Gaussian blur; a zero sigma leaves that axis."""
    image = tf.cast(image, tf.float32)
    sigma_y = tf.cast(sigma_y, tf.float32)
    sigma_x = tf.cast(sigma_x, tf.float32)

    def blur_axis(values: tf.Tensor, sigma: tf.Tensor, axis: int) -> tf.Tensor:
        def blur() -> tf.Tensor:
            kernel, radius = gaussian_kernel_1d(sigma)
            if axis == 0:
                padded = reflect_pad_hwc(values, radius, 0)
                return depthwise_hwc(padded, kernel[:, tf.newaxis])
            padded = reflect_pad_hwc(values, 0, radius)
            return depthwise_hwc(padded, kernel[tf.newaxis, :])

        return tf.cond(sigma > 0.0, blur, lambda: values)

    return blur_axis(blur_axis(image, sigma_y, 0), sigma_x, 1)


def correlate_hwc(image: tf.Tensor, kernel: tf.Tensor) -> tf.Tensor:
    """Correlate each channel with an odd-sized ``[kh, kw]`` kernel through the FFT.

    ``out[y, x] = sum(kernel[i, j] * image[y + i - kh // 2, x + j - kw // 2])``.
    The FFT keeps large kernels, such as long motion paths, affordable.
    """
    image = tf.cast(image, tf.float32)
    kernel = tf.cast(kernel, tf.float32)
    height, width = tf.shape(image)[0], tf.shape(image)[1]
    kernel_height, kernel_width = tf.shape(kernel)[0], tf.shape(kernel)[1]
    padded = reflect_pad_hwc(image, kernel_height // 2, kernel_width // 2)
    padded_size = tf.shape(padded)[:2]

    channels_first = tf.transpose(padded, [2, 0, 1])
    image_spectrum = tf.signal.rfft2d(channels_first, fft_length=padded_size)
    flipped = tf.reverse(kernel, axis=[0, 1])
    kernel_spectrum = tf.signal.rfft2d(flipped, fft_length=padded_size)
    full = tf.signal.irfft2d(image_spectrum * kernel_spectrum, fft_length=padded_size)
    valid = full[
        :,
        kernel_height - 1 : kernel_height - 1 + height,
        kernel_width - 1 : kernel_width - 1 + width,
    ]
    return tf.transpose(valid, [1, 2, 0])


def splat_points(
    points_yx: tf.Tensor,
    weights: tf.Tensor,
    size: tf.Tensor,
) -> tf.Tensor:
    """Bilinearly splat weighted points into an ``[H, W]`` grid of integer pixel centres.

    Contributions that fall outside the grid are dropped.
    """
    points_yx = tf.cast(points_yx, tf.float32)
    weights = tf.cast(weights, tf.float32)
    top_left = tf.floor(points_yx)
    fraction = points_yx - top_left
    top_left = tf.cast(top_left, tf.int32)
    corners = []
    corner_weights = []
    for dy in (0, 1):
        for dx in (0, 1):
            wy = fraction[:, 0] if dy else 1.0 - fraction[:, 0]
            wx = fraction[:, 1] if dx else 1.0 - fraction[:, 1]
            corners.append(top_left + tf.constant([dy, dx], tf.int32))
            corner_weights.append(weights * wy * wx)
    indices = tf.concat(corners, axis=0)
    values = tf.concat(corner_weights, axis=0)
    inside = tf.reduce_all((indices >= 0) & (indices < size[tf.newaxis, :]), axis=1)
    return tf.scatter_nd(
        tf.boolean_mask(indices, inside),
        tf.boolean_mask(values, inside),
        size,
    )


__all__ = [
    "correlate_hwc",
    "depthwise_hwc",
    "gaussian_blur_hwc",
    "gaussian_kernel_1d",
    "reflect_indices",
    "reflect_pad_hwc",
    "splat_points",
]

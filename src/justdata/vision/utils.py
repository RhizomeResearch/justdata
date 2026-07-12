import math
from typing import List, Optional, Tuple, Union

import tensorflow as tf


def _pad(
    image: tf.Tensor,
    filter_shape: Union[List[int], Tuple[int, ...]],
    pad_mode: str = "CONSTANT",
    constant_values: Union[int, tf.Tensor] = 0,
) -> tf.Tensor:
    """Explicitly pads a 4-D image.

    Equivalent to the implicit padding method offered in `tf.nn.conv2d` and
    `tf.nn.depthwise_conv2d`, but supports non-zero, reflect and symmetric
    padding mode. For the even-sized filter, it pads one more value to the
    right or the bottom side.

    Args:
      image: A 4-D `Tensor` of shape `[batch_size, height, width, channels]`.
      filter_shape: A `tuple`/`list` of 2 integers, specifying the height and
        width of the 2-D filter.
      pad_mode: A `string`, one of "REFLECT", "CONSTANT", or "SYMMETRIC". The type of
        padding algorithm to use, which is compatible with `pad_mode` argument in
        `tf.pad`. For more details, please refer to
        https://www.tensorflow.org/api_docs/python/tf/pad.
      constant_values: A `scalar`, the pad value to use in "CONSTANT" padding
        mode.

    Returns:
      A padded image.
    """
    if pad_mode.upper() not in {"REFLECT", "CONSTANT", "SYMMETRIC"}:
        raise ValueError(
            'padding should be one of "REFLECT", "CONSTANT", or "SYMMETRIC".'
        )
    constant_values = tf.convert_to_tensor(constant_values, image.dtype)
    filter_height, filter_width = filter_shape
    pad_top = (filter_height - 1) // 2
    pad_bottom = filter_height - 1 - pad_top
    pad_left = (filter_width - 1) // 2
    pad_right = filter_width - 1 - pad_left
    paddings = [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]]
    return tf.pad(image, paddings, mode=pad_mode, constant_values=constant_values)


def _get_gaussian_kernel(sigma: tf.Tensor, filter_size: tf.Tensor) -> tf.Tensor:
    """Computes 1D Gaussian kernel."""
    x = tf.range(
        tf.cast(-filter_size // 2 + 1, tf.float32),
        tf.cast(filter_size // 2 + 1, tf.float32),
    )
    x = tf.cast(x**2, sigma.dtype)
    x = tf.nn.softmax(-x / (2.0 * tf.cast(sigma**2, x.dtype)))
    return x


def _get_gaussian_kernel_2d(
    gaussian_filter_x: tf.Tensor, gaussian_filter_y: tf.Tensor
) -> tf.Tensor:
    """Computes 2D Gaussian kernel given 1D kernels."""
    return tf.matmul(gaussian_filter_y[:, tf.newaxis], gaussian_filter_x[tf.newaxis, :])


def _convert_translation_to_transform(translations: tf.Tensor) -> tf.Tensor:
    translations = tf.convert_to_tensor(translations, dtype=tf.float32)
    if translations.shape.rank == 1:
        translations = translations[None]

    num_translations = tf.shape(translations)[0]
    zeros = tf.zeros((num_translations,), dtype=tf.float32)
    ones = tf.ones((num_translations,), dtype=tf.float32)

    # The Matrix is [1, 0, -tx, 0, 1, -ty, 0, 0]
    return tf.stack(
        [
            ones,
            zeros,
            -translations[:, 0],
            zeros,
            ones,
            -translations[:, 1],
            zeros,
            zeros,
        ],
        axis=1,
    )


def _convert_angles_to_transform(
    angles: tf.Tensor, image_width: tf.Tensor, image_height: tf.Tensor
) -> tf.Tensor:
    angles = tf.convert_to_tensor(angles, dtype=tf.float32)
    x_offset = (
        (image_width - 1)
        - (
            tf.math.cos(angles) * (image_width - 1)
            - tf.math.sin(angles) * (image_height - 1)
        )
    ) / 2.0
    y_offset = (
        (image_height - 1)
        - (
            tf.math.sin(angles) * (image_width - 1)
            + tf.math.cos(angles) * (image_height - 1)
        )
    ) / 2.0
    num_angles = tf.shape(angles)[0]
    return tf.concat(
        values=[
            tf.math.cos(angles)[:, None],
            -tf.math.sin(angles)[:, None],
            x_offset[:, None],
            tf.math.sin(angles)[:, None],
            tf.math.cos(angles)[:, None],
            y_offset[:, None],
            tf.zeros((num_angles, 2), tf.float32),
        ],
        axis=1,
    )


def _apply_transform(
    image: tf.Tensor,
    transforms: tf.Tensor,
    fill_mode: str = "REFLECT",
    fill_value: float = 0.0,
    interpolation: str = "NEAREST",
) -> tf.Tensor:
    output_shape = tf.shape(image)[1:3]
    return tf.raw_ops.ImageProjectiveTransformV3(
        images=image,
        output_shape=output_shape,
        fill_value=fill_value,
        transforms=transforms,
        fill_mode=fill_mode.upper(),
        interpolation=interpolation.upper(),
    )


def _transform(
    image: tf.Tensor,
    transforms: tf.Tensor,
    interpolation: str = "NEAREST",
    fill_mode: str = "REFLECT",
    fill_value: float = 0.0,
) -> tf.Tensor:
    original_ndims = tf.rank(image)
    transforms = tf.convert_to_tensor(transforms, dtype=tf.float32)
    if transforms.shape.rank == 1:
        transforms = transforms[None]
    # Reuse existing to_4d
    image = to_4d(image)
    image = _apply_transform(
        image,
        transforms,
        fill_mode=fill_mode,
        fill_value=fill_value,
        interpolation=interpolation,
    )
    # Reuse existing from_4d
    return from_4d(image, original_ndims)


def _translate(
    image: tf.Tensor,
    translations: tf.Tensor,
    replace: int = 0,
    interpolation: str = "NEAREST",
) -> tf.Tensor:
    transforms = _convert_translation_to_transform(translations)
    return _transform(
        image,
        transforms=transforms,
        interpolation=interpolation,
        fill_value=float(replace),
        fill_mode="CONSTANT",
    )


def _rotate(
    image: tf.Tensor,
    degrees: float,
    replace: int = 0,
    interpolation: str = "NEAREST",
) -> tf.Tensor:
    degrees_to_radians = math.pi / 180.0
    radians = tf.convert_to_tensor(degrees * degrees_to_radians, dtype=tf.float32)
    if radians.shape.rank == 0:
        radians = radians[None]

    original_ndims = tf.rank(image)
    image = to_4d(image)
    image_height = tf.cast(tf.shape(image)[1], tf.float32)
    image_width = tf.cast(tf.shape(image)[2], tf.float32)

    transforms = _convert_angles_to_transform(radians, image_width, image_height)
    image = _transform(
        image,
        transforms=transforms,
        interpolation=interpolation,
        fill_mode="CONSTANT",
        fill_value=float(replace),
    )
    return from_4d(image, original_ndims)


def _blend(image1: tf.Tensor, image2: tf.Tensor, factor) -> tf.Tensor:
    image1 = tf.cast(image1, tf.float32)
    image2 = tf.cast(image2, tf.float32)
    factor = tf.cast(factor, tf.float32)

    blended = image1 + factor * (image2 - image1)
    blended = tf.clip_by_value(blended, 0.0, 255.0)
    return tf.cast(blended, tf.uint8)


def _fill_rectangle(
    image, center_width, center_height, half_width, half_height, replace=None, seed=None
):
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]

    lower_pad = tf.maximum(0, center_height - half_height)
    upper_pad = tf.maximum(0, image_height - center_height - half_height)
    left_pad = tf.maximum(0, center_width - half_width)
    right_pad = tf.maximum(0, image_width - center_width - half_width)

    cutout_shape = [
        image_height - (lower_pad + upper_pad),
        image_width - (left_pad + right_pad),
    ]
    padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]
    mask = tf.pad(
        tf.zeros(cutout_shape, dtype=image.dtype), padding_dims, constant_values=1
    )
    mask = tf.expand_dims(mask, -1)
    num_channels = tf.shape(image)[-1]
    mask = tf.tile(mask, [1, 1, num_channels])

    if replace is None:
        fill_seed = seed if seed is not None else tf.constant([0, 0], dtype=tf.int32)
        fill = tf.random.stateless_normal(
            tf.shape(image), seed=fill_seed, dtype=image.dtype
        )
    elif isinstance(replace, tf.Tensor):
        fill = replace
    else:
        fill = tf.ones_like(image, dtype=image.dtype) * replace
    return tf.where(tf.equal(mask, 0), fill, image)


def _cutout(
    image: tf.Tensor, pad_size: int, replace: int, seed: tf.Tensor
) -> tf.Tensor:
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]

    seeds = tf.random.split(seed, 3)
    cutout_center_height = tf.random.stateless_uniform(
        [], minval=0, maxval=image_height, dtype=tf.int32, seed=seeds[0]
    )
    cutout_center_width = tf.random.stateless_uniform(
        [], minval=0, maxval=image_width, dtype=tf.int32, seed=seeds[1]
    )

    return _fill_rectangle(
        image,
        cutout_center_width,
        cutout_center_height,
        pad_size,
        pad_size,
        replace=replace,
        seed=seeds[2],
    )


def _solarize_val(image: tf.Tensor, threshold: int = 128) -> tf.Tensor:
    threshold = tf.cast(threshold, image.dtype)
    max_val = tf.cast(255, image.dtype)
    return tf.where(image < threshold, image, max_val - image)


def _solarize_add(
    image: tf.Tensor, addition: int = 0, threshold: int = 128
) -> tf.Tensor:
    threshold = tf.cast(threshold, image.dtype)

    added_image = tf.cast(image, tf.int64) + tf.cast(addition, tf.int64)
    added_image = tf.cast(tf.clip_by_value(added_image, 0, 255), image.dtype)

    return tf.where(image < threshold, added_image, image)


def _posterize(image: tf.Tensor, bits: int) -> tf.Tensor:
    shift = tf.cast(8 - bits, image.dtype)
    return tf.bitwise.left_shift(tf.bitwise.right_shift(image, shift), shift)


def _autocontrast(image: tf.Tensor) -> tf.Tensor:
    def scale_channel(img):
        lo = tf.cast(tf.reduce_min(img), tf.float32)
        hi = tf.cast(tf.reduce_max(img), tf.float32)

        def scale_values(im):
            scale = 255.0 / (hi - lo)
            offset = -lo * scale
            im = tf.cast(im, tf.float32) * scale + offset
            im = tf.clip_by_value(im, 0.0, 255.0)
            return tf.cast(im, tf.uint8)

        return tf.cond(hi > lo, lambda: scale_values(img), lambda: img)

    s1 = scale_channel(image[..., 0])
    s2 = scale_channel(image[..., 1])
    s3 = scale_channel(image[..., 2])
    return tf.stack([s1, s2, s3], -1)


def _sharpness(image: tf.Tensor, factor: float) -> tf.Tensor:
    orig_image = image
    image = tf.cast(image, tf.float32)
    image = tf.expand_dims(image, 0)
    kernel = (
        tf.constant(
            [[1, 1, 1], [1, 5, 1], [1, 1, 1]], dtype=tf.float32, shape=[3, 3, 1, 1]
        )
        / 13.0
    )
    kernel = tf.tile(kernel, [1, 1, 3, 1])
    degenerate = tf.nn.depthwise_conv2d(
        image, kernel, strides=[1, 1, 1, 1], padding="VALID"
    )
    degenerate = tf.clip_by_value(degenerate, 0.0, 255.0)
    degenerate = tf.squeeze(tf.cast(degenerate, tf.uint8), [0])

    mask = tf.ones_like(degenerate)
    paddings = [[1, 1], [1, 1], [0, 0]]
    padded_mask = tf.pad(mask, paddings)
    padded_degenerate = tf.pad(degenerate, paddings)
    result = tf.where(tf.equal(padded_mask, 1), padded_degenerate, orig_image)
    return _blend(result, orig_image, factor)


def _equalize(image: tf.Tensor) -> tf.Tensor:
    def scale_channel(im, c):
        im = tf.cast(im[..., c], tf.int32)
        histo = tf.histogram_fixed_width(im, [0, 255], nbins=256)
        nonzero = tf.where(tf.not_equal(histo, 0))
        nonzero_histo = tf.reshape(tf.gather(histo, nonzero), [-1])
        step = (tf.reduce_sum(nonzero_histo) - nonzero_histo[-1]) // 255

        def build_lut(histo, step):
            lut = (tf.cumsum(histo) + (step // 2)) // step
            lut = tf.concat([[0], lut[:-1]], 0)
            return tf.clip_by_value(lut, 0, 255)

        result = tf.cond(
            tf.equal(step, 0),
            lambda: im,
            lambda: tf.gather(build_lut(histo, step), im),
        )
        return tf.cast(result, tf.uint8)

    s1 = scale_channel(image, 0)
    s2 = scale_channel(image, 1)
    s3 = scale_channel(image, 2)
    return tf.stack([s1, s2, s3], -1)


def _invert(image: tf.Tensor) -> tf.Tensor:
    return tf.cast(255, image.dtype) - image


def _wrap(image: tf.Tensor) -> tf.Tensor:
    shape = tf.shape(image)
    extended_channel = tf.expand_dims(tf.ones(shape[:-1], image.dtype), -1)
    return tf.concat([image, extended_channel], axis=-1)


def _unwrap(image: tf.Tensor, replace: int) -> tf.Tensor:
    image_shape = tf.shape(image)
    flattened_image = tf.reshape(image, [-1, image_shape[-1]])
    alpha_channel = tf.expand_dims(flattened_image[..., 3], axis=-1)

    if isinstance(replace, int):
        replace_t = tf.fill([3], tf.cast(replace, image.dtype))
    else:
        replace_t = tf.cast(replace, image.dtype)

    replace_t = tf.concat([replace_t, tf.ones([1], image.dtype)], 0)

    flattened_image = tf.where(
        tf.equal(alpha_channel, 0),
        tf.ones_like(flattened_image, dtype=image.dtype) * replace_t,
        flattened_image,
    )
    image = tf.reshape(flattened_image, image_shape)
    return image[..., :3]


def _wrapped_rotate(image: tf.Tensor, degrees: float, replace: int) -> tf.Tensor:
    image = _rotate(_wrap(image), degrees=degrees)
    return _unwrap(image, replace)


def _translate_x(image: tf.Tensor, pixels: int, replace: int) -> tf.Tensor:
    image = _translate(_wrap(image), [-pixels, 0])
    return _unwrap(image, replace)


def _translate_y(image: tf.Tensor, pixels: int, replace: int) -> tf.Tensor:
    image = _translate(_wrap(image), [0, -pixels])
    return _unwrap(image, replace)


def _shear_x(image: tf.Tensor, level: float, replace: int) -> tf.Tensor:
    image = _transform(
        image=_wrap(image), transforms=[1.0, level, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    )
    return _unwrap(image, replace)


def _shear_y(image: tf.Tensor, level: float, replace: int) -> tf.Tensor:
    image = _transform(
        image=_wrap(image), transforms=[1.0, 0.0, 0.0, level, 1.0, 0.0, 0.0, 0.0]
    )
    return _unwrap(image, replace)


def _grayscale(image: tf.Tensor) -> tf.Tensor:
    return tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image))


def _color(image: tf.Tensor, factor: float) -> tf.Tensor:
    degenerate = _grayscale(image)
    return _blend(degenerate, image, factor)


def _contrast(image: tf.Tensor, factor: float) -> tf.Tensor:
    degenerate = tf.image.rgb_to_grayscale(image)
    mean = tf.reduce_mean(tf.cast(degenerate, tf.float32))
    degenerate = tf.ones_like(degenerate, dtype=tf.float32) * mean
    degenerate = tf.clip_by_value(degenerate, 0.0, 255.0)
    degenerate = tf.image.grayscale_to_rgb(tf.cast(degenerate, tf.uint8))
    return _blend(degenerate, image, factor)


def _brightness(image: tf.Tensor, factor: float) -> tf.Tensor:
    degenerate = tf.zeros_like(image)
    return _blend(degenerate, image, factor)


def _clip_bbox(min_y, min_x, max_y, max_x):
    min_y = tf.clip_by_value(min_y, 0.0, 1.0)
    min_x = tf.clip_by_value(min_x, 0.0, 1.0)
    max_y = tf.clip_by_value(max_y, 0.0, 1.0)
    max_x = tf.clip_by_value(max_x, 0.0, 1.0)
    return min_y, min_x, max_y, max_x


def _check_bbox_area(min_y, min_x, max_y, max_x, delta=0.05):
    height = max_y - min_y
    width = max_x - min_x

    def _adjust(min_coord, max_coord):
        new_min = tf.maximum(min_coord - delta / 2, 0.0)
        new_max = tf.minimum(max_coord + delta / 2, 1.0)
        return new_min, new_max

    min_y, max_y = tf.cond(
        tf.equal(height, 0.0),
        lambda: _adjust(min_y, max_y),
        lambda: (min_y, max_y),
    )
    min_x, max_x = tf.cond(
        tf.equal(width, 0.0),
        lambda: _adjust(min_x, max_x),
        lambda: (min_x, max_x),
    )
    return min_y, min_x, max_y, max_x


def _rotate_bbox(bbox, image_height, image_width, degrees):
    image_height, image_width = (
        tf.cast(image_height, tf.float32),
        tf.cast(image_width, tf.float32),
    )
    degrees_to_radians = math.pi / 180.0
    radians = degrees * degrees_to_radians

    min_y = -tf.cast(image_height * (bbox[0] - 0.5), tf.int32)
    min_x = tf.cast(image_width * (bbox[1] - 0.5), tf.int32)
    max_y = -tf.cast(image_height * (bbox[2] - 0.5), tf.int32)
    max_x = tf.cast(image_width * (bbox[3] - 0.5), tf.int32)
    coordinates = tf.stack(
        [[min_y, min_x], [min_y, max_x], [max_y, min_x], [max_y, max_x]]
    )
    coordinates = tf.cast(coordinates, tf.float32)

    rotation_matrix = tf.stack(
        [
            [tf.cos(radians), tf.sin(radians)],
            [-tf.sin(radians), tf.cos(radians)],
        ]
    )
    new_coords = tf.cast(
        tf.matmul(rotation_matrix, tf.transpose(coordinates)), tf.int32
    )

    min_y = -(tf.cast(tf.reduce_max(new_coords[0, :]), tf.float32) / image_height - 0.5)
    min_x = tf.cast(tf.reduce_min(new_coords[1, :]), tf.float32) / image_width + 0.5
    max_y = -(tf.cast(tf.reduce_min(new_coords[0, :]), tf.float32) / image_height - 0.5)
    max_x = tf.cast(tf.reduce_max(new_coords[1, :]), tf.float32) / image_width + 0.5

    min_y, min_x, max_y, max_x = _clip_bbox(min_y, min_x, max_y, max_x)
    min_y, min_x, max_y, max_x = _check_bbox_area(min_y, min_x, max_y, max_x)
    return tf.stack([min_y, min_x, max_y, max_x])


def _rotate_with_bboxes(image, bboxes, degrees, replace):
    image = _wrapped_rotate(image, degrees, replace)
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]
    def wrapped_rotate_bbox(bbox):
        return _rotate_bbox(bbox, image_height, image_width, degrees)

    bboxes = tf.map_fn(wrapped_rotate_bbox, bboxes)
    return image, bboxes


def _shear_bbox(bbox, image_height, image_width, level, shear_horizontal):
    image_height, image_width = (
        tf.cast(image_height, tf.float32),
        tf.cast(image_width, tf.float32),
    )
    min_y = tf.cast(image_height * bbox[0], tf.int32)
    min_x = tf.cast(image_width * bbox[1], tf.int32)
    max_y = tf.cast(image_height * bbox[2], tf.int32)
    max_x = tf.cast(image_width * bbox[3], tf.int32)
    coordinates = tf.stack(
        [[min_y, min_x], [min_y, max_x], [max_y, min_x], [max_y, max_x]]
    )
    coordinates = tf.cast(coordinates, tf.float32)

    if shear_horizontal:
        translation_matrix = tf.stack([[1, 0], [-level, 1]])
    else:
        translation_matrix = tf.stack([[1, -level], [0, 1]])
    translation_matrix = tf.cast(translation_matrix, tf.float32)
    new_coords = tf.cast(
        tf.matmul(translation_matrix, tf.transpose(coordinates)), tf.int32
    )

    min_y = tf.cast(tf.reduce_min(new_coords[0, :]), tf.float32) / image_height
    min_x = tf.cast(tf.reduce_min(new_coords[1, :]), tf.float32) / image_width
    max_y = tf.cast(tf.reduce_max(new_coords[0, :]), tf.float32) / image_height
    max_x = tf.cast(tf.reduce_max(new_coords[1, :]), tf.float32) / image_width

    min_y, min_x, max_y, max_x = _clip_bbox(min_y, min_x, max_y, max_x)
    min_y, min_x, max_y, max_x = _check_bbox_area(min_y, min_x, max_y, max_x)
    return tf.stack([min_y, min_x, max_y, max_x])


def _shear_with_bboxes(image, bboxes, level, replace, shear_horizontal):
    if shear_horizontal:
        image = _shear_x(image, level, replace)
    else:
        image = _shear_y(image, level, replace)
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]
    def wrapped_shear_bbox(bbox):
        return _shear_bbox(
            bbox, image_height, image_width, level, shear_horizontal
        )

    bboxes = tf.map_fn(wrapped_shear_bbox, bboxes)
    return image, bboxes


def _shift_bbox(bbox, image_height, image_width, pixels, shift_horizontal):
    pixels = tf.cast(pixels, tf.int32)
    min_y = tf.cast(tf.cast(image_height, tf.float32) * bbox[0], tf.int32)
    min_x = tf.cast(tf.cast(image_width, tf.float32) * bbox[1], tf.int32)
    max_y = tf.cast(tf.cast(image_height, tf.float32) * bbox[2], tf.int32)
    max_x = tf.cast(tf.cast(image_width, tf.float32) * bbox[3], tf.int32)

    if shift_horizontal:
        min_x = tf.maximum(0, min_x - pixels)
        max_x = tf.minimum(image_width, max_x - pixels)
    else:
        min_y = tf.maximum(0, min_y - pixels)
        max_y = tf.minimum(image_height, max_y - pixels)

    min_y = tf.cast(min_y, tf.float32) / tf.cast(image_height, tf.float32)
    min_x = tf.cast(min_x, tf.float32) / tf.cast(image_width, tf.float32)
    max_y = tf.cast(max_y, tf.float32) / tf.cast(image_height, tf.float32)
    max_x = tf.cast(max_x, tf.float32) / tf.cast(image_width, tf.float32)

    min_y, min_x, max_y, max_x = _clip_bbox(min_y, min_x, max_y, max_x)
    min_y, min_x, max_y, max_x = _check_bbox_area(min_y, min_x, max_y, max_x)
    return tf.stack([min_y, min_x, max_y, max_x])


def _translate_bbox(image, bboxes, pixels, replace, shift_horizontal):
    if shift_horizontal:
        image = _translate_x(image, pixels, replace)
    else:
        image = _translate_y(image, pixels, replace)

    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]
    def wrapped_shift_bbox(bbox):
        return _shift_bbox(
            bbox, image_height, image_width, pixels, shift_horizontal
        )

    bboxes = tf.map_fn(wrapped_shift_bbox, bboxes)
    return image, bboxes


def _randomly_negate_tensor(tensor, seed):
    should_flip = tf.random.stateless_uniform([], seed=seed) > 0.5
    return tf.cond(should_flip, lambda: -tensor, lambda: tensor)


def to_4d(image: tf.Tensor) -> tf.Tensor:
    """Converts an input Tensor to 4 dimensions.

    4D image => [N, H, W, C] or [N, C, H, W]
    3D image => [1, H, W, C] or [1, C, H, W]
    2D image => [1, H, W, 1]

    Args:
      image: The 2/3/4D input tensor.

    Returns:
      A 4D image tensor.

    Raises:
      `TypeError` if `image` is not a 2/3/4D tensor.

    """
    shape = tf.shape(image)
    original_rank = tf.rank(image)
    left_pad = tf.cast(tf.less_equal(original_rank, 3), dtype=tf.int32)
    right_pad = tf.cast(tf.equal(original_rank, 2), dtype=tf.int32)
    new_shape = tf.concat(
        [
            tf.ones(shape=left_pad, dtype=tf.int32),
            shape,
            tf.ones(shape=right_pad, dtype=tf.int32),
        ],
        axis=0,
    )
    return tf.reshape(image, new_shape)


def from_4d(image: tf.Tensor, ndims: tf.Tensor) -> tf.Tensor:
    """Converts a 4D image back to `ndims` rank."""
    shape = tf.shape(image)
    begin = tf.cast(tf.less_equal(ndims, 3), dtype=tf.int32)
    end = 4 - tf.cast(tf.equal(ndims, 2), dtype=tf.int32)
    new_shape = shape[begin:end]
    return tf.reshape(image, new_shape)


@tf.function
def gaussian_filter2d(
    image: tf.Tensor,
    filter_shape: int = 3,
    sigma: float = 1.0,
    padding: str = "REFLECT",
    constant_values: Union[int, float] = 0,
    name: Optional[str] = None,
) -> tf.Tensor:
    """Performs Gaussian blur on image(s)."""
    with tf.name_scope(name or "gaussian_filter2d"):
        # Convert inputs to tensors
        image = tf.convert_to_tensor(image)

        # Handle filter_shape
        filter_h = filter_w = filter_shape

        # Handle sigma
        sigma_h = sigma_w = sigma

        # Input validation
        tf.debugging.assert_greater_equal(sigma_h, 0.0, "sigma_h must be >= 0")
        tf.debugging.assert_greater_equal(sigma_w, 0.0, "sigma_w must be >= 0")

        # Convert image to 4D
        original_ndims = tf.rank(image)
        image = to_4d(image)

        # Handle dtype
        orig_dtype = image.dtype
        if not image.dtype.is_floating:
            image = tf.cast(image, tf.float32)

        # Get number of channels
        channels = tf.shape(image)[3]

        # Create gaussian kernels
        sigma_t = tf.convert_to_tensor([sigma_h, sigma_w], dtype=image.dtype)

        gaussian_kernel_x = _get_gaussian_kernel(
            sigma_t[1], tf.cast(filter_w, tf.int32)
        )
        gaussian_kernel_y = _get_gaussian_kernel(
            sigma_t[0], tf.cast(filter_h, tf.int32)
        )

        # Create 2D kernel
        gaussian_kernel_2d = _get_gaussian_kernel_2d(
            gaussian_kernel_x, gaussian_kernel_y
        )

        # Reshape kernel for depthwise conv
        gaussian_kernel_2d = tf.reshape(gaussian_kernel_2d, [filter_h, filter_w, 1, 1])
        gaussian_kernel_2d = tf.tile(gaussian_kernel_2d, [1, 1, channels, 1])

        # Pad image
        image = _pad(
            image,
            [filter_h, filter_w],
            pad_mode=padding,
            constant_values=constant_values,
        )

        # Apply convolution
        output = tf.nn.depthwise_conv2d(
            input=image,
            filter=gaussian_kernel_2d,
            strides=[1, 1, 1, 1],
            padding="VALID",
        )

        # Convert back to original dimensions and dtype
        output = from_4d(output, original_ndims)
        return tf.cast(output, orig_dtype)

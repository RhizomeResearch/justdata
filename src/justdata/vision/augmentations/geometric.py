from typing import Tuple, Union

import tensorflow as tf

from justdata.vision.augmentations.registry import register_crop_strategy


@tf.function
def _transform_bboxes_for_crop(
    bboxes, y_float, x_float, crop_height, crop_width, original_height, original_width
):
    bboxes = tf.cast(bboxes, tf.float32)
    ymin_abs = bboxes[:, 0] * original_height
    xmin_abs = bboxes[:, 1] * original_width
    ymax_abs = bboxes[:, 2] * original_height
    xmax_abs = bboxes[:, 3] * original_width

    intersect_ymin = tf.maximum(ymin_abs, y_float)
    intersect_xmin = tf.maximum(xmin_abs, x_float)
    intersect_ymax = tf.minimum(ymax_abs, y_float + crop_height)
    intersect_xmax = tf.minimum(xmax_abs, x_float + crop_width)

    adjusted_ymin = intersect_ymin - y_float
    adjusted_xmin = intersect_xmin - x_float
    adjusted_ymax = intersect_ymax - y_float
    adjusted_xmax = intersect_xmax - x_float

    norm_ymin = adjusted_ymin / crop_height
    norm_xmin = adjusted_xmin / crop_width
    norm_ymax = adjusted_ymax / crop_height
    norm_xmax = adjusted_xmax / crop_width

    valid_indices = tf.where(
        (intersect_xmax > intersect_xmin) & (intersect_ymax > intersect_ymin)
    )

    transformed_bboxes = tf.gather_nd(
        tf.stack([norm_ymin, norm_xmin, norm_ymax, norm_xmax], axis=-1),
        valid_indices,
    )

    transformed_bboxes = tf.clip_by_value(transformed_bboxes, 0.0, 1.0)
    return transformed_bboxes, valid_indices


@tf.function
def random_resized_crop(
    image,
    size,
    seed,
    scale=(0.08, 1.0),
    ratio=(0.75, 1.3333333333333333),
    interpolation="bilinear",
    bboxes=None,
):
    """
    Random resized crop for tensorflow, similar to torchvision's RandomResizedCrop.

    Args:
        image: 3D tensor of shape [height, width, channels]
        size: tuple of (height, width) or int for square size
        scale: tuple of (min_scale, max_scale) for area ratio
        ratio: tuple of (min_ratio, max_ratio) for aspect ratio
        interpolation: interpolation method, one of 'bilinear', 'nearest', 'bicubic'
        bboxes: Optional tensor of shape [N, 4] with normalized coordinates [x_min, y_min, x_max, y_max]

    Returns:
        Cropped and resized image tensor
    """
    seeds = tf.random.split(seed, 4)

    if isinstance(size, int):
        size = (size, size)

    # Get image shape
    original_height = tf.cast(tf.shape(image)[0], tf.float32)
    original_width = tf.cast(tf.shape(image)[1], tf.float32)
    original_area = original_height * original_width

    # Get random area ratio
    target_area = original_area * tf.random.stateless_uniform(
        [], minval=scale[0], maxval=scale[1], seed=seeds[0]
    )

    # Get random aspect ratio
    log_ratio = (tf.math.log(ratio[0]), tf.math.log(ratio[1]))
    aspect_ratio = tf.math.exp(
        tf.random.stateless_uniform(
            [], minval=log_ratio[0], maxval=log_ratio[1], seed=seeds[1]
        )
    )

    # Calculate target height and width
    height = tf.sqrt(target_area / aspect_ratio)
    width = height * aspect_ratio

    # Clip dimensions to image size
    height = tf.minimum(height, original_height)
    width = tf.minimum(width, original_width)

    # Get random crop coordinates
    height_int = tf.cast(height, tf.int32)
    width_int = tf.cast(width, tf.int32)

    max_x = tf.maximum(tf.cast(original_width - width, tf.int32), 0)
    max_y = tf.maximum(tf.cast(original_height - height, tf.int32), 0)

    x = tf.random.stateless_uniform(
        [], minval=0, maxval=max_x + 1, dtype=tf.int32, seed=seeds[2]
    )
    y = tf.random.stateless_uniform(
        [], minval=0, maxval=max_y + 1, dtype=tf.int32, seed=seeds[3]
    )

    # Crop the image
    crop = tf.image.crop_to_bounding_box(image, y, x, height_int, width_int)

    # Resize to target size
    resized = tf.image.resize(crop, size, method=interpolation, antialias=True)

    if bboxes is not None:
        y_float = tf.cast(y, tf.float32)
        x_float = tf.cast(x, tf.float32)
        crop_height = tf.cast(height_int, tf.float32)
        crop_width = tf.cast(width_int, tf.float32)

        transformed_bboxes, valid_indices = _transform_bboxes_for_crop(
            bboxes,
            y_float,
            x_float,
            crop_height,
            crop_width,
            original_height,
            original_width,
        )
        return resized, transformed_bboxes, valid_indices

    return resized


@tf.function
def random_horizontal_flip(image, seed, p=0.5, bboxes=None):
    """
    Randomly flips an image horizontally with a given probability.
    Optionally transforms bounding boxes as well.

    Args:
        image: 3D tensor of shape [height, width, channels].
        p: Float, the probability of flipping the image.
        seed: tf.Tensor seed for stateless random operations.
        bboxes: Optional Nx4 tensor of bounding boxes [ymin, xmin, ymax, xmax]
                in normalized coordinates (relative to the image).

    Returns:
        If bboxes is None:
            The (potentially) flipped image tensor.
        If bboxes is not None:
            A tuple of (the (potentially) flipped image tensor,
                        the (potentially) transformed bboxes tensor).
    """
    should_flip = tf.random.stateless_uniform([], seed=seed) < p

    # Flip image if needed
    flipped_image = tf.cond(
        should_flip, lambda: tf.image.flip_left_right(image), lambda: image
    )

    if bboxes is None:
        return flipped_image
    else:
        bboxes = tf.cast(bboxes, tf.float32)

        # If flipping, transform bounding boxes
        # Original: [ymin, xmin, ymax, xmax]
        # Flipped:  [ymin, 1 - xmax, ymax, 1 - xmin]
        def flip_bboxes(b):
            ymin, xmin, ymax, xmax = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
            flipped_xmin = 1.0 - xmax
            flipped_xmax = 1.0 - xmin
            return tf.stack([ymin, flipped_xmin, ymax, flipped_xmax], axis=-1)

        transformed_bboxes = tf.cond(
            should_flip, lambda: flip_bboxes(bboxes), lambda: bboxes
        )
        return flipped_image, transformed_bboxes


@tf.function
def random_crop_with_pad(
    image: tf.Tensor,
    size: Union[Tuple[int, int], int],
    padding: int,
    seed,
    pad_mode: str = "REFLECT",
    bboxes=None,
):
    if isinstance(size, int):
        size = (size, size)

    seeds = tf.random.split(seed, 2)

    paddings = [[padding, padding], [padding, padding], [0, 0]]
    padded_image = tf.pad(image, paddings, mode=pad_mode)

    original_height = tf.cast(tf.shape(image)[0], tf.float32)
    original_width = tf.cast(tf.shape(image)[1], tf.float32)

    image_shape = tf.shape(padded_image)
    max_y = tf.maximum(0, image_shape[0] - size[0])
    max_x = tf.maximum(0, image_shape[1] - size[1])

    y = tf.random.stateless_uniform(
        [], minval=0, maxval=max_y + 1, dtype=tf.int32, seed=seeds[0]
    )
    x = tf.random.stateless_uniform(
        [], minval=0, maxval=max_x + 1, dtype=tf.int32, seed=seeds[1]
    )

    crop = tf.image.crop_to_bounding_box(padded_image, y, x, size[0], size[1])

    if bboxes is not None:
        y_float = tf.cast(y, tf.float32) - tf.cast(padding, tf.float32)
        x_float = tf.cast(x, tf.float32) - tf.cast(padding, tf.float32)
        crop_height = tf.cast(size[0], tf.float32)
        crop_width = tf.cast(size[1], tf.float32)

        transformed_bboxes, valid_indices = _transform_bboxes_for_crop(
            bboxes,
            y_float,
            x_float,
            crop_height,
            crop_width,
            original_height,
            original_width,
        )
        return crop, transformed_bboxes, valid_indices

    return crop


@tf.function
def resize_to_square(
    image: tf.Tensor,
    size: Union[Tuple[int, int], int],
    interpolation: str = "bilinear",
) -> tf.Tensor:
    if isinstance(size, int):
        size = (size, size)
    return tf.image.resize(
        tf.cast(image, tf.float32),
        size,
        method=interpolation,
        antialias=True,
    )


@tf.function
def random_rot90(image: tf.Tensor, seed) -> tf.Tensor:
    k = tf.random.stateless_uniform([], minval=0, maxval=4, dtype=tf.int32, seed=seed)
    return tf.image.rot90(image, k=k)


@register_crop_strategy("random_resized")
def _crop_random_resized(image, size, seed, interpolation="bilinear", **kwargs):
    s = tf.random.split(seed, 2)
    cropped = random_resized_crop(
        image, size=size, seed=s[0], interpolation=interpolation
    )
    return random_horizontal_flip(cropped, seed=s[1])


@register_crop_strategy("random_pad")
def _crop_random_pad(image, size, seed, padding=4, pad_mode="REFLECT", **kwargs):
    s = tf.random.split(seed, 2)
    cropped = random_crop_with_pad(
        image, size=size, padding=padding, pad_mode=pad_mode, seed=s[0]
    )
    return random_horizontal_flip(cropped, seed=s[1])


@register_crop_strategy("random_hflip")
def _crop_random_hflip(image, size, seed, **kwargs):
    return random_horizontal_flip(image, seed=seed)


@register_crop_strategy("resize_random_hflip")
def _crop_resize_random_hflip(
    image, size, seed, interpolation="bilinear", **kwargs
):
    s = tf.random.split(seed, 2)
    resized = resize_to_square(image, size=size, interpolation=interpolation)
    return random_horizontal_flip(resized, seed=s[1])


@register_crop_strategy("random_rot90_hflip")
def _crop_random_rot90_hflip(image, size, seed, **kwargs):
    s = tf.random.split(seed, 2)
    rotated = random_rot90(image, seed=s[0])
    return random_horizontal_flip(rotated, seed=s[1])


@register_crop_strategy("none")
def _crop_none(image, size, seed, **kwargs):
    return image

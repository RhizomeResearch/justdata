import tensorflow as tf


@tf.function
def nhwc_to_nchw(image: tf.Tensor) -> tf.Tensor:
    if len(image.shape) == 4:
        return tf.transpose(image, [0, 3, 1, 2])
    elif len(image.shape) == 3:
        return tf.transpose(image, [2, 0, 1])
    else:
        raise NotImplementedError


@tf.function
def normalize(image, mean, std):
    """
    Normalizes an image tensor of any rank, assuming channels are the last dimension.

    Args:
        image: Input image tensor (any rank, channel-last).
               Expected to have integer values in the range [0, 255].
        mean: List, tuple, or 1D tensor of channel means.
        std: List, tuple, or 1D tensor of channel standard deviations.

    Returns:
        Normalized image tensor (tf.float32).
    """
    image = tf.cast(image, tf.float32)
    image /= 255.0

    img_mean = tf.constant(mean, dtype=tf.float32)
    img_std = tf.constant(std, dtype=tf.float32)

    rank = tf.rank(image)
    num_channels = tf.shape(image)[-1]
    broadcast_shape = tf.concat(
        [tf.ones(rank - 1, dtype=tf.int32), [num_channels]], axis=0
    )

    img_mean = tf.reshape(img_mean, broadcast_shape)
    img_std = tf.reshape(img_std, broadcast_shape)

    normalized_image = (image - img_mean) / (img_std + 1e-8)
    return normalized_image


@tf.function
def resize_short_side(
    image: tf.Tensor, target_size: int, method: str = "bilinear"
) -> tf.Tensor:
    """Resizes image so the shorter side matches target_size, preserving aspect ratio."""
    shape = tf.shape(image)
    height, width = tf.cast(shape[0], tf.float32), tf.cast(shape[1], tf.float32)

    scale = tf.cast(target_size, tf.float32) / tf.minimum(height, width)

    new_height = tf.cast(tf.round(height * scale), tf.int32)
    new_width = tf.cast(tf.round(width * scale), tf.int32)

    return tf.image.resize(image, [new_height, new_width], method=method)


@tf.function
def center_crop(image: tf.Tensor, crop_size: int) -> tf.Tensor:
    """Crops the center of the image to the specific size."""
    shape = tf.shape(image)
    height, width = shape[0], shape[1]

    offset_height = (height - crop_size) // 2
    offset_width = (width - crop_size) // 2

    return tf.image.crop_to_bounding_box(
        image, offset_height, offset_width, crop_size, crop_size
    )


@tf.function
def resize_image(
    image: tf.Tensor,
    image_size: int,
    resize_size: int | None = 256,
    method: str = "bilinear",
):
    image = tf.cast(image, tf.float32)

    if resize_size is not None:
        image = resize_short_side(image, target_size=resize_size, method=method)
        image = center_crop(image, crop_size=image_size)
    else:
        image = tf.image.resize(image, [image_size, image_size], method=method)

    return image


@tf.function
def pad_to_patch_multiple(
    image: tf.Tensor,
    patch_size: int = 14,
    pad_mode: str = "CONSTANT",
) -> tf.Tensor:
    """Pads an image so height and width are multiples of ``patch_size``.

    Padding is applied to the bottom and right edges only, suitable for
    dense ViT evaluation where boundary pixels must not be dropped.

    Args:
        image: 3-D tensor ``[H, W, C]``.
        patch_size: ViT patch size (e.g. 14 or 16).
        pad_mode: One of ``CONSTANT``, ``REFLECT``, ``SYMMETRIC``.

    Returns:
        Padded image tensor.
    """
    shape = tf.shape(image)
    h, w = shape[0], shape[1]

    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size

    paddings = [[0, pad_h], [0, pad_w], [0, 0]]
    return tf.pad(image, paddings, mode=pad_mode)

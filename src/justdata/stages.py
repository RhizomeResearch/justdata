import tensorflow as tf

from justdata.augmentations.registry import get_crop_strategy
from justdata.transforms import nhwc_to_nchw, normalize, resize_image


def normalize_image_format(sample: dict, *, image_key: str = "image") -> dict:
    """Rank fix, CHW->HWC, grayscale->RGB, RGBA->RGB."""
    image = sample[image_key]

    image = tf.cond(
        tf.equal(tf.rank(image), 2),
        lambda: tf.expand_dims(image, -1),
        lambda: image,
    )

    shape = tf.shape(image)
    is_chw = tf.logical_and(shape[0] <= 4, shape[-1] > 4)
    image = tf.cond(is_chw, lambda: tf.transpose(image, [1, 2, 0]), lambda: image)

    shape = tf.shape(image)
    image = tf.cond(
        tf.equal(shape[-1], 1),
        lambda: tf.image.grayscale_to_rgb(image),
        lambda: image,
    )

    shape = tf.shape(image)
    image = tf.cond(shape[-1] > 3, lambda: image[..., :3], lambda: image)

    image.set_shape([None, None, 3])
    return sample | {image_key: image}


def resize_and_normalize(
    sample: dict,
    *,
    image_keys: list[str],
    image_size: int,
    resize_size: int | None = None,
    normalize_image: bool = True,
    normalization_params: tuple | None = None,
    permute: bool = True,
) -> dict:
    """Resize, [0,1]->normalized, optionally HWC->CHW."""
    res = {}
    for key in image_keys:
        if key not in sample:
            continue
        image = sample[key]

        if "crops" not in key:
            image = resize_image(
                image,
                image_size=image_size,
                resize_size=resize_size,
            )
        if normalize_image:
            if normalization_params is None:
                raise ValueError(
                    "`normalization_params` needs to be provided if `normalize_image` is True"
                )
            image = normalize(image, *normalization_params)
        if permute:
            image = nhwc_to_nchw(image)

        res[key] = image
    return sample | res


def apply_crop_strategy(
    sample: dict, *, image_key: str, crop_type: str, seed, **kwargs
) -> dict:
    """Dispatches to registered crop strategy."""
    crop_fn = get_crop_strategy(crop_type)
    cropped = crop_fn(sample[image_key], seed=seed, **kwargs)
    return sample | {image_key: cropped}


def co_transform(
    sample: dict, *, image_key: str, mask_key: str, transform_fn, seed
) -> dict:
    """Applies the same spatial transform to both image and mask."""
    image = transform_fn(sample[image_key], seed)
    mask = transform_fn(sample[mask_key], seed)
    return sample | {image_key: image, mask_key: mask}

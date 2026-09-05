from dataclasses import dataclass
from typing import Literal

import tensorflow as tf

from justdata.vision.augmentations.registry import get_crop_strategy
from justdata.vision.transforms import nhwc_to_nchw, normalize, resize_image


@dataclass(frozen=True)
class EvalViewConfig:
    image_size: int
    mode: Literal["center_crop", "resize_crop", "multi_crop"] = "center_crop"
    resize_size: int | None = None
    num_crops: int = 1
    include_flip: bool = False
    interpolation: str = "bilinear"

    def __post_init__(self) -> None:
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if self.mode not in {"center_crop", "resize_crop", "multi_crop"}:
            raise ValueError("mode must be center_crop, resize_crop, or multi_crop")
        if self.resize_size is not None and self.resize_size <= 0:
            raise ValueError("resize_size must be positive when set")
        if self.num_crops <= 0:
            raise ValueError("num_crops must be positive")


def _eval_view_config(config: EvalViewConfig | dict) -> EvalViewConfig:
    if isinstance(config, EvalViewConfig):
        return config
    return EvalViewConfig(**dict(config))


def _resize_for_eval_view(
    image: tf.Tensor,
    config: EvalViewConfig,
) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.cast(image, tf.float32)
    shape = tf.shape(image)
    h = tf.cast(shape[0], tf.float32)
    w = tf.cast(shape[1], tf.float32)
    min_side = tf.minimum(h, w)

    if config.mode == "resize_crop" or config.resize_size is not None:
        resize_size = config.resize_size
        if resize_size is None:
            resize_size = int(config.image_size / 0.875)
        scale = tf.cast(resize_size, tf.float32) / min_side

        new_h = tf.cast(tf.round(h * scale), tf.int32)
        new_w = tf.cast(tf.round(w * scale), tf.int32)
        image = tf.image.resize(
            image,
            [new_h, new_w],
            method=config.interpolation,
        )
        return image, scale

    min_scale = tf.cast(config.image_size, tf.float32) / min_side
    should_resize = min_side < tf.cast(config.image_size, tf.float32)
    scale = tf.cond(
        should_resize,
        lambda: min_scale,
        lambda: tf.constant(1.0, dtype=tf.float32),
    )

    def _resize() -> tf.Tensor:
        new_h = tf.cast(tf.round(h * scale), tf.int32)
        new_w = tf.cast(tf.round(w * scale), tf.int32)
        return tf.image.resize(
            image,
            [new_h, new_w],
            method=config.interpolation,
        )

    return tf.cond(should_resize, _resize, lambda: image), scale


def _vision_crop_boxes(image: tf.Tensor, config: EvalViewConfig) -> tf.Tensor:
    shape = tf.shape(image)
    crop_size = tf.cast(config.image_size, tf.int32)
    max_y = tf.maximum(shape[0] - crop_size, 0)
    max_x = tf.maximum(shape[1] - crop_size, 0)
    center_y = max_y // 2
    center_x = max_x // 2

    if config.mode != "multi_crop" or config.num_crops == 1:
        return tf.reshape(
            tf.stack([center_y, center_x, crop_size, crop_size]),
            [1, 4],
        )

    if config.num_crops == 5:
        return tf.stack(
            [
                tf.stack([0, 0, crop_size, crop_size]),
                tf.stack([0, max_x, crop_size, crop_size]),
                tf.stack([max_y, 0, crop_size, crop_size]),
                tf.stack([max_y, max_x, crop_size, crop_size]),
                tf.stack([center_y, center_x, crop_size, crop_size]),
            ],
            axis=0,
        )

    ys = tf.cast(
        tf.round(tf.linspace(0.0, tf.cast(max_y, tf.float32), config.num_crops)),
        tf.int32,
    )
    xs = tf.cast(
        tf.round(tf.linspace(0.0, tf.cast(max_x, tf.float32), config.num_crops)),
        tf.int32,
    )
    sizes = tf.fill([config.num_crops], crop_size)
    return tf.stack([ys, xs, sizes, sizes], axis=1)


def apply_eval_views(
    sample: dict,
    *,
    config: EvalViewConfig | dict,
    image_key: str = "image",
) -> dict:
    config = _eval_view_config(config)
    image, scale = _resize_for_eval_view(sample[image_key], config)
    crop_boxes = _vision_crop_boxes(image, config)

    def _crop(box: tf.Tensor) -> tf.Tensor:
        return tf.image.crop_to_bounding_box(
            image,
            box[0],
            box[1],
            box[2],
            box[3],
        )

    if config.mode != "multi_crop" or config.num_crops == 1:
        crops = _crop(crop_boxes[0])[tf.newaxis, ...]
    else:
        crops = tf.map_fn(_crop, crop_boxes, fn_output_signature=image.dtype)
    flip = tf.zeros([tf.shape(crops)[0]], dtype=tf.bool)

    if config.include_flip:
        flipped = tf.image.flip_left_right(crops)
        crops = tf.concat([crops, flipped], axis=0)
        crop_boxes = tf.concat([crop_boxes, crop_boxes], axis=0)
        flip = tf.concat([flip, tf.ones_like(flip)], axis=0)

    num_views = tf.shape(crops)[0]
    view_metadata = {
        "crop_box": crop_boxes,
        "scale": tf.fill([num_views], scale),
        "flip": flip,
        "view_index": tf.range(num_views, dtype=tf.int32),
    }
    return sample | {image_key: crops, "view_metadata": view_metadata}


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


def _per_image_channel_standardize(image: tf.Tensor) -> tf.Tensor:
    image = tf.cast(image, tf.float32) / 255.0
    if image.shape.rank == 4:
        axes = [1, 2]
    else:
        axes = [0, 1]

    mean = tf.reduce_mean(image, axis=axes, keepdims=True)
    variance = tf.reduce_mean(tf.square(image - mean), axis=axes, keepdims=True)
    std = tf.sqrt(variance)
    centered = image - mean
    has_variance = std > 1e-6
    safe_std = tf.where(has_variance, std, tf.ones_like(std))
    return tf.where(has_variance, centered / safe_std, tf.zeros_like(centered))


def _normalize_image(
    image: tf.Tensor,
    *,
    normalization_mode: Literal["mean_std", "per_image"],
    normalization_params: tuple | None,
) -> tf.Tensor:
    if normalization_mode == "per_image":
        return _per_image_channel_standardize(image)
    if normalization_mode != "mean_std":
        raise ValueError("normalization_mode must be 'mean_std' or 'per_image'")
    if normalization_params is None:
        raise ValueError(
            "`normalization_params` needs to be provided when "
            "`normalization_mode='mean_std'`"
        )
    return normalize(image, *normalization_params)


def resize_and_normalize(
    sample: dict,
    *,
    image_keys: list[str],
    image_size: int,
    resize_size: int | None = None,
    normalize_image: bool = True,
    normalization_mode: Literal["mean_std", "per_image"] = "mean_std",
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
            image = _normalize_image(
                image,
                normalization_mode=normalization_mode,
                normalization_params=normalization_params,
            )
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

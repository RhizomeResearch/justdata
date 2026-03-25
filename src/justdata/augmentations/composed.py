from typing import Tuple, Union

import tensorflow as tf

from justdata.augmentations.color import color_jitter, gaussian_blur, solarize
from justdata.augmentations.geometric import random_horizontal_flip, random_resized_crop
from justdata.encodings.heatmaps import bboxes_to_gaussian_heatmaps


@tf.function
def create_global_crops(
    image: tf.Tensor,
    crops_number: int,
    size: Union[Tuple[int], int],
    scale: Tuple[float, float],
    seed,
    image_label: tf.Tensor | None = None,
    image_label_interp: str = "nearest",
    brightness: float = 0.4,
    contrast: float = 0.4,
    saturation: float = 0.2,
    hue: float = 0.1,
    p_grayscale: float = 0.2,
    p_gaussian_blur: float = 1.0,
    p_solarize: float = 0.2,
    bboxes: tf.Tensor | None = None,
    labels: tf.Tensor | None = None,
    num_classes: int | None = None,
) -> tf.Tensor | Tuple[tf.Tensor, ...]:
    if bboxes is not None:
        if labels is None or num_classes is None:
            raise ValueError(
                "You must provide `labels` and `num_classes` when passing `bboxes`."
            )
    seed_crops = tf.random.split(seed, crops_number)
    should_transform_label = image_label is not None

    crops = []
    crops_heatmaps = []
    label_crops = [] if should_transform_label else None
    for i in range(crops_number):
        seeds = tf.random.split(seed_crops[i], 5)

        if bboxes is not None:
            crop, crop_bboxes, crop_validbboxes = random_resized_crop(
                image, size=size, scale=scale, bboxes=bboxes, seed=seeds[0]
            )
            crop, crop_bboxes = random_horizontal_flip(
                crop, bboxes=crop_bboxes, seed=seeds[1]
            )
            crop_heatmaps = bboxes_to_gaussian_heatmaps(
                crop, crop_bboxes, tf.gather_nd(labels, crop_validbboxes), num_classes
            )
        else:
            crop = random_resized_crop(image, size=size, scale=scale, seed=seeds[0])
            crop = random_horizontal_flip(crop, seed=seeds[1])

        crop = color_jitter(
            crop,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
            p_grayscale=p_grayscale,
            seed=seeds[2],
        )
        crop = gaussian_blur(crop, p=p_gaussian_blur, seed=seeds[3])
        crop = solarize(crop, p=p_solarize, seed=seeds[4])

        crops.append(crop)
        if bboxes is not None:
            crops_heatmaps.append(crop_heatmaps)

        if should_transform_label:
            label = random_resized_crop(
                image_label,
                size=size,
                scale=scale,
                interpolation=image_label_interp,
                seed=seeds[0],
            )
            label = random_horizontal_flip(label, seed=seeds[1])

            label_crops.append(label)

    crops = tf.stack(crops)

    if bboxes is not None:
        return crops, tf.stack(crops_heatmaps)

    if should_transform_label:
        return crops, tf.stack(label_crops)

    return crops


@tf.function
def create_local_crops(
    image: tf.Tensor,
    crops_number: int,
    size: Union[Tuple[int, int], int],
    scale: Tuple[float, float],
    seed,
    brightness: float = 0.4,
    contrast: float = 0.4,
    saturation: float = 0.2,
    hue: float = 0.1,
    p_grayscale: float = 0.2,
) -> tf.Tensor:
    seed_crops = tf.random.split(seed, crops_number)

    crops = []
    for i in range(crops_number):
        seeds = tf.random.split(seed_crops[i], 4)

        crop = random_resized_crop(image, size=size, scale=scale, seed=seeds[0])
        crop = random_horizontal_flip(crop, seed=seeds[1])
        crop = color_jitter(
            crop,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
            p_grayscale=p_grayscale,
            seed=seeds[2],
        )
        crop = gaussian_blur(crop, p=0.5, seed=seeds[3])
        crops.append(crop)
    return tf.stack(crops)

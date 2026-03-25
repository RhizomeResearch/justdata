from typing import Optional

import tensorflow as tf

from justdata.vision.augmentations.registry import register_augment_strategy
from justdata.vision.utils import gaussian_filter2d


@tf.function
def color_jitter(
    image: tf.Tensor,
    seed,
    brightness: Optional[float] = 0.0,
    contrast: Optional[float] = 0.0,
    saturation: Optional[float] = 0.0,
    hue: Optional[float] = 0.0,
    p: Optional[float] = 1.0,
    p_grayscale: Optional[float] = 0.0,
) -> tf.Tensor:
    """Applies color jitter and random grayscale to an image.

    ColorJitter is applied with probability ``p``, then RandomGrayscale is
    applied independently with probability ``p_grayscale``.  The two
    operations are *not* mutually exclusive.

    Args:
      image (tf.Tensor): Of shape [height, width, 3] and type uint8.
      seed: Random seed tensor.
      brightness (float, optional): Magnitude for brightness jitter. Defaults to
        0.
      contrast (float, optional): Magnitude for contrast jitter. Defaults to 0.
      saturation (float, optional): Magnitude for saturation jitter. Defaults to
        0.
      hue (float, optional): Magnitude for hue jitter. Defaults to 0.
      p (float, optional): Probability of applying color jitter. Defaults to 1.0.
      p_grayscale (float, optional): Probability to convert the image to
        grayscale. Defaults to 0.

    Returns:
      tf.Tensor: The augmented ``image`` of type uint8.
    """
    seeds = tf.random.split(seed, 6)

    def apply_color_jitter(img):
        img = tf.image.stateless_random_brightness(img, brightness, seed=seeds[0])
        img = tf.image.stateless_random_contrast(
            img, 1 - contrast, 1 + contrast, seed=seeds[1]
        )
        img = tf.image.stateless_random_saturation(
            img, 1 - saturation, 1 + saturation, seed=seeds[2]
        )
        img = tf.image.stateless_random_hue(img, hue, seed=seeds[3])
        return tf.clip_by_value(img, 0, 255)

    def apply_grayscale(img):
        gray = tf.image.rgb_to_grayscale(img)
        return tf.image.grayscale_to_rgb(gray)

    # Apply color jitter with probability p
    should_jitter = tf.random.stateless_uniform([], seed=seeds[4]) < p
    result = tf.cond(should_jitter, lambda: apply_color_jitter(image), lambda: image)

    # Apply grayscale independently with probability p_grayscale
    should_grayscale = tf.random.stateless_uniform([], seed=seeds[5]) < p_grayscale
    result = tf.cond(should_grayscale, lambda: apply_grayscale(result), lambda: result)

    return result


@tf.function
def gaussian_blur(
    image: tf.Tensor,
    seed,
    p: float = 0.5,
) -> tf.Tensor:
    seeds = tf.random.split(seed, 2)

    random_value = tf.random.stateless_uniform([], seed=seeds[0])
    sigma = tf.random.stateless_uniform([], minval=0.1, maxval=2.0, seed=seeds[1])

    should_blur = tf.less(random_value, p)

    def do_blur():
        return gaussian_filter2d(image, filter_shape=9, sigma=sigma)

    return tf.cond(should_blur, do_blur, lambda: image)


@tf.function
def solarize(
    image: tf.Tensor,
    seed,
    p: float = 0.2,
    normalized_image: bool = False,  # i.e., pixels \in [0,1]
) -> tf.Tensor:
    random_value = tf.random.stateless_uniform([], seed=seed)

    should_solarize = tf.less(random_value, p)

    threshold = 0.5 if normalized_image else 127.5
    max_value = 1.0 if normalized_image else 255.0

    max_value_t = tf.cast(max_value, image.dtype)
    threshold_t = tf.cast(threshold, image.dtype)
    solarized = tf.where(image < threshold_t, image, max_value_t - image)

    return tf.cond(should_solarize, lambda: solarized, lambda: image)


@register_augment_strategy("color_jitter")
def _aug_color_jitter(image, seed, **kwargs):
    return color_jitter(image, seed, **kwargs)

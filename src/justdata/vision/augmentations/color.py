import math
from typing import Optional

import tensorflow as tf

from justdata.vision.augmentations.registry import register_augment_strategy
from justdata.vision.utils import _brightness, _color, _contrast, gaussian_filter2d


@tf.function
def photometric_distortion(
    image: tf.Tensor,
    seed,
    brightness: float = 32 / 255,
    contrast: float = 0.5,
    saturation: float = 0.5,
    hue: float = 0.05,
    probability: float = 0.5,
) -> tf.Tensor:
    """Stateless, float32 RGB distortion for dense segmentation.

    Input and output use the unnormalized [0, 255] RGB domain. The four
    operations are independently gated; contrast is placed on either side of
    saturation and hue with equal probability.
    """
    for name, value in (
        ("brightness", brightness),
        ("contrast", contrast),
        ("saturation", saturation),
        ("hue", hue),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"{name} must be a finite nonnegative number")
    if hue > 0.5:
        raise ValueError("hue must be at most 0.5")
    if (
        isinstance(probability, bool)
        or not isinstance(probability, (int, float))
        or not math.isfinite(probability)
        or not 0 <= probability <= 1
    ):
        raise ValueError("probability must be in [0, 1]")

    seeds = tf.random.experimental.stateless_split(seed, 9)
    image = tf.cast(image, tf.float32)
    image = tf.ensure_shape(image, [None, None, 3])
    tf.debugging.assert_greater_equal(image, 0.0)
    tf.debugging.assert_less_equal(image, 255.0)
    gates = [tf.random.stateless_uniform([], seeds[i]) < probability for i in range(4)]
    factors = [
        tf.random.stateless_uniform(
            [], seeds[i + 4], minval=max(0.0, 1.0 - amount), maxval=1.0 + amount
        )
        if amount > 0
        else tf.constant(1.0)
        for i, amount in enumerate((brightness, contrast, saturation))
    ]
    hue_delta = (
        tf.random.stateless_uniform([], seeds[7], minval=-hue, maxval=hue)
        if hue > 0
        else tf.constant(0.0)
    )
    contrast_first = tf.random.stateless_uniform([], seeds[8]) < 0.5

    def clipped(value):
        return tf.clip_by_value(value, 0.0, 255.0)

    image = tf.cond(gates[0], lambda: clipped(image * factors[0]), lambda: image)

    def adjust_contrast(value):
        gray = tf.reduce_sum(value * [0.2989, 0.5870, 0.1140], axis=-1)
        mean = tf.reduce_mean(gray)
        return clipped(mean + factors[1] * (value - mean))

    def adjust_saturation_and_hue(value):
        def adjust_saturation():
            gray = tf.reduce_sum(
                value * [0.2989, 0.5870, 0.1140], axis=-1, keepdims=True
            )
            return clipped(gray + factors[2] * (value - gray))

        value = tf.cond(gates[2], adjust_saturation, lambda: value)

        def adjust_hue():
            hsv = tf.image.rgb_to_hsv(value / 255.0)
            shifted = tf.concat(
                [tf.math.floormod(hsv[..., :1] + hue_delta, 1.0), hsv[..., 1:]], axis=-1
            )
            return clipped(tf.image.hsv_to_rgb(shifted) * 255.0)

        return tf.cond(gates[3], adjust_hue, lambda: value)

    def contrast_then_color():
        adjusted = tf.cond(gates[1], lambda: adjust_contrast(image), lambda: image)
        return adjust_saturation_and_hue(adjusted)

    def color_then_contrast():
        adjusted = adjust_saturation_and_hue(image)
        return tf.cond(gates[1], lambda: adjust_contrast(adjusted), lambda: adjusted)

    return tf.cond(contrast_first, contrast_then_color, color_then_contrast)


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
      image (tf.Tensor): Of shape [height, width, 3] in the image domain
        ([0, 255]), either uint8 or float from resize operations.
      seed: Random seed tensor.
      brightness (float, optional): Factor magnitude for brightness jitter.
        Defaults to 0.
      contrast (float, optional): Factor magnitude for contrast jitter. Defaults
        to 0.
      saturation (float, optional): Factor magnitude for saturation jitter.
        Defaults to 0.
      hue (float, optional): Hue delta magnitude. Defaults to 0.
      p (float, optional): Probability of applying color jitter. Defaults to 1.0.
      p_grayscale (float, optional): Probability to convert the image to
        grayscale. Defaults to 0.

    Returns:
      tf.Tensor: The augmented ``image`` with the input dtype preserved.
    """
    seeds = tf.random.split(seed, 6)
    input_image_type = image.dtype

    def magnitude(value):
        if value is None:
            return tf.constant(0.0, dtype=tf.float32)
        return tf.cast(value, dtype=tf.float32)

    def image_to_uint8(img):
        if img.dtype == tf.uint8:
            return img
        img = tf.clip_by_value(img, 0.0, 255.0)
        return tf.cast(img, dtype=tf.uint8)

    def restore_dtype(img):
        return tf.cast(img, dtype=input_image_type)

    def sample_factor(amount, factor_seed):
        amount = magnitude(amount)

        def jitter_factor():
            lower = tf.maximum(tf.constant(0.0, dtype=tf.float32), 1.0 - amount)
            upper = 1.0 + amount
            return tf.random.stateless_uniform(
                [], minval=lower, maxval=upper, seed=factor_seed
            )

        return tf.cond(amount > 0.0, jitter_factor, lambda: tf.constant(1.0))

    def apply_hue(img, amount, hue_seed):
        amount = magnitude(amount)

        def jitter_hue():
            delta = tf.random.stateless_uniform(
                [], minval=-amount, maxval=amount, seed=hue_seed
            )
            img_f = tf.cast(img, tf.float32) / 255.0
            hsv = tf.image.rgb_to_hsv(img_f)
            hue_channel = tf.math.floormod(hsv[..., 0] + delta, 1.0)
            hsv = tf.concat([hue_channel[..., tf.newaxis], hsv[..., 1:]], axis=-1)
            rgb = tf.image.hsv_to_rgb(hsv)
            rgb = tf.clip_by_value(rgb * 255.0, 0.0, 255.0)
            return tf.cast(rgb, dtype=tf.uint8)

        return tf.cond(amount > 0.0, jitter_hue, lambda: img)

    def apply_color_jitter(img):
        should_transform = tf.logical_or(
            tf.logical_or(magnitude(brightness) > 0.0, magnitude(contrast) > 0.0),
            tf.logical_or(magnitude(saturation) > 0.0, magnitude(hue) > 0.0),
        )

        def transform():
            jittered = image_to_uint8(img)
            jittered = _brightness(jittered, sample_factor(brightness, seeds[0]))
            jittered = _contrast(jittered, sample_factor(contrast, seeds[1]))
            jittered = _color(jittered, sample_factor(saturation, seeds[2]))
            jittered = apply_hue(jittered, hue, seeds[3])
            return restore_dtype(jittered)

        return tf.cond(should_transform, transform, lambda: img)

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

    def apply_solarize():
        return tf.where(image < threshold_t, image, max_value_t - image)

    if isinstance(p, (int, float)) and p >= 1.0:
        solarized = apply_solarize()
        return tf.cond(should_solarize, lambda: solarized, lambda: image)
    return tf.cond(should_solarize, apply_solarize, lambda: image)


@register_augment_strategy("color_jitter")
def _aug_color_jitter(image, seed, **kwargs):
    return color_jitter(image, seed, **kwargs)

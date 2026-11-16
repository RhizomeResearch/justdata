"""Float32 RGB colour-space conversions on values in ``[0, 1]``.

sRGB follows IEC 61966-2-1 with the D65 white point; CIELAB uses the same
white, so ``lab_to_rgb(rgb_to_lab(x))`` returns ``x`` up to float32 rounding.
``estimate_illuminant`` gives a frame's illuminant colour for chromatic
adaptation between frames.
"""

import tensorflow as tf

from justdata.vision.filters import gaussian_blur_hwc

_RGB_TO_XYZ = (
    (0.4124564, 0.3575761, 0.1804375),
    (0.2126729, 0.7151522, 0.0721750),
    (0.0193339, 0.1191920, 0.9503041),
)
_D65_WHITE_XYZ = (0.95047, 1.0, 1.08883)
_LAB_EPSILON = 216.0 / 24389.0
_LAB_KAPPA = 24389.0 / 27.0
MINKOWSKI_ORDER = 6
# Grey-Edge smoothing in pixels per 1024 pixels of image width.
GREY_EDGE_SIGMA_PER_1024_PX = 1.0
_CLIPPED_LEVEL = 254.0 / 255.0


def srgb_to_linear(image: tf.Tensor) -> tf.Tensor:
    image = tf.cast(image, tf.float32)
    return tf.where(
        image <= 0.04045,
        image / 12.92,
        tf.pow((tf.maximum(image, 0.04045) + 0.055) / 1.055, 2.4),
    )


def linear_to_srgb(image: tf.Tensor) -> tf.Tensor:
    image = tf.maximum(tf.cast(image, tf.float32), 0.0)
    return tf.where(
        image <= 0.0031308,
        image * 12.92,
        1.055 * tf.pow(tf.maximum(image, 0.0031308), 1.0 / 2.4) - 0.055,
    )


def rgb_to_lab(image: tf.Tensor) -> tf.Tensor:
    """Convert sRGB ``[..., 3]`` to CIELAB with L in ``[0, 100]``."""
    linear = srgb_to_linear(image)
    xyz = tf.linalg.matvec(tf.constant(_RGB_TO_XYZ, tf.float32), linear)
    ratio = xyz / tf.constant(_D65_WHITE_XYZ, tf.float32)
    f = tf.where(
        ratio > _LAB_EPSILON,
        tf.pow(tf.maximum(ratio, _LAB_EPSILON), 1.0 / 3.0),
        (_LAB_KAPPA * ratio + 16.0) / 116.0,
    )
    lightness = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return tf.stack([lightness, a, b], axis=-1)


def lab_to_rgb(lab: tf.Tensor) -> tf.Tensor:
    """Convert CIELAB ``[..., 3]`` to sRGB, clipped to ``[0, 1]``."""
    lab = tf.cast(lab, tf.float32)
    fy = (lab[..., 0] + 16.0) / 116.0
    fx = fy + lab[..., 1] / 500.0
    fz = fy - lab[..., 2] / 200.0
    f = tf.stack([fx, fy, fz], axis=-1)
    ratio = tf.where(
        tf.pow(f, 3.0) > _LAB_EPSILON,
        tf.pow(f, 3.0),
        (116.0 * f - 16.0) / _LAB_KAPPA,
    )
    xyz = ratio * tf.constant(_D65_WHITE_XYZ, tf.float32)
    xyz_to_rgb = tf.linalg.inv(tf.constant(_RGB_TO_XYZ, tf.float64))
    linear = tf.linalg.matvec(tf.cast(xyz_to_rgb, tf.float32), xyz)
    return tf.clip_by_value(linear_to_srgb(linear), 0.0, 1.0)


def estimate_illuminant(image: tf.Tensor, valid: tf.Tensor | None = None) -> tf.Tensor:
    """Linear-RGB illuminant of an sRGB image in ``[0, 1]``, colour and level.

    First-order Grey-Edge with Minkowski order 6 (van de Weijer, Gevers and
    Gijsenij, "Edge-Based Color Constancy", IEEE TIP 2007): the Minkowski mean
    of gradient magnitudes of the image smoothed by a Gaussian of
    ``width / 1024`` pixels, with gradients per ``width / 1024`` pixels so the
    level does not depend on resolution. Gradients make large uniform regions,
    such as open water or sky, count little, so the estimate follows the light
    and exposure rather than the dominant surface colour and brightness. Clipped pixels (a channel at or above 254/255)
    and their 3 x 3 neighbours, as in the authors' reference code, border
    pixels, and pixels outside the optional bool ``valid`` map are left out;
    excluding the whole smoothing reach instead drops the bright cloud and glare
    edges that carry most of the light's colour. A diagonal (von Kries) change of
    illumination or exposure scales the estimate by the same per-channel
    factors, so the ratio of two frames' estimates adapts colours from one frame
    to the other.
    """
    image = tf.cast(image, tf.float32)
    if image.shape.rank != 3 or image.shape[1] is None:
        raise ValueError("estimate_illuminant needs an HWC image with a static width.")
    pixels_per_reference = image.shape[1] / 1024.0
    sigma = GREY_EDGE_SIGMA_PER_1024_PX * pixels_per_reference
    smooth = gaussian_blur_hwc(srgb_to_linear(image), sigma, sigma)
    half_step = 0.5 * pixels_per_reference
    gradient_y = half_step * (smooth[2:, 1:-1] - smooth[:-2, 1:-1])
    gradient_x = half_step * (smooth[1:-1, 2:] - smooth[1:-1, :-2])
    magnitude = tf.sqrt(tf.square(gradient_x) + tf.square(gradient_y))

    clipped = tf.cast(tf.reduce_max(image, axis=-1) >= _CLIPPED_LEVEL, tf.float32)
    near_clipped = tf.nn.max_pool2d(
        clipped[tf.newaxis, :, :, tf.newaxis], ksize=3, strides=1, padding="SAME"
    )[0, 1:-1, 1:-1, 0]
    keep = near_clipped == 0.0
    if valid is not None:
        keep &= tf.cast(valid, tf.bool)[1:-1, 1:-1]
    edges = tf.boolean_mask(magnitude, keep)
    tf.debugging.assert_positive(
        tf.size(edges), message="estimate_illuminant needs unclipped valid pixels."
    )
    order = float(MINKOWSKI_ORDER)
    estimate = tf.pow(tf.reduce_mean(tf.pow(edges, order), axis=0), 1.0 / order)
    tf.debugging.assert_positive(
        estimate, message="estimate_illuminant needs edges in every channel."
    )
    return estimate


__all__ = [
    "estimate_illuminant",
    "lab_to_rgb",
    "linear_to_srgb",
    "rgb_to_lab",
    "srgb_to_linear",
]

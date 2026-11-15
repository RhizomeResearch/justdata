"""Float32 RGB colour-space conversions on values in ``[0, 1]``.

sRGB follows IEC 61966-2-1 with the D65 white point; CIELAB uses the same
white, so ``lab_to_rgb(rgb_to_lab(x))`` returns ``x`` up to float32 rounding.
"""

import tensorflow as tf

_RGB_TO_XYZ = (
    (0.4124564, 0.3575761, 0.1804375),
    (0.2126729, 0.7151522, 0.0721750),
    (0.0193339, 0.1191920, 0.9503041),
)
_D65_WHITE_XYZ = (0.95047, 1.0, 1.08883)
_LAB_EPSILON = 216.0 / 24389.0
_LAB_KAPPA = 24389.0 / 27.0


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


__all__ = ["lab_to_rgb", "linear_to_srgb", "rgb_to_lab", "srgb_to_linear"]

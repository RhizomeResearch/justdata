"""Linear motion blur with ImageNet-C's ImageMagick kernel and severity table.

ImageMagick's ``MotionBlurImage(radius, sigma, angle)`` averages ``2 radius + 1``
samples along a ray from each pixel, weighted by a one-sided Gaussian of
``sigma``. Here the samples are splatted bilinearly instead of rounded, so the
kernel varies continuously with strength and angle, and the last sample fades
in as the radius grows. The table is ImageNet-C's at its 224-pixel width.
"""

import math

import tensorflow as tf

from justdata.vision.corruptions._conditions import photometric_condition
from justdata.vision.corruptions.registry import draw_seed, register_corruption
from justdata.vision.filters import correlate_hwc, splat_points

MOTION_RADIUS_PX = (10.0, 15.0, 15.0, 15.0, 20.0)
MOTION_SIGMA_PX = (3.0, 5.0, 8.0, 12.0, 15.0)
MOTION_ANGLE_RANGE_DEGREES = 45.0
MOTION_ANGLE_JITTER_DEGREES = 2.0

MOTION_BLUR_DESCRIPTOR = photometric_condition(
    "motion_blur",
    severity={"radius_px": MOTION_RADIUS_PX, "sigma_px": MOTION_SIGMA_PX},
    identity={"radius_px": 0.0, "sigma_px": MOTION_SIGMA_PX[0]},
    interpolation={"radius_px": "linear", "sigma_px": "linear"},
    implementation="justdata_tensorflow_imagemagick_motion_blur_v1",
    implementation_parameters=(
        ("signal_domain", "display_srgb"),
        ("kernel", "one_sided_gaussian_ray_samples_0_to_2_radius"),
        ("last_sample_weight", "clip_2_radius_plus_1_minus_index_0_1"),
        ("sample_placement", "bilinear_splat"),
        ("direction", "x_cos_angle_y_sin_angle_image_coordinates"),
        ("angle_range_degrees", MOTION_ANGLE_RANGE_DEGREES),
        ("frame_jitter_degrees", MOTION_ANGLE_JITTER_DEGREES),
        ("convolution", "fft_correlation_reflect_101"),
    ),
    spatial_parameters=("radius_px", "sigma_px"),
    reference_width=224,
    episode_draws=("angle",),
    frame_draws=("angle_jitter",),
)


def motion_kernel(radius: tf.Tensor, sigma: tf.Tensor, angle: tf.Tensor) -> tf.Tensor:
    """Correlation kernel whose centre is the blurred pixel itself."""
    span = 2.0 * radius
    last_tap = tf.cast(tf.math.ceil(span), tf.int32)
    taps = tf.cast(tf.range(last_tap + 1), tf.float32)
    weights = tf.exp(-tf.square(taps) / (2.0 * tf.square(sigma)))
    weights *= tf.clip_by_value(span + 1.0 - taps, 0.0, 1.0)
    half = last_tap + 1
    centre = tf.cast(half, tf.float32)
    points = tf.stack(
        [centre + taps * tf.sin(angle), centre + taps * tf.cos(angle)], axis=1
    )
    kernel = splat_points(points, weights, tf.stack([2 * half + 1, 2 * half + 1]))
    return kernel / tf.reduce_sum(kernel)


@register_corruption("motion_blur", descriptor=MOTION_BLUR_DESCRIPTOR)
@tf.function
def motion_blur(image, parameters, *, frame_seed, episode_seed, mode):
    del mode
    angle = tf.random.stateless_uniform(
        [],
        seed=draw_seed(episode_seed, 0),
        minval=-MOTION_ANGLE_RANGE_DEGREES,
        maxval=MOTION_ANGLE_RANGE_DEGREES,
        alg="philox",
    )
    angle += tf.random.stateless_uniform(
        [],
        seed=draw_seed(frame_seed, 0),
        minval=-MOTION_ANGLE_JITTER_DEGREES,
        maxval=MOTION_ANGLE_JITTER_DEGREES,
        alg="philox",
    )
    kernel = motion_kernel(
        parameters["radius_px"], parameters["sigma_px"], angle * (math.pi / 180.0)
    )
    return correlate_hwc(image, kernel)


__all__ = ["MOTION_BLUR_DESCRIPTOR", "motion_blur", "motion_kernel"]

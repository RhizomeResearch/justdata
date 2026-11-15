"""Falling rain streaks after Albumentations ``RandomRain``.

Albumentations draws ``drops_per_row * height`` grey (200) one-pixel lines of
length ``height // 8`` with one slant per image in ``[-10, 10]`` pixels, box
blurs them over 7 x 7, adds them with saturation, and multiplies by 0.7.
Severities 1-4 are its ``drizzle``, ``default``, ``heavy`` and ``torrential``
drop counts; severity 5 doubles ``torrential``. Counts refer to the image
height at the reference width, so coverage does not change with resolution.
The slant is fixed per episode and drop positions change per frame; drop ``k``
fades in as the count passes from ``k`` to ``k + 1``.
"""

import tensorflow as tf

from justdata.vision.corruptions._conditions import photometric_condition
from justdata.vision.corruptions.registry import draw_seed, register_corruption
from justdata.vision.filters import (
    correlate_hwc,
    depthwise_hwc,
    reflect_pad_hwc,
    splat_points,
)

RAIN_DROPS_PER_ROW = (0.25, 1.0 / 3.0, 1.0, 2.0, 4.0)
RAIN_BRIGHTNESS = 0.7
RAIN_DROP_LENGTH_FRACTION_OF_HEIGHT = 0.125
RAIN_SLANT_LIMIT_PX = 10.0
RAIN_DROP_WIDTH_PX = 1.0
RAIN_BLUR_SIZE_PX = 7.0
RAIN_DROP_VALUE = 200.0 / 255.0
RAIN_REFERENCE_WIDTH = 1024
_SAMPLE_SPACING_PX = 0.5

RAIN_DESCRIPTOR = photometric_condition(
    "rain",
    severity={
        "drops_per_row": RAIN_DROPS_PER_ROW,
        "brightness_coefficient": (RAIN_BRIGHTNESS,) * 5,
        "drop_length_fraction_of_height": (RAIN_DROP_LENGTH_FRACTION_OF_HEIGHT,) * 5,
        "slant_limit_px": (RAIN_SLANT_LIMIT_PX,) * 5,
        "drop_width_px": (RAIN_DROP_WIDTH_PX,) * 5,
        "blur_size_px": (RAIN_BLUR_SIZE_PX,) * 5,
    },
    identity={
        "drops_per_row": 0.0,
        "brightness_coefficient": 1.0,
        "drop_length_fraction_of_height": RAIN_DROP_LENGTH_FRACTION_OF_HEIGHT,
        "slant_limit_px": RAIN_SLANT_LIMIT_PX,
        "drop_width_px": RAIN_DROP_WIDTH_PX,
        "blur_size_px": RAIN_BLUR_SIZE_PX,
    },
    interpolation={
        "drops_per_row": "linear",
        "brightness_coefficient": "linear",
        "drop_length_fraction_of_height": "linear",
        "slant_limit_px": "linear",
        "drop_width_px": "linear",
        "blur_size_px": "linear",
    },
    implementation="justdata_tensorflow_albumentations_rain_v1",
    implementation_parameters=(
        ("signal_domain", "display_srgb"),
        ("drop_count", "drops_per_row_times_height_times_reference_width_over_width"),
        ("drop_weight", "clip_count_minus_index_0_1"),
        ("drop_start", "uniform_x_in_0_width_y_in_0_height_minus_length"),
        ("drop_value", RAIN_DROP_VALUE),
        ("streak", "antialiased_segment_coverage_clipped_0_1"),
        ("blur", "box_odd_rounded_size_reflect_101"),
        ("blend", "clip_input_plus_value_times_layer_then_times_brightness"),
    ),
    spatial_parameters=("slant_limit_px", "drop_width_px", "blur_size_px"),
    reference_width=RAIN_REFERENCE_WIDTH,
    episode_draws=("slant",),
    frame_draws=("drop_positions",),
)


def _streak_kernel(vector: tf.Tensor, width: tf.Tensor) -> tf.Tensor:
    """Correlation kernel that draws a streak from each impulse along ``vector``."""
    length = tf.norm(vector)
    along_count = tf.maximum(
        1, tf.cast(tf.math.ceil(length / _SAMPLE_SPACING_PX), tf.int32)
    )
    across_count = tf.maximum(
        1, tf.cast(tf.math.ceil(width / _SAMPLE_SPACING_PX), tf.int32)
    )
    along = (tf.cast(tf.range(along_count), tf.float32) + 0.5) / tf.cast(
        along_count, tf.float32
    )
    across = (
        (tf.cast(tf.range(across_count), tf.float32) + 0.5)
        / tf.cast(across_count, tf.float32)
        - 0.5
    ) * width
    normal = tf.stack([-vector[1], vector[0]]) / tf.maximum(length, 1e-6)
    offsets = (
        along[:, tf.newaxis, tf.newaxis] * vector
        + across[tf.newaxis, :, tf.newaxis] * normal
    )
    offsets = tf.reshape(offsets, [-1, 2])
    weight = (length / tf.cast(along_count, tf.float32)) * (
        width / tf.cast(across_count, tf.float32)
    )
    half = tf.cast(tf.math.ceil(tf.reduce_max(tf.abs(vector)) + width), tf.int32) + 1
    centre = tf.cast(half, tf.float32)
    return splat_points(
        centre - offsets,
        tf.fill([tf.shape(offsets)[0]], weight),
        tf.stack([2 * half + 1, 2 * half + 1]),
    )


def _box_blur(layer: tf.Tensor, size_px: tf.Tensor) -> tf.Tensor:
    radius = tf.maximum(0, tf.cast(tf.round((size_px - 1.0) / 2.0), tf.int32))
    size = 2 * radius + 1
    kernel = tf.fill([size, size], 1.0 / tf.cast(size * size, tf.float32))
    return depthwise_hwc(reflect_pad_hwc(layer, radius, radius), kernel)


@register_corruption("rain", descriptor=RAIN_DESCRIPTOR)
@tf.function
def rain(image, parameters, *, frame_seed, episode_seed, mode):
    del mode
    height, width = tf.shape(image)[0], tf.shape(image)[1]
    height_f = tf.cast(height, tf.float32)
    width_f = tf.cast(width, tf.float32)
    reference_rows = height_f * RAIN_REFERENCE_WIDTH / width_f
    count = parameters["drops_per_row"] * reference_rows
    pool = tf.cast(tf.math.ceil(max(RAIN_DROPS_PER_ROW) * reference_rows), tf.int32)
    length = parameters["drop_length_fraction_of_height"] * height_f

    starts = tf.random.stateless_uniform(
        [pool, 2], seed=draw_seed(frame_seed, 0), alg="philox"
    ) * tf.stack([tf.maximum(height_f - length, 0.0), width_f])
    weights = tf.clip_by_value(count - tf.range(tf.cast(pool, tf.float32)), 0.0, 1.0)
    impulses = splat_points(starts, weights, tf.stack([height, width]))

    slant_limit = parameters["slant_limit_px"]
    slant = tf.random.stateless_uniform(
        [],
        seed=draw_seed(episode_seed, 0),
        minval=-slant_limit,
        maxval=slant_limit,
        alg="philox",
    )
    kernel = _streak_kernel(tf.stack([length, slant]), parameters["drop_width_px"])
    layer = tf.clip_by_value(correlate_hwc(impulses[..., tf.newaxis], kernel), 0.0, 1.0)
    layer = _box_blur(layer, parameters["blur_size_px"])
    rained = tf.clip_by_value(image + RAIN_DROP_VALUE * layer, 0.0, 1.0)
    return rained * parameters["brightness_coefficient"]


__all__ = ["RAIN_DESCRIPTOR", "rain"]

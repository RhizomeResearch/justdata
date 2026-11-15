"""Water droplets on the lens, after the refraction model of Halimeh and Roser.

Each droplet is a feathered ellipse that shows a mirrored, minified, blurred
view of the scene around its centre, with a darker rim. The layout is drawn
from the episode seed for the largest declared droplet count, so a fixed
episode keeps every droplet in place. Droplet ``k`` fades in as the count
parameter passes from ``k`` to ``k + 1``, which keeps drifts smooth.
"""

import math

import tensorflow as tf

from justdata.vision.corruptions._conditions import photometric_condition
from justdata.vision.corruptions.registry import draw_seed, register_corruption
from justdata.vision.filters import gaussian_blur_hwc, reflect_pad_hwc

DROPLET_COUNT = (4, 8, 16, 28, 45)
DROPLET_MAX_RADIUS_PX = (14.0, 18.0, 24.0, 30.0, 38.0)
DROPLET_OPACITY = (0.7, 0.8, 0.85, 0.9, 0.9)
DROPLET_MIN_RADIUS_PX = 6.0
DROPLET_FEATHER_PX = 2.0
DROPLET_MINIFICATION = 3.0
DROPLET_RIM_DARKENING = 0.25
DROPLET_ASPECT_RANGE = (0.8, 1.25)
DROPLET_BLUR_FRACTION_OF_RADIUS = 0.25
DROPLET_POOL = max(DROPLET_COUNT)

LENS_DROPLETS_DESCRIPTOR = photometric_condition(
    "lens_droplets",
    severity={
        "droplet_count": DROPLET_COUNT,
        "max_radius_px": DROPLET_MAX_RADIUS_PX,
        "opacity": DROPLET_OPACITY,
        "min_radius_px": (DROPLET_MIN_RADIUS_PX,) * 5,
        "feather_px": (DROPLET_FEATHER_PX,) * 5,
    },
    identity={
        "droplet_count": 0,
        "max_radius_px": DROPLET_MAX_RADIUS_PX[0],
        "opacity": DROPLET_OPACITY[0],
        "min_radius_px": DROPLET_MIN_RADIUS_PX,
        "feather_px": DROPLET_FEATHER_PX,
    },
    interpolation={
        "droplet_count": "linear",
        "max_radius_px": "linear",
        "opacity": "linear",
        "min_radius_px": "linear",
        "feather_px": "linear",
    },
    implementation="justdata_tensorflow_lens_droplets_v1",
    implementation_parameters=(
        ("signal_domain", "display_srgb"),
        ("droplet_pool", DROPLET_POOL),
        ("droplet_weight", "clip_count_minus_index_0_1"),
        ("centre", "uniform_fraction_of_height_and_width"),
        ("radius", "log_uniform_min_to_max"),
        ("aspect", "log_uniform"),
        ("aspect_range", DROPLET_ASPECT_RANGE),
        ("alpha", "clip_0.5_minus_(rho_minus_1)_radius_over_feather"),
        ("content", "mirror_minify_about_centre_bilinear_reflect"),
        ("minification", DROPLET_MINIFICATION),
        ("content_blur_sigma_fraction_of_radius", DROPLET_BLUR_FRACTION_OF_RADIUS),
        ("compositing", "over_in_index_order"),
        ("blend", "one_minus_opacity_alpha_input_plus_opacity_alpha_content"),
        ("rim", "central_difference_alpha_gradient_times_feather_clipped"),
        ("rim_darkening", DROPLET_RIM_DARKENING),
    ),
    spatial_parameters=("max_radius_px", "min_radius_px", "feather_px"),
    reference_width=1024,
    episode_draws=("droplet_layout",),
)


def _droplet(image, centre, radii, feather, weight):
    """Alpha ``[H, W, 1]`` and content ``[H, W, 3]`` of one droplet, zero outside it."""
    height, width = tf.shape(image)[0], tf.shape(image)[1]
    radius = tf.sqrt(radii[0] * radii[1])
    sigma = DROPLET_BLUR_FRACTION_OF_RADIUS * radius
    margin = feather + 3.0 * sigma + 1.0
    top = tf.clip_by_value(
        tf.cast(tf.floor(centre[0] - radii[0] - margin), tf.int32), 0, height
    )
    bottom = tf.clip_by_value(
        tf.cast(tf.math.ceil(centre[0] + radii[0] + margin), tf.int32), 0, height
    )
    left = tf.clip_by_value(
        tf.cast(tf.floor(centre[1] - radii[1] - margin), tf.int32), 0, width
    )
    right = tf.clip_by_value(
        tf.cast(tf.math.ceil(centre[1] + radii[1] + margin), tf.int32), 0, width
    )

    # Output canvas pixel (x, y) samples input x = c - z (left + x - c) in
    # integer-centred coordinates.
    z = DROPLET_MINIFICATION
    centre_index = centre - 0.5
    transform = tf.stack(
        [
            -z,
            0.0,
            (1.0 + z) * centre_index[1] - z * tf.cast(left, tf.float32),
            0.0,
            -z,
            (1.0 + z) * centre_index[0] - z * tf.cast(top, tf.float32),
            0.0,
            0.0,
        ]
    )
    content = tf.raw_ops.ImageProjectiveTransformV3(
        images=image[tf.newaxis],
        transforms=transform[tf.newaxis],
        output_shape=tf.stack([bottom - top, right - left]),
        fill_value=0.0,
        interpolation="BILINEAR",
        fill_mode="REFLECT",
    )[0]
    content = gaussian_blur_hwc(content, sigma, sigma)

    ys = tf.cast(tf.range(top, bottom), tf.float32)[:, tf.newaxis] + 0.5
    xs = tf.cast(tf.range(left, right), tf.float32)[tf.newaxis, :] + 0.5
    rho = tf.sqrt(
        tf.square((ys - centre[0]) / radii[0]) + tf.square((xs - centre[1]) / radii[1])
    )
    alpha = tf.clip_by_value(0.5 - (rho - 1.0) * radius / feather, 0.0, 1.0) * weight
    alpha = alpha[..., tf.newaxis]

    paddings = [[top, height - bottom], [left, width - right], [0, 0]]
    return tf.pad(alpha, paddings), tf.pad(alpha * content, paddings)


def _rim(alpha: tf.Tensor, feather: tf.Tensor) -> tf.Tensor:
    padded = reflect_pad_hwc(alpha, 1, 1)
    gradient_y = 0.5 * (padded[2:, 1:-1] - padded[:-2, 1:-1])
    gradient_x = 0.5 * (padded[1:-1, 2:] - padded[1:-1, :-2])
    magnitude = tf.sqrt(tf.square(gradient_y) + tf.square(gradient_x))
    return tf.clip_by_value(magnitude * feather, 0.0, 1.0)


@register_corruption("lens_droplets", descriptor=LENS_DROPLETS_DESCRIPTOR)
@tf.function
def lens_droplets(image, parameters, *, frame_seed, episode_seed, mode):
    del frame_seed, mode
    height = tf.cast(tf.shape(image)[0], tf.float32)
    width = tf.cast(tf.shape(image)[1], tf.float32)
    layout = tf.random.stateless_uniform(
        [DROPLET_POOL, 4], seed=draw_seed(episode_seed, 0), alg="philox"
    )
    centres = layout[:, :2] * tf.stack([height, width])
    min_radius = parameters["min_radius_px"]
    radius = min_radius * tf.pow(parameters["max_radius_px"] / min_radius, layout[:, 2])
    log_low, log_high = (math.log(value) for value in DROPLET_ASPECT_RANGE)
    aspect = tf.exp(log_low + layout[:, 3] * (log_high - log_low))
    radii = tf.stack([radius / tf.sqrt(aspect), radius * tf.sqrt(aspect)], axis=1)
    count = parameters["droplet_count"]
    weights = tf.clip_by_value(
        count - tf.range(DROPLET_POOL, dtype=tf.float32), 0.0, 1.0
    )
    active = tf.minimum(tf.cast(tf.math.ceil(count), tf.int32), DROPLET_POOL)
    feather = parameters["feather_px"]

    def body(index, alpha_total, content_total):
        alpha, content = _droplet(
            image, centres[index], radii[index], feather, weights[index]
        )
        return (
            index + 1,
            alpha_total * (1.0 - alpha) + alpha,
            content_total * (1.0 - alpha) + content,
        )

    _index, alpha_total, content_total = tf.while_loop(
        lambda index, *_: index < active,
        body,
        (tf.constant(0), tf.zeros_like(image[..., :1]), tf.zeros_like(image)),
    )
    opacity = parameters["opacity"]
    output = image * (1.0 - opacity * alpha_total) + opacity * content_total
    return output * (1.0 - DROPLET_RIM_DARKENING * _rim(alpha_total, feather))


__all__ = ["LENS_DROPLETS_DESCRIPTOR", "lens_droplets"]

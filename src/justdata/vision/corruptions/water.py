"""Sun glitter on water: bright, horizontally elongated highlights under the sun.

No reference library renders glitter; the parameters are the request's starting
values. Highlights are band-limited noise thresholded to a lit share inside a
column band (the sun's azimuth, fixed per episode) that fades from the horizon
toward the image bottom, then bloomed, screen-blended, and veiled toward the
haze airlight.
"""

import tensorflow as tf

from justdata.vision.corruptions._conditions import photometric_condition
from justdata.vision.corruptions.atmosphere import estimate_airlight
from justdata.vision.corruptions.registry import draw_seed, register_corruption
from justdata.vision.filters import gaussian_blur_hwc
from justdata.vision.horizon import horizon_pixel_coordinates

GLITTER_BAND_SIGMA = (0.08, 0.10, 0.14, 0.18, 0.25)
GLITTER_LIT_SHARE = (0.02, 0.04, 0.08, 0.14, 0.22)
GLITTER_BLOOM_GAIN = (0.1, 0.2, 0.3, 0.4, 0.5)
GLITTER_VEILING = (1.0, 0.97, 0.93, 0.87, 0.80)
GLITTER_HIGHLIGHT_SIGMA_X_PX = 3.0
GLITTER_HIGHLIGHT_SIGMA_Y_PX = 1.0
GLITTER_BLOOM_SIGMA_PX = 6.0
GLITTER_BOTTOM_ENVELOPE = 0.3
GLITTER_COLOR = (1.0, 0.98, 0.92)
GLITTER_COLUMN_RANGE = (0.2, 0.8)
GLITTER_CALIBRATION_ENVELOPE = 0.5

SUN_GLITTER_DESCRIPTOR = photometric_condition(
    "sun_glitter",
    severity={
        "band_sigma_fraction_of_width": GLITTER_BAND_SIGMA,
        "lit_share": GLITTER_LIT_SHARE,
        "bloom_gain": GLITTER_BLOOM_GAIN,
        "veiling": GLITTER_VEILING,
        "highlight_sigma_x_px": (GLITTER_HIGHLIGHT_SIGMA_X_PX,) * 5,
        "highlight_sigma_y_px": (GLITTER_HIGHLIGHT_SIGMA_Y_PX,) * 5,
        "bloom_sigma_px": (GLITTER_BLOOM_SIGMA_PX,) * 5,
    },
    identity={
        "band_sigma_fraction_of_width": GLITTER_BAND_SIGMA[0],
        "lit_share": 0.0,
        "bloom_gain": 0.0,
        "veiling": 1.0,
        "highlight_sigma_x_px": GLITTER_HIGHLIGHT_SIGMA_X_PX,
        "highlight_sigma_y_px": GLITTER_HIGHLIGHT_SIGMA_Y_PX,
        "bloom_sigma_px": GLITTER_BLOOM_SIGMA_PX,
    },
    interpolation={
        "band_sigma_fraction_of_width": "linear",
        "lit_share": "linear",
        "bloom_gain": "linear",
        "veiling": "linear",
        "highlight_sigma_x_px": "linear",
        "highlight_sigma_y_px": "linear",
        "bloom_sigma_px": "linear",
    },
    implementation="justdata_tensorflow_sun_glitter_v1",
    implementation_parameters=(
        ("signal_domain", "display_srgb"),
        ("envelope", "column_gaussian_times_ramp_1_at_boundary_to_bottom_value"),
        ("bottom_envelope", GLITTER_BOTTOM_ENVELOPE),
        ("column_range_fraction_of_width", GLITTER_COLUMN_RANGE),
        ("boundary_row", "horizon_line_else_middle_row"),
        ("highlight_noise", "standard_normal_blurred_anisotropic_gaussian"),
        (
            "threshold",
            "top_lit_share_of_noise_where_envelope_above_calibration_in_region",
        ),
        ("calibration_envelope", GLITTER_CALIBRATION_ENVELOPE),
        ("highlights", "lit_times_envelope_times_region"),
        ("bloom", "highlights_plus_gain_times_gaussian_blur_clipped_0_1"),
        ("blend", "screen_with_color"),
        ("color", GLITTER_COLOR),
        ("veiling_toward", "haze_airlight_estimate"),
    ),
    spatial_parameters=(
        "highlight_sigma_x_px",
        "highlight_sigma_y_px",
        "bloom_sigma_px",
    ),
    reference_width=1024,
    episode_draws=("sun_column",),
    frame_draws=("highlight_noise",),
    auxiliary_inputs=("region", "horizon"),
    modes=(
        ("region", ("region",)),
        ("horizon", ("horizon",)),
        ("lower_half", ()),
    ),
)


def _lit_mask(noise: tf.Tensor, calibration: tf.Tensor, share: tf.Tensor) -> tf.Tensor:
    """Noise at or above the value that lights ``share`` of the calibration pixels."""
    values = tf.boolean_mask(noise, calibration)
    count = tf.cast(
        tf.floor(share * tf.cast(tf.size(values), tf.float32)),
        tf.int32,
    )

    def threshold_lit() -> tf.Tensor:
        threshold = tf.math.top_k(values, k=count, sorted=True).values[-1]
        return noise >= threshold

    return tf.cond(count > 0, threshold_lit, lambda: tf.zeros_like(calibration))


@register_corruption("sun_glitter", descriptor=SUN_GLITTER_DESCRIPTOR)
@tf.function
def sun_glitter(image, parameters, *, frame_seed, episode_seed, mode, region, horizon):
    height, width = tf.shape(image)[0], tf.shape(image)[1]
    height_f = tf.cast(height, tf.float32)
    width_f = tf.cast(width, tf.float32)
    rows = tf.cast(tf.range(height), tf.float32)[:, tf.newaxis] + 0.5
    columns = tf.cast(tf.range(width), tf.float32)[tf.newaxis, :] + 0.5

    if horizon is not None:
        line = horizon["rows"]
        boundary = line[0] + (line[1] - line[0]) * columns / width_f
    else:
        boundary = tf.fill([1, width], 0.5 * height_f)
    if mode == "region":
        water = region
    elif mode == "horizon":
        below, _along = horizon_pixel_coordinates(horizon["rows"], height, width)
        water = below > 0.0
    else:
        water = tf.broadcast_to(rows > 0.5 * height_f, [height, width])

    depth = tf.math.divide_no_nan(rows - boundary, height_f - boundary)
    ramp = 1.0 - (1.0 - GLITTER_BOTTOM_ENVELOPE) * tf.clip_by_value(depth, 0.0, 1.0)
    sun_column = tf.random.stateless_uniform(
        [],
        seed=draw_seed(episode_seed, 0),
        minval=GLITTER_COLUMN_RANGE[0],
        maxval=GLITTER_COLUMN_RANGE[1],
        alg="philox",
    )
    band = tf.exp(
        -0.5
        * tf.square(
            (columns / width_f - sun_column)
            / parameters["band_sigma_fraction_of_width"]
        )
    )
    envelope = band * ramp

    noise = tf.random.stateless_normal(
        [height, width, 1], seed=draw_seed(frame_seed, 0), alg="philox"
    )
    noise = gaussian_blur_hwc(
        noise,
        parameters["highlight_sigma_y_px"],
        parameters["highlight_sigma_x_px"],
    )[..., 0]
    calibration = water & (envelope > GLITTER_CALIBRATION_ENVELOPE)
    lit = _lit_mask(noise, calibration, parameters["lit_share"])
    highlights = tf.cast(lit & water, tf.float32) * envelope
    highlights = highlights[..., tf.newaxis]
    bloom = gaussian_blur_hwc(
        highlights, parameters["bloom_sigma_px"], parameters["bloom_sigma_px"]
    )
    highlights = tf.clip_by_value(
        highlights + parameters["bloom_gain"] * bloom, 0.0, 1.0
    )

    color = tf.constant(GLITTER_COLOR, tf.float32)
    screened = 1.0 - (1.0 - image) * (1.0 - color * highlights)
    veiling = parameters["veiling"]
    return veiling * screened + (1.0 - veiling) * estimate_airlight(image)


__all__ = ["SUN_GLITTER_DESCRIPTOR", "sun_glitter"]

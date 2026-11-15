"""Scattering haze: ``I = J * t + A * (1 - t)`` with ``t = exp(-beta * d)``.

The model, the display-space application, and the atmospheric light estimate
follow Sakaridis, Dai and Van Gool, "Semantic Foggy Scene Understanding with
Synthetic Data" (IJCV 2018), whose Foggy Cityscapes levels are severities 2-4.
Patchiness scales ``beta`` by a diamond-square plasma field, ImageNet-C's
``plasma_fractal``, drawn once per episode.
"""

import tensorflow as tf

from justdata.vision.corruptions._conditions import photometric_condition
from justdata.vision.corruptions.registry import draw_seed, register_corruption
from justdata.vision.horizon import horizon_pixel_coordinates

HAZE_ATTENUATION_PER_M = (0.003, 0.005, 0.01, 0.02, 0.04)
HAZE_HOMOGENEOUS_TRANSMISSION = (0.85, 0.70, 0.55, 0.40, 0.25)
HAZE_PATCHINESS = 0.3
HAZE_FAR_DISTANCE_M = 2000.0
DARK_CHANNEL_WINDOW_PX = 15
BRIGHTEST_DARK_CHANNEL_FRACTION = 1.0 / 1000.0
PLASMA_MAP_SIZE = 256
PLASMA_WIBBLE_DECAY = 3.0
_GRAY_WEIGHTS = (0.2989, 0.5870, 0.1140)

HAZE_DESCRIPTOR = photometric_condition(
    "haze",
    severity={
        "attenuation_per_m": HAZE_ATTENUATION_PER_M,
        "homogeneous_transmission": HAZE_HOMOGENEOUS_TRANSMISSION,
        "patchiness": (HAZE_PATCHINESS,) * 5,
    },
    identity={
        "attenuation_per_m": 0.0,
        "homogeneous_transmission": 1.0,
        "patchiness": HAZE_PATCHINESS,
    },
    interpolation={
        "attenuation_per_m": "linear",
        "homogeneous_transmission": "linear",
        "patchiness": "linear",
    },
    implementation="justdata_tensorflow_scattering_haze_v1",
    implementation_parameters=(
        ("signal_domain", "display_srgb"),
        ("far_distance_m", HAZE_FAR_DISTANCE_M),
        (
            "horizon_distance",
            "min(h_sqrt(f2_plus_below2_plus_along2)_over_below_far)_below_line_"
            "else_far; along_measured_from_image_centre_column; tall_objects_get_"
            "the_distance_of_the_water_behind_them",
        ),
        ("homogeneous_transmission", "transmission_power_patch_factor"),
        ("patch_factor", "1_plus_patchiness_times_plasma_minus_0.5"),
        ("plasma", "imagenet_c_diamond_square_wibble_100"),
        ("plasma_map_size", PLASMA_MAP_SIZE),
        ("plasma_wibble_decay", PLASMA_WIBBLE_DECAY),
        ("plasma_to_image", "crop_longer_side_to_aspect_then_bilinear_resize"),
        ("airlight", "sakaridis_dark_channel_median_gray_of_brightest"),
        ("dark_channel_window_px", DARK_CHANNEL_WINDOW_PX),
        ("brightest_fraction", BRIGHTEST_DARK_CHANNEL_FRACTION),
        ("gray_weights", _GRAY_WEIGHTS),
    ),
    episode_draws=("patchiness_field",),
    auxiliary_inputs=("depth_m", "horizon"),
    modes=(
        ("depth", ("depth_m",)),
        ("horizon", ("horizon",)),
        ("homogeneous", ()),
    ),
)


def estimate_airlight(image: tf.Tensor) -> tf.Tensor:
    """Atmospheric light as in Sakaridis et al.: an input pixel, ``[3]``.

    Take the odd count ``floor(N / 1000)`` (rounded up to odd) of pixels with
    the brightest 15 x 15 dark channel, then the first of them, in descending
    dark-channel order, whose grey value equals their median grey value.
    """
    dark = tf.reduce_min(image, axis=-1)
    eroded = -tf.nn.max_pool2d(
        -dark[tf.newaxis, :, :, tf.newaxis],
        ksize=DARK_CHANNEL_WINDOW_PX,
        strides=1,
        padding="SAME",
    )
    eroded = tf.reshape(eroded, [-1])
    pixels = tf.reshape(image, [-1, 3])
    count = tf.cast(
        tf.floor(
            tf.cast(tf.size(eroded), tf.float32) * BRIGHTEST_DARK_CHANNEL_FRACTION
        ),
        tf.int32,
    )
    count = count + tf.math.floormod(count + 1, 2)
    order = tf.argsort(eroded, direction="DESCENDING", stable=True)[:count]
    gray = tf.linalg.matvec(
        tf.gather(pixels, order), tf.constant(_GRAY_WEIGHTS, tf.float32)
    )
    median = tf.sort(gray)[count // 2]
    first = tf.argmax(tf.cast(tf.equal(gray, median), tf.int32))
    return tf.gather(pixels, order[first])


def _wibbled_mean(total: tf.Tensor, seed: tf.Tensor, wibble: float) -> tf.Tensor:
    noise = tf.random.stateless_uniform(
        tf.shape(total), seed=seed, minval=-wibble, maxval=wibble, alg="philox"
    )
    return total / 4.0 + wibble * noise


def plasma_field(seed: tf.Tensor, height: tf.Tensor, width: tf.Tensor) -> tf.Tensor:
    """ImageNet-C diamond-square heightmap in ``[0, 1]``, fitted to ``[H, W]``.

    The square map is cropped along its longer image side to the image aspect
    and resized, so the field covers the same image fraction at any resolution.
    """
    grid = tf.zeros([1, 1], tf.float32)
    wibble = 100.0
    level = 0
    stepsize = PLASMA_MAP_SIZE
    while stepsize >= 2:
        n = tf.shape(grid)[0]
        draws = [draw_seed(seed, 3 * level + index) for index in range(3)]
        corners = grid + tf.roll(grid, shift=-1, axis=0)
        corners += tf.roll(corners, shift=-1, axis=1)
        centres = _wibbled_mean(corners, draws[0], wibble)
        top = _wibbled_mean(
            centres + tf.roll(centres, 1, axis=0) + grid + tf.roll(grid, -1, axis=1),
            draws[1],
            wibble,
        )
        left = _wibbled_mean(
            centres + tf.roll(centres, 1, axis=1) + grid + tf.roll(grid, -1, axis=0),
            draws[2],
            wibble,
        )
        even_rows = tf.reshape(tf.stack([grid, top], axis=-1), [n, 2 * n])
        odd_rows = tf.reshape(tf.stack([left, centres], axis=-1), [n, 2 * n])
        grid = tf.reshape(tf.stack([even_rows, odd_rows], axis=1), [2 * n, 2 * n])
        stepsize //= 2
        wibble /= PLASMA_WIBBLE_DECAY
        level += 1

    grid -= tf.reduce_min(grid)
    grid /= tf.reduce_max(grid)
    height_f = tf.cast(height, tf.float32)
    width_f = tf.cast(width, tf.float32)
    longer = tf.maximum(height_f, width_f)
    crop_h = tf.maximum(
        1, tf.cast(tf.math.ceil(PLASMA_MAP_SIZE * height_f / longer), tf.int32)
    )
    crop_w = tf.maximum(
        1, tf.cast(tf.math.ceil(PLASMA_MAP_SIZE * width_f / longer), tf.int32)
    )
    cropped = grid[:crop_h, :crop_w, tf.newaxis]
    return tf.image.resize(cropped, [height, width], method="bilinear")[..., 0]


def _flat_water_distance(horizon, height, width) -> tf.Tensor:
    camera_height = horizon["camera_height_m"]
    focal = horizon["focal_length_px"]
    if camera_height is None or focal is None:
        raise ValueError(
            "haze horizon mode needs Horizon.camera_height_m and focal_length_px."
        )
    below, along = horizon_pixel_coordinates(horizon["rows"], height, width)
    safe_below = tf.where(below > 0.0, below, 1.0)
    distance = (
        camera_height
        * tf.sqrt(tf.square(focal) + tf.square(below) + tf.square(along))
        / safe_below
    )
    return tf.where(
        below > 0.0,
        tf.minimum(distance, HAZE_FAR_DISTANCE_M),
        HAZE_FAR_DISTANCE_M,
    )


@register_corruption("haze", descriptor=HAZE_DESCRIPTOR)
@tf.function
def haze(image, parameters, *, frame_seed, episode_seed, mode, depth_m, horizon):
    del frame_seed
    height, width = tf.shape(image)[0], tf.shape(image)[1]
    field = plasma_field(draw_seed(episode_seed, 0), height, width)
    patch_factor = 1.0 + parameters["patchiness"] * (field - 0.5)
    beta = parameters["attenuation_per_m"] * patch_factor
    if mode == "depth":
        transmission = tf.exp(-beta * depth_m)
    elif mode == "horizon":
        transmission = tf.exp(-beta * _flat_water_distance(horizon, height, width))
    else:
        transmission = tf.pow(parameters["homogeneous_transmission"], patch_factor)
    transmission = transmission[..., tf.newaxis]
    airlight = estimate_airlight(image)
    return image * transmission + airlight * (1.0 - transmission)


__all__ = [
    "HAZE_DESCRIPTOR",
    "estimate_airlight",
    "haze",
    "plasma_field",
]

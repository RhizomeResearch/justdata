"""Exposure, colour-cast, and sensor-noise conditions.

``low_light`` follows the request's camera model in linear light, with the
heteroscedastic Gaussian noise of Brooks et al., "Unprocessing Images for
Learned Raw Denoising" (CVPR 2019): variance ``signal * shot + read`` and read
noise on their line ``ln(read) = 2.18 ln(shot) + 1.20``. ``shot_noise`` is
ImageNet-C's display-space Poisson noise. The white-balance conditions use the
blackbody table shared by Albumentations ``PlanckianJitter`` and Kornia
``RandomPlanckianJitter``.
"""

import math

import tensorflow as tf

from justdata.vision.colorspace import linear_to_srgb, srgb_to_linear
from justdata.vision.corruptions._conditions import photometric_condition
from justdata.vision.corruptions.registry import draw_seed, register_corruption

LOW_LIGHT_EXPOSURE_GAIN = (0.5, 0.25, 0.12, 0.06, 0.03)
LOW_LIGHT_SHOT_NOISE = (1 / 2000, 1 / 1000, 1 / 500, 1 / 250, 1 / 120)
LOW_LIGHT_READ_NOISE_VARIANCE = tuple(
    math.exp(2.18 * math.log(shot) + 1.20) for shot in LOW_LIGHT_SHOT_NOISE
)
LOW_LIGHT_COMPENSATION_CAP = (1.5, 2.0, 3.0, 4.0, 6.0)
LOW_LIGHT_TARGET_MEAN_FRACTION = 0.6
OVEREXPOSURE_GAIN = (1.3, 1.6, 2.0, 2.6, 3.4)
WHITE_BALANCE_NEUTRAL_KELVIN = 6500.0
WHITE_BALANCE_WARM_KELVIN = (5500.0, 4500.0, 4000.0, 3500.0, 3000.0)
WHITE_BALANCE_COOL_KELVIN = (7500.0, 9000.0, 10500.0, 12500.0, 15000.0)
SHOT_NOISE_PHOTONS = (60, 25, 12, 5, 3)

# Albumentations PLANCKIAN_COEFFS["blackbody"]: RGB from 3000 K to 15000 K in
# 500 K steps.
BLACKBODY_FIRST_KELVIN = 3000.0
BLACKBODY_STEP_KELVIN = 500.0
BLACKBODY_RGB = (
    (0.6743, 0.4029, 0.0013),
    (0.6281, 0.4241, 0.1665),
    (0.5919, 0.4372, 0.2513),
    (0.5623, 0.4457, 0.3154),
    (0.5376, 0.4515, 0.3672),
    (0.5163, 0.4555, 0.4103),
    (0.4979, 0.4584, 0.4468),
    (0.4816, 0.4604, 0.4782),
    (0.4672, 0.4619, 0.5053),
    (0.4542, 0.4630, 0.5289),
    (0.4426, 0.4638, 0.5497),
    (0.4320, 0.4644, 0.5681),
    (0.4223, 0.4648, 0.5844),
    (0.4135, 0.4651, 0.5990),
    (0.4054, 0.4653, 0.6121),
    (0.3980, 0.4654, 0.6239),
    (0.3911, 0.4655, 0.6346),
    (0.3847, 0.4656, 0.6444),
    (0.3787, 0.4656, 0.6532),
    (0.3732, 0.4656, 0.6613),
    (0.3680, 0.4655, 0.6688),
    (0.3632, 0.4655, 0.6756),
    (0.3586, 0.4655, 0.6820),
    (0.3544, 0.4654, 0.6878),
    (0.3503, 0.4653, 0.6933),
)

_SRGB = ("transfer_function", "iec_61966_2_1_srgb")

LOW_LIGHT_DESCRIPTOR = photometric_condition(
    "low_light",
    severity={
        "exposure_gain": LOW_LIGHT_EXPOSURE_GAIN,
        "shot_noise": LOW_LIGHT_SHOT_NOISE,
        "read_noise_variance": LOW_LIGHT_READ_NOISE_VARIANCE,
        "compensation_cap": LOW_LIGHT_COMPENSATION_CAP,
    },
    identity={
        "exposure_gain": 1.0,
        "shot_noise": 0.0,
        "read_noise_variance": 0.0,
        "compensation_cap": 1.0,
    },
    interpolation={
        "exposure_gain": "log",
        "shot_noise": "linear",
        "read_noise_variance": "linear",
        "compensation_cap": "log",
    },
    implementation="justdata_tensorflow_low_light_v1",
    implementation_parameters=(
        _SRGB,
        ("signal_domain", "linear_light"),
        ("noise_model", "gaussian_variance_signal_times_shot_plus_read"),
        ("read_noise_line", "ln_read_equals_2.18_ln_shot_plus_1.20"),
        ("target_mean_fraction", LOW_LIGHT_TARGET_MEAN_FRACTION),
        (
            "compensation_gain",
            "clip_target_mean_fraction_times_mean_input_over_mean_noisy_1_cap",
        ),
        ("mean_over", "all_pixels_and_channels"),
    ),
    frame_draws=("sensor_noise",),
)

OVEREXPOSURE_DESCRIPTOR = photometric_condition(
    "overexposure",
    severity={"exposure_gain": OVEREXPOSURE_GAIN},
    identity={"exposure_gain": 1.0},
    interpolation={"exposure_gain": "log"},
    implementation="justdata_tensorflow_linear_light_gain_v1",
    implementation_parameters=(_SRGB, ("signal_domain", "linear_light")),
)


def _white_balance_descriptor(name: str, temperatures: tuple[float, ...]):
    return photometric_condition(
        name,
        severity={"temperature_kelvin": temperatures},
        identity={"temperature_kelvin": WHITE_BALANCE_NEUTRAL_KELVIN},
        interpolation={"temperature_kelvin": "linear"},
        implementation="justdata_tensorflow_planckian_white_balance_v1",
        implementation_parameters=(
            ("coefficient_table", "albumentations_planckian_blackbody"),
            ("table_interpolation", "linear_in_kelvin"),
            ("gains", "red_and_blue_over_green_relative_to_neutral"),
            ("neutral_kelvin", WHITE_BALANCE_NEUTRAL_KELVIN),
            ("signal_domain", "display_srgb"),
        ),
    )


WHITE_BALANCE_WARM_DESCRIPTOR = _white_balance_descriptor(
    "white_balance_warm", WHITE_BALANCE_WARM_KELVIN
)
WHITE_BALANCE_COOL_DESCRIPTOR = _white_balance_descriptor(
    "white_balance_cool", WHITE_BALANCE_COOL_KELVIN
)

SHOT_NOISE_DESCRIPTOR = photometric_condition(
    "shot_noise",
    severity={"photon_scale_inverse": tuple(1 / c for c in SHOT_NOISE_PHOTONS)},
    identity={"photon_scale_inverse": 0.0},
    interpolation={"photon_scale_inverse": "linear"},
    implementation="justdata_tensorflow_poisson_shot_noise_v1",
    implementation_parameters=(
        ("signal_domain", "display_srgb"),
        ("formula", "poisson_of_value_over_scale_times_scale"),
        ("random_distribution", "stateless_poisson"),
    ),
    frame_draws=("photon_noise",),
)


@register_corruption("low_light", descriptor=LOW_LIGHT_DESCRIPTOR)
@tf.function
def low_light(image, parameters, *, frame_seed, episode_seed, mode):
    del episode_seed, mode
    linear = srgb_to_linear(image)
    dimmed = parameters["exposure_gain"] * linear
    variance = dimmed * parameters["shot_noise"] + parameters["read_noise_variance"]
    noise = tf.random.stateless_normal(
        tf.shape(image), seed=draw_seed(frame_seed, 0), alg="philox"
    )
    noisy = dimmed + tf.sqrt(tf.maximum(variance, 0.0)) * noise
    target = LOW_LIGHT_TARGET_MEAN_FRACTION * tf.reduce_mean(linear)
    gain = tf.clip_by_value(
        tf.math.divide_no_nan(target, tf.reduce_mean(noisy)),
        1.0,
        parameters["compensation_cap"],
    )
    return linear_to_srgb(tf.clip_by_value(gain * noisy, 0.0, 1.0))


@register_corruption("overexposure", descriptor=OVEREXPOSURE_DESCRIPTOR)
@tf.function
def overexposure(image, parameters, *, frame_seed, episode_seed, mode):
    del frame_seed, episode_seed, mode
    linear = parameters["exposure_gain"] * srgb_to_linear(image)
    return linear_to_srgb(tf.clip_by_value(linear, 0.0, 1.0))


def _blackbody_chroma(temperature: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Red/green and blue/green ratios of the table, linear in kelvin."""
    table = tf.constant(BLACKBODY_RGB, tf.float32)
    position = (temperature - BLACKBODY_FIRST_KELVIN) / BLACKBODY_STEP_KELVIN
    lower = tf.cast(tf.floor(position), tf.int32)
    lower = tf.clip_by_value(lower, 0, len(BLACKBODY_RGB) - 2)
    fraction = position - tf.cast(lower, tf.float32)
    rgb = table[lower] + fraction * (table[lower + 1] - table[lower])
    return rgb[0] / rgb[1], rgb[2] / rgb[1]


def _white_balance(image, parameters, *, frame_seed, episode_seed, mode):
    del frame_seed, episode_seed, mode
    red, blue = _blackbody_chroma(parameters["temperature_kelvin"])
    neutral_red, neutral_blue = _blackbody_chroma(
        tf.constant(WHITE_BALANCE_NEUTRAL_KELVIN, tf.float32)
    )
    gains = tf.stack([red / neutral_red, tf.constant(1.0), blue / neutral_blue])
    return image * gains


white_balance_warm = register_corruption(
    "white_balance_warm", descriptor=WHITE_BALANCE_WARM_DESCRIPTOR
)(tf.function(_white_balance))
white_balance_cool = register_corruption(
    "white_balance_cool", descriptor=WHITE_BALANCE_COOL_DESCRIPTOR
)(tf.function(_white_balance))


@register_corruption("shot_noise", descriptor=SHOT_NOISE_DESCRIPTOR)
@tf.function
def shot_noise(image, parameters, *, frame_seed, episode_seed, mode):
    del episode_seed, mode
    scale = parameters["photon_scale_inverse"]
    counts = tf.random.stateless_poisson(
        tf.shape(image),
        seed=draw_seed(frame_seed, 0),
        lam=image / scale,
        dtype=tf.float32,
    )
    return counts * scale


__all__ = [
    "BLACKBODY_RGB",
    "LOW_LIGHT_DESCRIPTOR",
    "OVEREXPOSURE_DESCRIPTOR",
    "SHOT_NOISE_DESCRIPTOR",
    "WHITE_BALANCE_COOL_DESCRIPTOR",
    "WHITE_BALANCE_WARM_DESCRIPTOR",
    "low_light",
    "overexposure",
    "shot_noise",
    "white_balance_cool",
    "white_balance_warm",
]

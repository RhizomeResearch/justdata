"""Versioned TensorFlow-native decoded-image corruptions."""

import tensorflow as tf

from justdata.vision.corruptions.registry import (
    CorruptionDescriptor,
    register_corruption,
)

GAUSSIAN_BLUR_SIGMA = (0.5, 1.0, 2.0, 3.0, 4.0)
GAUSSIAN_NOISE_STD_UNIT = (0.01, 0.02, 0.04, 0.08, 0.16)
JPEG_QUALITY = (90, 75, 55, 35, 15)
CONTRAST_FACTOR = (0.9, 0.75, 0.6, 0.45, 0.3)
BRIGHTNESS_FACTOR = (0.9, 0.8, 0.7, 0.6, 0.5)

_SEVERITIES = (1, 2, 3, 4, 5)
_UINT8_INPUT = "decoded_srgb_rgb_uint8_hwc_0_255"
_UINT8_OUTPUT = "decoded_srgb_rgb_uint8_hwc_0_255"
_STATELESS_SEED = (
    "caller_supplied_int32_shape_2; no_global_rng; exact_seed_consumed_directly"
)
_DETERMINISTIC_SEED = "caller_supplied_int32_shape_2; accepted_validated_and_ignored"

GAUSSIAN_BLUR_DESCRIPTOR = CorruptionDescriptor(
    name="gaussian_blur",
    version="1.0.0",
    input_domain=_UINT8_INPUT,
    input_dtypes=("uint8",),
    output_domain=_UINT8_OUTPUT,
    output_shape="same_as_input_hwc",
    output_dtype="uint8",
    severity_values=_SEVERITIES,
    severity_parameters=tuple(
        (level, (("sigma_pixels", sigma),))
        for level, sigma in enumerate(GAUSSIAN_BLUR_SIGMA, start=1)
    ),
    uses_randomness=False,
    seed_contract=_DETERMINISTIC_SEED,
    clipping_policy="clip_float32_to_closed_interval_0_255",
    rounding_policy="clip_then_round_half_to_even",
    implementation="justdata_tensorflow_gaussian_blur_v1",
    implementation_parameters=(
        ("kernel_radius_formula", "ceil_3_times_sigma"),
        ("kernel_size_formula", "2_times_radius_plus_1"),
        ("kernel_normalization", "float32_sum_to_one"),
        ("convolution", "tensorflow_depthwise_conv2d"),
        ("border_mode", "reflect_101_without_repeating_edge"),
        ("single_pixel_axis_border_mode", "repeat_only_pixel"),
        ("compute_dtype", "float32"),
    ),
)

GAUSSIAN_NOISE_DESCRIPTOR = CorruptionDescriptor(
    name="gaussian_noise",
    version="1.0.0",
    input_domain=_UINT8_INPUT,
    input_dtypes=("uint8",),
    output_domain=_UINT8_OUTPUT,
    output_shape="same_as_input_hwc",
    output_dtype="uint8",
    severity_values=_SEVERITIES,
    severity_parameters=tuple(
        (level, (("standard_deviation_unit_range", std),))
        for level, std in enumerate(GAUSSIAN_NOISE_STD_UNIT, start=1)
    ),
    uses_randomness=True,
    seed_contract=f"{_STATELESS_SEED}; tensorflow_philox",
    clipping_policy="clip_float32_to_closed_interval_0_255",
    rounding_policy="clip_then_round_half_to_even",
    implementation="justdata_tensorflow_stateless_gaussian_noise_v1",
    implementation_parameters=(
        ("random_distribution", "normal"),
        ("random_algorithm", "philox"),
        ("mean_unit_range", 0.0),
        ("noise_scale_to_uint8", 255.0),
        ("compute_dtype", "float32"),
        (
            "version_guarantee",
            "fixed_tensorflow_version_and_cpu_or_gpu_hardware",
        ),
    ),
)

JPEG_COMPRESSION_DESCRIPTOR = CorruptionDescriptor(
    name="jpeg_compression",
    version="1.0.0",
    input_domain=_UINT8_INPUT,
    input_dtypes=("uint8",),
    output_domain=_UINT8_OUTPUT,
    output_shape="same_as_input_hwc",
    output_dtype="uint8",
    severity_values=_SEVERITIES,
    severity_parameters=tuple(
        (level, (("quality", quality),))
        for level, quality in enumerate(JPEG_QUALITY, start=1)
    ),
    uses_randomness=False,
    seed_contract=_DETERMINISTIC_SEED,
    clipping_policy="codec_native_uint8_closed_interval_0_255",
    rounding_policy="codec_native_uint8_decode",
    implementation="justdata_tensorflow_jpeg_roundtrip_v1",
    implementation_parameters=(
        ("encode_color_mode", "rgb"),
        ("chroma_downsampling", True),
        ("optimize_size", False),
        ("progressive", False),
        ("decode_channels", 3),
        ("decode_ratio", 1),
        ("decode_fancy_upscaling", True),
        ("decode_try_recover_truncated", False),
        ("decode_acceptable_fraction", 1.0),
        ("decode_dct_method", "INTEGER_ACCURATE"),
        (
            "byte_guarantee",
            "repeatable_for_fixed_tensorflow_and_codec_build_not_cross_version",
        ),
    ),
)

CONTRAST_REDUCTION_DESCRIPTOR = CorruptionDescriptor(
    name="contrast_reduction",
    version="1.0.0",
    input_domain=_UINT8_INPUT,
    input_dtypes=("uint8",),
    output_domain=_UINT8_OUTPUT,
    output_shape="same_as_input_hwc",
    output_dtype="uint8",
    severity_values=_SEVERITIES,
    severity_parameters=tuple(
        (level, (("factor", factor), ("center_0_255", 127.5)))
        for level, factor in enumerate(CONTRAST_FACTOR, start=1)
    ),
    uses_randomness=False,
    seed_contract=_DETERMINISTIC_SEED,
    clipping_policy="clip_float32_to_closed_interval_0_255",
    rounding_policy="clip_then_round_half_to_even",
    implementation="justdata_tensorflow_contrast_reduction_v1",
    implementation_parameters=(("compute_dtype", "float32"),),
)

BRIGHTNESS_REDUCTION_DESCRIPTOR = CorruptionDescriptor(
    name="brightness_reduction",
    version="1.0.0",
    input_domain=_UINT8_INPUT,
    input_dtypes=("uint8",),
    output_domain=_UINT8_OUTPUT,
    output_shape="same_as_input_hwc",
    output_dtype="uint8",
    severity_values=_SEVERITIES,
    severity_parameters=tuple(
        (level, (("factor", factor),))
        for level, factor in enumerate(BRIGHTNESS_FACTOR, start=1)
    ),
    uses_randomness=False,
    seed_contract=_DETERMINISTIC_SEED,
    clipping_policy="clip_float32_to_closed_interval_0_255",
    rounding_policy="clip_then_round_half_to_even",
    implementation="justdata_tensorflow_brightness_reduction_v1",
    implementation_parameters=(("compute_dtype", "float32"),),
)


def _severity_index(severity: tf.Tensor) -> tf.Tensor:
    return tf.cast(severity, tf.int32) - 1


def _reflect_indices(length: tf.Tensor, radius: tf.Tensor) -> tf.Tensor:
    length = tf.cast(length, tf.int32)
    radius = tf.cast(radius, tf.int32)

    def repeat_single_pixel() -> tf.Tensor:
        return tf.zeros([length + 2 * radius], dtype=tf.int32)

    def reflect_101() -> tf.Tensor:
        positions = tf.range(-radius, length + radius)
        period = 2 * (length - 1)
        reflected = tf.math.floormod(positions, period)
        return tf.where(reflected < length, reflected, period - reflected)

    return tf.cond(length > 1, reflect_101, repeat_single_pixel)


def _reflect_pad_hwc(image: tf.Tensor, radius: tf.Tensor) -> tf.Tensor:
    y_indices = _reflect_indices(tf.shape(image)[0], radius)
    x_indices = _reflect_indices(tf.shape(image)[1], radius)
    image = tf.gather(image, y_indices, axis=0)
    return tf.gather(image, x_indices, axis=1)


@register_corruption("gaussian_blur", descriptor=GAUSSIAN_BLUR_DESCRIPTOR)
@tf.function
def gaussian_blur(
    image: tf.Tensor,
    severity: int | tf.Tensor,
    seed: tf.Tensor,
) -> tf.Tensor:
    del seed
    sigma = tf.gather(
        tf.constant(GAUSSIAN_BLUR_SIGMA, dtype=tf.float32),
        _severity_index(severity),
    )
    radius = tf.cast(tf.math.ceil(3.0 * sigma), tf.int32)
    positions = tf.cast(tf.range(-radius, radius + 1), tf.float32)
    kernel_1d = tf.exp(-tf.square(positions) / (2.0 * tf.square(sigma)))
    kernel_1d /= tf.reduce_sum(kernel_1d)
    kernel_2d = kernel_1d[:, tf.newaxis] * kernel_1d[tf.newaxis, :]
    kernel = kernel_2d[:, :, tf.newaxis, tf.newaxis]
    kernel = tf.tile(kernel, [1, 1, 3, 1])

    image_f = tf.cast(image, tf.float32)
    padded = _reflect_pad_hwc(image_f, radius)
    return tf.nn.depthwise_conv2d(
        padded[tf.newaxis, ...],
        kernel,
        strides=[1, 1, 1, 1],
        padding="VALID",
    )[0]


@register_corruption("gaussian_noise", descriptor=GAUSSIAN_NOISE_DESCRIPTOR)
@tf.function
def gaussian_noise(
    image: tf.Tensor,
    severity: int | tf.Tensor,
    seed: tf.Tensor,
) -> tf.Tensor:
    std_unit = tf.gather(
        tf.constant(GAUSSIAN_NOISE_STD_UNIT, dtype=tf.float32),
        _severity_index(severity),
    )
    noise = tf.random.stateless_normal(
        tf.shape(image),
        seed=seed,
        mean=0.0,
        stddev=std_unit * 255.0,
        dtype=tf.float32,
        alg="philox",
    )
    return tf.cast(image, tf.float32) + noise


def _jpeg_roundtrip(image: tf.Tensor, quality: int) -> tf.Tensor:
    encoded = tf.io.encode_jpeg(
        image,
        format="rgb",
        quality=quality,
        progressive=False,
        optimize_size=False,
        chroma_downsampling=True,
    )
    decoded = tf.io.decode_jpeg(
        encoded,
        channels=3,
        ratio=1,
        fancy_upscaling=True,
        try_recover_truncated=False,
        acceptable_fraction=1.0,
        dct_method="INTEGER_ACCURATE",
    )
    return tf.ensure_shape(decoded, image.shape)


@register_corruption("jpeg_compression", descriptor=JPEG_COMPRESSION_DESCRIPTOR)
@tf.function
def jpeg_compression(
    image: tf.Tensor,
    severity: int | tf.Tensor,
    seed: tf.Tensor,
) -> tf.Tensor:
    del seed
    branches = tuple(
        lambda quality=quality: _jpeg_roundtrip(image, quality)
        for quality in JPEG_QUALITY
    )
    return tf.switch_case(_severity_index(severity), branch_fns=branches)


@register_corruption(
    "contrast_reduction",
    descriptor=CONTRAST_REDUCTION_DESCRIPTOR,
)
@tf.function
def contrast_reduction(
    image: tf.Tensor,
    severity: int | tf.Tensor,
    seed: tf.Tensor,
) -> tf.Tensor:
    del seed
    factor = tf.gather(
        tf.constant(CONTRAST_FACTOR, dtype=tf.float32),
        _severity_index(severity),
    )
    image_f = tf.cast(image, tf.float32)
    return (image_f - 127.5) * factor + 127.5


@register_corruption(
    "brightness_reduction",
    descriptor=BRIGHTNESS_REDUCTION_DESCRIPTOR,
)
@tf.function
def brightness_reduction(
    image: tf.Tensor,
    severity: int | tf.Tensor,
    seed: tf.Tensor,
) -> tf.Tensor:
    del seed
    factor = tf.gather(
        tf.constant(BRIGHTNESS_FACTOR, dtype=tf.float32),
        _severity_index(severity),
    )
    return tf.cast(image, tf.float32) * factor


__all__ = [
    "BRIGHTNESS_FACTOR",
    "BRIGHTNESS_REDUCTION_DESCRIPTOR",
    "CONTRAST_FACTOR",
    "CONTRAST_REDUCTION_DESCRIPTOR",
    "GAUSSIAN_BLUR_DESCRIPTOR",
    "GAUSSIAN_BLUR_SIGMA",
    "GAUSSIAN_NOISE_DESCRIPTOR",
    "GAUSSIAN_NOISE_STD_UNIT",
    "JPEG_COMPRESSION_DESCRIPTOR",
    "JPEG_QUALITY",
    "brightness_reduction",
    "contrast_reduction",
    "gaussian_blur",
    "gaussian_noise",
    "jpeg_compression",
]

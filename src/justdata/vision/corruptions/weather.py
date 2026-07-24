import tensorflow as tf

from justdata.vision.corruptions.registry import (
    CorruptionDescriptor,
    _get_severity_index,
    register_corruption,
)
from justdata.vision.utils import _pad, _rotate, from_4d, to_4d

SNOW_PARAMS = (
    (0.1, 0.3),
    (0.2, 0.3),
    (0.55, 0.3),
    (0.55, 0.45),
    (0.85, 0.55),
)

WEATHER_DESCRIPTOR = CorruptionDescriptor(
    name="weather",
    version="1.0.0",
    input_domain="decoded_rgb_hwc_0_255",
    input_dtypes=("uint8", "float32"),
    output_domain="decoded_rgb_hwc_0_255",
    output_shape="same_as_input_hwc",
    output_dtype="uint8",
    severity_values=(1, 2, 3, 4, 5),
    severity_parameters=tuple(
        (
            level,
            (
                ("snow_intensity", snow_intensity),
                ("brightness_factor", brightness_factor),
            ),
        )
        for level, (snow_intensity, brightness_factor) in enumerate(
            SNOW_PARAMS,
            start=1,
        )
    ),
    uses_randomness=True,
    seed_contract=(
        "caller_supplied_int32_shape_2; tensorflow_stateless_split_then_normal; "
        "stable_for_fixed_tensorflow_version_and_supported_hardware"
    ),
    clipping_policy="clip_float32_to_closed_interval_0_255_after_compositing",
    rounding_policy="clip_then_truncate_toward_zero",
    implementation="justdata_minic_snow_tensorflow_v1",
    implementation_parameters=(
        ("snow_layer_scale", 0.25),
        ("snow_threshold_formula", "2.5_minus_0.5_times_snow_intensity"),
        ("resize_interpolation", "nearest"),
        ("motion_sigma_vertical", 4.0),
        ("motion_sigma_horizontal", 0.5),
        ("motion_kernel_vertical", 17),
        ("motion_kernel_horizontal", 3),
        ("motion_padding", "constant_zero"),
        ("rotation_degrees", -15.0),
        ("rotation_fill", 0),
        ("compute_dtype", "float32"),
    ),
)


@register_corruption("weather", descriptor=WEATHER_DESCRIPTOR)
@tf.function
def snow(image: tf.Tensor, severity: int, seed: tf.Tensor) -> tf.Tensor:
    """
    Simulates Snow by:
    1. Whitening/Fogging the image stats.
    2. Generating "snowflakes" via thresholded noise and motion blur.
    """
    idx = _get_severity_index(severity)
    params = tf.constant(SNOW_PARAMS, dtype=tf.float32)
    snow_intensity = tf.gather(params[:, 0], idx)
    brightness_factor = tf.gather(params[:, 1], idx)

    seeds = tf.random.split(seed, 2)
    image_f = tf.cast(image, tf.float32)

    h, w = tf.shape(image)[0], tf.shape(image)[1]

    # 1. Generate Snowflakes at lower resolution
    scale = 0.25
    h_snow, w_snow = (
        tf.cast(tf.cast(h, tf.float32) * scale, tf.int32),
        tf.cast(tf.cast(w, tf.float32) * scale, tf.int32),
    )

    snow_layer = tf.random.stateless_normal([h_snow, w_snow, 1], seed=seeds[0])
    snow_threshold = 2.5 - (snow_intensity * 0.5)
    snow_layer = tf.cast(snow_layer > snow_threshold, tf.float32)
    snow_layer = tf.image.resize(snow_layer, [h, w], method="nearest")

    # 2. Apply Asymmetric Gaussian Blur (Motion Blur)
    # Strong vertical blur (sigma=4.0) to create streaks
    # Weak horizontal blur (sigma=0.5) to give slight width
    sigma_motion = 4.0
    sigma_width = 0.5

    # Kernel sizes approx 4*sigma + 1
    k_motion = 17
    k_width = 3

    ky = _get_gaussian_kernel(sigma_motion, k_motion)
    kx = _get_gaussian_kernel(sigma_width, k_width)

    # Create 2D Separable Kernel [k_motion, k_width]
    # Shape: [k_motion, k_width]
    kernel_2d = tf.matmul(ky[:, tf.newaxis], kx[tf.newaxis, :])
    # Shape for depthwise conv: [H, W, In, Multiplier] -> [k_motion, k_width, 1, 1]
    kernel_4d = kernel_2d[:, :, tf.newaxis, tf.newaxis]

    # Pad snow layer to avoid border artifacts during blur
    snow_4d = to_4d(snow_layer)
    snow_padded = _pad(snow_4d, [k_motion, k_width], pad_mode="CONSTANT")

    snow_blur = tf.nn.depthwise_conv2d(
        snow_padded, kernel_4d, strides=[1, 1, 1, 1], padding="VALID"
    )

    # 3. Rotate to simulate falling angle (-15 degrees)
    # We rotate the streaks to make them fall diagonally
    # Note: _rotate automatically handles the 4D conversion internally if needed
    snow_blur_3d = from_4d(snow_blur, tf.rank(snow_layer))
    snow_final = _rotate(snow_blur_3d, degrees=-15.0, replace=0)

    # Scale intensity
    snow_final = snow_final * 255.0 * snow_intensity

    # 4. Whiten the original image
    image_whitened = image_f * (1.0 - brightness_factor) + (255.0 * brightness_factor)

    output = tf.clip_by_value(image_whitened + snow_final, 0.0, 255.0)

    return output


def _get_gaussian_kernel(sigma: float, filter_size: int) -> tf.Tensor:
    """Computes 1D Gaussian kernel."""
    x = tf.range(
        tf.cast(-filter_size // 2 + 1, tf.float32),
        tf.cast(filter_size // 2 + 1, tf.float32),
    )
    x = x**2
    x = tf.nn.softmax(-x / (2.0 * (sigma**2)))
    return x

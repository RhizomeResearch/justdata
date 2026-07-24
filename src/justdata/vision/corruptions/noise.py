import tensorflow as tf

from justdata.vision.corruptions.registry import (
    CorruptionDescriptor,
    _get_severity_index,
    register_corruption,
)

NOISE_STD = (8.0, 12.0, 18.0, 26.0, 38.0)

NOISE_DESCRIPTOR = CorruptionDescriptor(
    name="noise",
    version="1.0.0",
    input_domain="decoded_rgb_hwc_0_255",
    input_dtypes=("uint8", "float32"),
    output_domain="decoded_rgb_hwc_0_255",
    output_shape="same_as_input_hwc",
    output_dtype="uint8",
    severity_values=(1, 2, 3, 4, 5),
    severity_parameters=tuple(
        (level, (("standard_deviation_0_255", std),))
        for level, std in enumerate(NOISE_STD, start=1)
    ),
    uses_randomness=True,
    seed_contract=(
        "caller_supplied_int32_shape_2; tensorflow_stateless_normal_auto_select; "
        "stable_for_fixed_tensorflow_version_and_supported_hardware"
    ),
    clipping_policy="clip_float32_to_closed_interval_0_255",
    rounding_policy="clip_then_truncate_toward_zero",
    implementation="justdata_minic_gaussian_noise_tensorflow_v1",
)


@register_corruption("noise", descriptor=NOISE_DESCRIPTOR)
@tf.function
def gaussian_noise(image: tf.Tensor, severity: int, seed: tf.Tensor) -> tf.Tensor:
    """Applies Additive Gaussian Noise."""
    idx = _get_severity_index(severity)
    std = tf.gather(tf.constant(NOISE_STD, dtype=tf.float32), idx)

    image = tf.cast(image, tf.float32)
    noise = tf.random.stateless_normal(tf.shape(image), seed=seed, mean=0.0, stddev=std)
    return tf.clip_by_value(image + noise, 0.0, 255.0)

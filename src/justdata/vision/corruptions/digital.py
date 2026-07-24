import tensorflow as tf

from justdata.vision.corruptions.registry import (
    CorruptionDescriptor,
    _get_severity_index,
    register_corruption,
)

PIXELATE_RATIOS = (0.6, 0.5, 0.4, 0.3, 0.25)

DIGITAL_DESCRIPTOR = CorruptionDescriptor(
    name="digital",
    version="1.0.0",
    input_domain="decoded_rgb_hwc_0_255",
    input_dtypes=("uint8", "float32"),
    output_domain="decoded_rgb_hwc_0_255",
    output_shape="same_as_input_hwc",
    output_dtype="uint8",
    severity_values=(1, 2, 3, 4, 5),
    severity_parameters=tuple(
        (level, (("downsample_ratio", ratio),))
        for level, ratio in enumerate(PIXELATE_RATIOS, start=1)
    ),
    uses_randomness=False,
    seed_contract="caller_supplied_int32_shape_2; accepted_and_ignored",
    clipping_policy="clip_float32_to_closed_interval_0_255_after_resize",
    rounding_policy="clip_then_truncate_toward_zero",
    implementation="justdata_minic_nearest_pixelate_tensorflow_v1",
    implementation_parameters=(
        ("downsample_dimension_rounding", "truncate_toward_zero"),
        ("downsample_interpolation", "nearest"),
        ("upsample_interpolation", "nearest"),
    ),
)


@register_corruption("digital", descriptor=DIGITAL_DESCRIPTOR)
@tf.function
def pixelate(image: tf.Tensor, severity: int, seed: tf.Tensor) -> tf.Tensor:
    """Pixelates image by downsampling and nearest-neighbor upsampling."""
    idx = _get_severity_index(severity)
    ratio = tf.gather(tf.constant(PIXELATE_RATIOS, dtype=tf.float32), idx)

    shape = tf.shape(image)
    h, w = tf.cast(shape[0], tf.float32), tf.cast(shape[1], tf.float32)

    h_small = tf.cast(h * ratio, tf.int32)
    w_small = tf.cast(w * ratio, tf.int32)

    img_small = tf.image.resize(image, [h_small, w_small], method="nearest")

    return tf.image.resize(img_small, [shape[0], shape[1]], method="nearest")

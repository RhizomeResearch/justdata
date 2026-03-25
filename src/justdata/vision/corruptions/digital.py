import tensorflow as tf

from justdata.vision.corruptions.registry import _get_severity_index, register_corruption

PIXELATE_RATIOS = (0.6, 0.5, 0.4, 0.3, 0.25)


@register_corruption("digital")
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

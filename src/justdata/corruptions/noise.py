import tensorflow as tf

from justdata.corruptions.registry import _get_severity_index, register_corruption

NOISE_STD = (8.0, 12.0, 18.0, 26.0, 38.0)


@register_corruption("noise")
@tf.function
def gaussian_noise(image: tf.Tensor, severity: int, seed: tf.Tensor) -> tf.Tensor:
    """Applies Additive Gaussian Noise."""
    idx = _get_severity_index(severity)
    std = tf.gather(tf.constant(NOISE_STD, dtype=tf.float32), idx)

    image = tf.cast(image, tf.float32)
    noise = tf.random.stateless_normal(tf.shape(image), seed=seed, mean=0.0, stddev=std)
    return tf.clip_by_value(image + noise, 0.0, 255.0)

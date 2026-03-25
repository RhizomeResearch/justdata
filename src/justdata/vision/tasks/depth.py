import tensorflow as tf


@tf.function
def prepare_depth(
    depth: tf.Tensor,
    image_size: int,
    method: str = "nearest",
    normalize: bool = True,
    min_depth: float = 0.1,
    max_depth: float = 10.0,
):
    depth = tf.cast(depth, tf.float32)

    # Needed to avoid error with depth maps when resizing
    depth = tf.expand_dims(depth, axis=-1)
    depth = tf.image.resize(depth, [image_size, image_size], method=method)
    depth = tf.clip_by_value(depth, min_depth, max_depth)

    if normalize:
        depth = (depth - min_depth) / (max_depth - min_depth)

    depth = tf.squeeze(depth, axis=-1)

    return depth

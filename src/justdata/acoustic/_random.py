from __future__ import annotations

import tensorflow as tf


def _seed_tensor(seed: tf.Tensor | int | None) -> tf.Tensor:
    if seed is None:
        return tf.constant([0, 0], dtype=tf.int32)

    seed = tf.cast(tf.convert_to_tensor(seed), tf.int32)
    if seed.shape.rank == 0:
        return tf.stack([seed, tf.constant(0, dtype=tf.int32)])
    if seed.shape.rank == 1 and seed.shape[0] == 1:
        return tf.stack([seed[0], tf.constant(0, dtype=tf.int32)])
    return seed[:2]


def _uniform_float(seed: tf.Tensor, minval: float, maxval: float) -> tf.Tensor:
    if minval == maxval:
        return tf.constant(minval, dtype=tf.float32)
    return tf.random.stateless_uniform(
        [],
        seed=seed,
        minval=tf.cast(minval, tf.float32),
        maxval=tf.cast(maxval, tf.float32),
        dtype=tf.float32,
    )


def _uniform_int(seed: tf.Tensor, minval: tf.Tensor, maxval: tf.Tensor) -> tf.Tensor:
    minval = tf.cast(minval, tf.int32)
    maxval = tf.cast(maxval, tf.int32)
    return tf.cond(
        tf.equal(minval, maxval),
        lambda: minval,
        lambda: tf.random.stateless_uniform(
            [],
            seed=seed,
            minval=minval,
            maxval=maxval + 1,
            dtype=tf.int32,
        ),
    )


def _maybe_apply(x: tf.Tensor, prob: float, seed: tf.Tensor, apply_fn) -> tf.Tensor:
    if prob <= 0.0:
        return x
    if prob >= 1.0:
        return apply_fn()
    should_apply = tf.random.stateless_uniform([], seed=seed) < tf.cast(
        prob, tf.float32
    )
    return tf.cond(should_apply, apply_fn, lambda: x)

import numpy as np
import tensorflow as tf


def _tree_all_equal(left, right) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _tree_all_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (tuple, list)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            _tree_all_equal(l_val, r_val) for l_val, r_val in zip(left, right)
        )

    if hasattr(left, "numpy"):
        left = left.numpy()
    if hasattr(right, "numpy"):
        right = right.numpy()
    return np.array_equal(np.asarray(left), np.asarray(right))


def assert_same_seed_same_output(transform, input_value):
    seed = tf.constant([123, 7], dtype=tf.int32)

    first = transform(input_value, seed=seed, is_training=True)
    second = transform(input_value, seed=seed, is_training=True)

    assert _tree_all_equal(first, second)


def assert_different_seed_can_change_output(transform, input_value):
    baseline = transform(
        input_value,
        seed=tf.constant([123, 7], dtype=tf.int32),
        is_training=True,
    )

    for offset in range(1, 12):
        changed = transform(
            input_value,
            seed=tf.constant([123 + offset * 97, 7 + offset * 31], dtype=tf.int32),
            is_training=True,
        )
        if not _tree_all_equal(baseline, changed):
            return

    raise AssertionError("different stateless seeds did not change the transform output")


def assert_eval_disables_transform(transform, input_value):
    result = transform(
        input_value,
        seed=tf.constant([123, 7], dtype=tf.int32),
        is_training=False,
    )

    assert _tree_all_equal(result, input_value)

from __future__ import annotations

from typing import Literal

import tensorflow as tf


LabelMixMode = Literal["single_label", "multi_label", "event_frames"]


def apply_label_smoothing(
    labels: tf.Tensor,
    *,
    num_classes: int,
    label_smoothing: float,
) -> tf.Tensor:
    labels = tf.cast(labels, tf.float32)
    if label_smoothing <= 0.0:
        return labels
    off_value = label_smoothing / float(num_classes)
    return labels * (1.0 - label_smoothing) + off_value


def prepare_labels_for_mixing(
    labels: tf.Tensor,
    *,
    label_mode: LabelMixMode,
    num_classes: int | None = None,
    label_smoothing: float = 0.0,
) -> tf.Tensor:
    labels = tf.convert_to_tensor(labels)

    if label_mode == "single_label":
        if num_classes is None:
            raise ValueError("`num_classes` is required for single-label mixing.")

        is_dense = tf.logical_and(
            tf.greater_equal(tf.rank(labels), 2),
            tf.equal(tf.shape(labels)[-1], num_classes),
        )

        def dense() -> tf.Tensor:
            return apply_label_smoothing(
                labels,
                num_classes=num_classes,
                label_smoothing=label_smoothing,
            )

        def sparse() -> tf.Tensor:
            off_value = label_smoothing / float(num_classes)
            on_value = 1.0 - label_smoothing + off_value
            return tf.one_hot(
                tf.cast(tf.reshape(labels, [-1]), tf.int32),
                num_classes,
                on_value=on_value,
                off_value=off_value,
            )

        return tf.cond(is_dense, dense, sparse)

    if label_mode == "multi_label":
        return tf.cast(labels, tf.float32)

    if label_mode == "event_frames":
        return tf.cast(labels, tf.float32)

    raise ValueError(f"Unsupported label mixing mode: {label_mode!r}")


def broadcast_mix_coefficient(coefficient: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
    coefficient = tf.cast(tf.convert_to_tensor(coefficient), tf.float32)
    rank_delta = tf.maximum(tf.rank(labels) - tf.rank(coefficient), 0)
    target_shape = tf.concat(
        [tf.shape(coefficient), tf.ones([rank_delta], dtype=tf.int32)],
        axis=0,
    )
    return tf.reshape(coefficient, target_shape)


def blend_prepared_labels(
    labels: tf.Tensor,
    partner_labels: tf.Tensor,
    coefficient: tf.Tensor,
    *,
    bce_target: bool = False,
) -> tf.Tensor:
    labels = tf.cast(labels, tf.float32)
    partner_labels = tf.cast(partner_labels, tf.float32)
    if bce_target:
        return tf.maximum(labels, partner_labels)
    coefficient = broadcast_mix_coefficient(coefficient, labels)
    return coefficient * labels + (1.0 - coefficient) * partner_labels


def mix_labels(
    labels: tf.Tensor,
    partner_labels: tf.Tensor,
    coefficient: tf.Tensor,
    *,
    label_mode: LabelMixMode,
    num_classes: int | None = None,
    label_smoothing: float = 0.0,
    bce_target: bool = False,
) -> tf.Tensor:
    labels = prepare_labels_for_mixing(
        labels,
        label_mode=label_mode,
        num_classes=num_classes,
        label_smoothing=label_smoothing,
    )
    partner_labels = prepare_labels_for_mixing(
        partner_labels,
        label_mode=label_mode,
        num_classes=num_classes,
        label_smoothing=label_smoothing,
    )
    return blend_prepared_labels(labels, partner_labels, coefficient, bce_target=bce_target)


__all__ = [
    "LabelMixMode",
    "apply_label_smoothing",
    "blend_prepared_labels",
    "broadcast_mix_coefficient",
    "mix_labels",
    "prepare_labels_for_mixing",
]

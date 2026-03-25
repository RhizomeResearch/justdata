import numpy as np
import tensorflow as tf

from justdata.acoustic.configs import LabelTransformConfig
from justdata.acoustic.labels import transform_label


def test_label_index():
    label, _metadata = transform_label(3, LabelTransformConfig(mode="index"))

    assert label.dtype == tf.int64
    assert label.numpy() == 3


def test_label_one_hot():
    label, _metadata = transform_label(
        2,
        LabelTransformConfig(mode="one_hot", num_classes=4),
    )

    np.testing.assert_array_equal(label.numpy(), [0.0, 0.0, 1.0, 0.0])


def test_label_one_hot_smoothing():
    label, _metadata = transform_label(
        1,
        LabelTransformConfig(mode="one_hot", num_classes=4, smoothing=0.2),
    )

    np.testing.assert_allclose(label.numpy(), [0.05, 0.85, 0.05, 0.05])


def test_label_multi_hot_from_indices():
    label, _metadata = transform_label(
        [1, 3],
        LabelTransformConfig(mode="multi_hot", num_classes=5),
    )

    np.testing.assert_array_equal(label.numpy(), [0.0, 1.0, 0.0, 1.0, 0.0])


def test_label_multi_hot_from_indices_same_length_as_classes():
    label, _metadata = transform_label(
        [0, 1, 2],
        LabelTransformConfig(mode="multi_hot", num_classes=3),
    )

    np.testing.assert_array_equal(label.numpy(), [1.0, 1.0, 1.0])


def test_label_multi_hot_from_dense():
    label, _metadata = transform_label(
        [0, 1, 0, 1],
        LabelTransformConfig(mode="multi_hot", num_classes=4),
    )

    np.testing.assert_array_equal(label.numpy(), [0.0, 1.0, 0.0, 1.0])


def test_label_text_passthrough():
    label, _metadata = transform_label(
        "engine idling",
        LabelTransformConfig(mode="text"),
    )

    assert label.numpy() == b"engine idling"


def test_event_frames_alignment_to_hop_length():
    label, _metadata = transform_label(
        [{"start_time": 0.02, "end_time": 0.04, "class_id": 1}],
        LabelTransformConfig(
            mode="event_frames",
            num_classes=2,
            num_frames=6,
            hop_length=160,
            sample_rate=16000,
        ),
    )

    expected = np.zeros((6, 2), dtype=np.float32)
    expected[2:4, 1] = 1.0
    np.testing.assert_array_equal(label.numpy(), expected)


def test_hard_label_retained_in_metadata_eval():
    _label, metadata = transform_label(
        1,
        LabelTransformConfig(
            mode="one_hot",
            num_classes=2,
            class_names=("negative", "positive"),
        ),
    )

    assert metadata["hard_label"].numpy() == 1
    assert metadata["class_name"].numpy() == b"positive"


def test_soft_labels_allowed_train():
    label, metadata = transform_label(
        tf.constant([0.2, 0.7, 0.1], dtype=tf.float32),
        LabelTransformConfig(mode="one_hot", num_classes=3),
    )

    np.testing.assert_allclose(label.numpy(), [0.2, 0.7, 0.1])
    assert "hard_label" not in metadata

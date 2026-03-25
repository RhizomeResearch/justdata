import numpy as np

from justdata.acoustic.evals import (
    confusion_matrix,
    macro_classwise_accuracy,
    per_device_accuracy,
    per_scene_accuracy,
)


def test_macro_classwise_accuracy():
    y_true = np.array([0, 0, 1, 1, 2])
    logits = np.array(
        [
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 4.0],
        ]
    )

    assert np.isclose(
        macro_classwise_accuracy(y_true, logits, num_classes=10), 5.0 / 6.0
    )


def test_per_device_accuracy():
    result = per_device_accuracy([0, 1, 1], [0, 0, 1], ["A", "A", "B"])

    assert result == {"A": 0.5, "B": 1.0}


def test_per_scene_accuracy():
    result = per_scene_accuracy([0, 1, 1], [0, 0, 1], ["airport", "airport", "bus"])

    assert result == {"airport": 0.5, "bus": 1.0}


def test_confusion_matrix():
    matrix = confusion_matrix([0, 1, 1], [0, 0, 1], num_classes=2)

    np.testing.assert_array_equal(matrix, [[1, 0], [1, 1]])

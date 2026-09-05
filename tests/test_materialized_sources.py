from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from justdata.acoustic import dcase2025
from justdata.acoustic import sources as acoustic_sources
from justdata.vision import sources as vision_sources


def _local_records(labels):
    return [
        {
            "path": f"/missing/{index}.wav",
            "label": label,
            "split": "eval",
            "clip_id": str(index),
            "source_id": "device-a",
            "start_time": -0.0,
            "end_time": 0.1,
            "dataset": "local",
            "example_id": f"é-{index}",
            "filename": f"{index}.wav",
        }
        for index, label in enumerate(labels)
    ]


def _assert_datasets_equal(actual, expected):
    assert actual.element_spec == expected.element_spec
    assert actual.cardinality().numpy() == expected.cardinality().numpy()
    for _ in range(2):
        actual_records = list(actual.as_numpy_iterator())
        expected_records = list(expected.as_numpy_iterator())
        assert len(actual_records) == len(expected_records)
        for left, right in zip(actual_records, expected_records):
            tf.nest.assert_same_structure(left, right)
            for a, b in zip(tf.nest.flatten(left), tf.nest.flatten(right)):
                np.testing.assert_array_equal(a, b)
                if np.asarray(a).dtype.kind in "fiu":
                    assert np.asarray(a).tobytes() == np.asarray(b).tobytes()


def _legacy_dataset(monkeypatch, module, build):
    with monkeypatch.context() as patch:
        patch.setattr(
            module,
            "_dataset_from_materialized_records",
            tf.data.Dataset.from_generator,
            raising=False,
        )
        return build()


@pytest.mark.parametrize("labels", [[], [1, 2], ["airport"], [1, "park"]])
def test_local_records_match_generator(monkeypatch, labels):
    records = _local_records(labels)

    def build():
        return acoustic_sources._records_to_dataset(records)

    expected = _legacy_dataset(monkeypatch, acoustic_sources, build)
    _assert_datasets_equal(build(), expected)


@pytest.mark.parametrize("split", ["dev_train_25", "dev_test", "eval", "absent"])
def test_dcase_records_match_generator(monkeypatch, split):
    fixture = Path(__file__).parent / "acoustic" / "fixtures" / "dcase2025"

    def build():
        return dcase2025.load_dcase2025_task1_splits(
            "dcase2025_task1", [split], fixture
        )[0]

    expected = _legacy_dataset(monkeypatch, dcase2025, build)
    _assert_datasets_equal(build(), expected)


def _vision_records(path):
    return [
        {
            "_path": str(path),
            "label": np.int64(index),
            "metadata": {
                "dataset": "zenodo:123",
                "record_id": "123",
                "archive": "images.zip",
                "split": "eval",
                "class_name": f"class-{index}",
                "class_index": np.int64(index),
                "filename": path.name,
                "path": str(path),
                "example_id": f"é-{index}",
            },
        }
        for index in range(2)
    ]


@pytest.mark.parametrize("empty", [False, True])
def test_vision_records_match_generator(monkeypatch, tmp_path, empty):
    path = tmp_path / "image.png"
    path.write_bytes(tf.io.encode_png(tf.ones([8, 10, 3], tf.uint8)).numpy())
    records = [] if empty else _vision_records(path)

    def build():
        return vision_sources._records_to_vision_dataset(records)

    expected = _legacy_dataset(monkeypatch, vision_sources, build)
    _assert_datasets_equal(build(), expected)


def test_materialized_source_has_no_python_generator():
    dataset = acoustic_sources._records_to_dataset(_local_records([1, 2]))
    graph = tf.compat.v1.GraphDef()
    graph.ParseFromString(dataset._as_serialized_graph().numpy())
    nodes = list(graph.node)
    for function in graph.library.function:
        nodes.extend(function.node_def)
    assert not {"GeneratorDataset", "PyFunc", "EagerPyFunc"}.intersection(
        node.op for node in nodes
    )


def test_invalid_label_keeps_iteration_time_error(monkeypatch):
    records = _local_records([1, 2**100])

    def build():
        return acoustic_sources._records_to_dataset(records)

    expected = _legacy_dataset(monkeypatch, acoustic_sources, build)
    actual = build()
    for dataset in (expected, actual):
        iterator = iter(dataset)
        assert next(iterator)["label"].numpy() == 1
        with pytest.raises(tf.errors.UnknownError, match="OverflowError"):
            next(iterator)


def test_vision_decoding_remains_lazy_and_repeats(tmp_path):
    path = tmp_path / "image.png"
    dataset = vision_sources._records_to_vision_dataset(_vision_records(path))
    with pytest.raises(tf.errors.NotFoundError):
        next(iter(dataset))
    for pixel in (11, 29):
        path.write_bytes(
            tf.io.encode_png(tf.fill([8, 10, 3], tf.constant(pixel, tf.uint8))).numpy()
        )
        for sample in dataset:
            np.testing.assert_array_equal(sample["image"].numpy(), pixel)

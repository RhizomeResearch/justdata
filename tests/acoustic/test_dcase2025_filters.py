import tensorflow as tf

from justdata.acoustic.filters import filter_by_metadata, groupby_metadata


def _dataset():
    return tf.data.Dataset.from_tensor_slices(
        {
            "x": [1, 2, 3, 4],
            "metadata": {
                "device": ["A", "B", "S1", "A"],
                "device_type": ["real", "real", "simulated", "real"],
                "is_known_device": [True, True, True, True],
                "scene_label": ["airport", "airport", "bus", "bus"],
                "city": ["Paris", "London", "Paris", "Rome"],
            },
        }
    )


def _values(ds):
    return [int(sample["x"].numpy()) for sample in ds]


def test_filter_device():
    assert _values(filter_by_metadata(_dataset(), device="A")) == [1, 4]


def test_filter_device_in():
    assert _values(filter_by_metadata(_dataset(), device_in=["B", "S1"])) == [2, 3]


def test_filter_device_type():
    assert _values(filter_by_metadata(_dataset(), device_type="simulated")) == [3]


def test_filter_known_device():
    assert _values(filter_by_metadata(_dataset(), is_known_device=True)) == [1, 2, 3, 4]


def test_groupby_device_counts():
    groups = groupby_metadata(_dataset(), "device")

    assert {key: len(_values(ds)) for key, ds in groups.items()} == {
        "A": 2,
        "B": 1,
        "S1": 1,
    }


def test_groupby_device_scene_counts():
    groups = groupby_metadata(_dataset(), ["device", "scene_label"])

    assert {key: len(_values(ds)) for key, ds in groups.items()} == {
        ("A", "airport"): 1,
        ("B", "airport"): 1,
        ("S1", "bus"): 1,
        ("A", "bus"): 1,
    }

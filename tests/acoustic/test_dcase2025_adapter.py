import tensorflow as tf

from justdata.acoustic.dcase2025 import DCASE2025Task1Adapter


def _adapt(**row):
    sample = {
        "waveform": tf.zeros([8], dtype=tf.float32),
        "sample_rate": tf.constant(8, dtype=tf.int32),
        "split": "dev_train_25",
        **row,
    }
    return DCASE2025Task1Adapter()(sample)


def _value(value):
    if hasattr(value, "numpy"):
        value = value.numpy()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def test_dcase_adapter_parses_scene_label():
    sample = _adapt(filename="shopping_mall-Paris-loc1-0001-A.wav")

    assert _value(sample["metadata"]["scene_label"]) == "shopping_mall"
    assert _value(sample["metadata"]["scene_id"]) == 1


def test_dcase_adapter_parses_device():
    sample = _adapt(filename="airport-Paris-loc1-0001-S3.wav")

    assert _value(sample["metadata"]["device"]) == "S3"


def test_dcase_adapter_marks_known_devices():
    known = _adapt(device="S3")
    unknown = _adapt(device="S7")

    assert _value(known["metadata"]["is_known_device"]) is True
    assert _value(unknown["metadata"]["is_known_device"]) is False


def test_dcase_adapter_marks_unknown_eval_devices():
    sample = DCASE2025Task1Adapter()(
        {"filename": "evaluation-file.wav", "split": "eval", "device": "unknown"}
    )

    assert sample["metadata"]["device"] == "unknown"
    assert sample["metadata"]["device_type"] == "unknown"
    assert sample["metadata"]["is_known_device"] is False


def test_dcase_adapter_sets_device_type_real():
    sample = _adapt(device="A")

    assert _value(sample["metadata"]["device_type"]) == "real"


def test_dcase_adapter_sets_device_type_simulated():
    sample = _adapt(device="S10")

    assert _value(sample["metadata"]["device_type"]) == "simulated"

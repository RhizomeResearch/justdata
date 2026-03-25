import tensorflow as tf

from justdata.acoustic import dcase2025


def test_make_source_dataset_uses_dev_train_25(monkeypatch):
    calls = {}

    def fake_load_ds(dataset, split, dataset_type, batch_size, seed, **kwargs):
        calls.update(
            {
                "dataset": dataset,
                "split": split,
                "dataset_type": dataset_type,
                "batch_size": batch_size,
                "seed": seed,
                "kwargs": kwargs,
            }
        )
        return tf.data.Dataset.from_tensors({"x": 1}), tf.constant(1)

    monkeypatch.setattr(dcase2025, "load_ds", fake_load_ds)

    dcase2025.make_source_dataset(batch_size=8, seed=123)

    assert calls["dataset"] == "dcase2025_task1"
    assert calls["split"] == "dev_train_25"
    assert calls["dataset_type"] == "train"
    assert calls["batch_size"] == 8
    assert calls["seed"] == 123


def test_make_target_dataset_default_disallows_stats(monkeypatch):
    calls = {}

    def fake_load_ds(dataset, split, dataset_type, batch_size, seed, **kwargs):
        calls["split"] = split
        calls["dataset_type"] = dataset_type
        return tf.data.Dataset.from_tensors({"x": 1}), tf.constant(1)

    monkeypatch.setattr(dcase2025, "load_ds", fake_load_ds)

    dcase2025.make_target_dataset()

    assert calls == {"split": "dev_test", "dataset_type": "validation"}

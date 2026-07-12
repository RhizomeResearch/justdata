from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from justdata.acoustic import dcase2025


class _IdentityPipeline:
    kwargs = {}

    def build(self, *, is_training):
        del is_training

        def identity(sample, *args, **kwargs):
            del args, kwargs
            return sample

        return identity, identity, identity, identity


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


@pytest.mark.parametrize(
    ("builder", "domain_arg", "domain"),
    [
        (dcase2025.make_source_dataset, "source_domain", {"device": "A"}),
        (dcase2025.make_target_dataset, "target_domain", {"device": "B"}),
    ],
)
def test_domain_and_caller_filters_use_independent_hooks(
    monkeypatch, builder, domain_arg, domain
):
    calls = {}
    source_filter_fn = lambda sample: tf.equal(sample["split"], "selected")
    filter_fn = lambda sample: tf.equal(sample["features"], 1)

    def fake_load_ds(dataset, split, dataset_type, batch_size, seed, **kwargs):
        del dataset, split, dataset_type, batch_size, seed
        calls.update(kwargs)
        return tf.data.Dataset.from_tensors({"x": 1}), tf.constant(1)

    monkeypatch.setattr(dcase2025, "load_ds", fake_load_ds)

    builder(
        **{
            domain_arg: domain,
            "source_filter_fn": source_filter_fn,
            "filter_fn": filter_fn,
        }
    )

    assert calls["filter_fn"] is filter_fn
    combined_source_filter = calls["source_filter_fn"]
    expected_device = domain["device"]
    assert bool(
        combined_source_filter(
            {"split": tf.constant("selected"), "device": tf.constant(expected_device)}
        ).numpy()
    )
    assert not bool(
        combined_source_filter(
            {"split": tf.constant("other"), "device": tf.constant(expected_device)}
        ).numpy()
    )
    assert not bool(
        combined_source_filter(
            {"split": tf.constant("selected"), "device": tf.constant("other")}
        ).numpy()
    )


@pytest.mark.parametrize(
    ("builder", "split"),
    [
        (dcase2025.make_source_dataset, "dev_train_25"),
        (dcase2025.make_target_dataset, "dev_test"),
    ],
)
def test_domain_filter_skips_rejected_wav_decoding(
    tmp_path, write_wav_file, monkeypatch, builder, split
):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    retained_paths = [audio_dir / "airport-A.wav", audio_dir / "bus-A.wav"]
    rejected_path = audio_dir / "park-B.wav"
    for path in [*retained_paths, rejected_path]:
        write_wav_file(path, np.arange(8, dtype=np.int16))

    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "path,split,scene_label,device,example_id\n"
        f"audio/airport-A.wav,{split},airport,A,first\n"
        f"audio/park-B.wav,{split},park,B,rejected\n"
        f"audio/bus-A.wav,{split},bus,A,last\n",
        encoding="utf-8",
    )

    decoded_paths = []

    def counted_decode(path):
        def open_file(path_value):
            decoded_path = Path(path_value.numpy().decode("utf-8"))
            decoded_path.read_bytes()
            decoded_paths.append(decoded_path)
            return np.zeros((8, 1), dtype=np.float32), np.int32(8000)

        waveform, sample_rate = tf.py_function(
            open_file, [path], Tout=[tf.float32, tf.int32]
        )
        waveform.set_shape([None, 1])
        sample_rate.set_shape([])
        return waveform, sample_rate

    monkeypatch.setattr(
        "justdata.acoustic.adapters.decode_audio_file", counted_decode
    )

    domain_kwargs = (
        {"source_domain": {"device": "A"}}
        if builder is dcase2025.make_source_dataset
        else {"target_domain": {"device": "A"}}
    )
    ds, _tools = builder(
        dataset="dcase2025_task1",
        split=split,
        data_dir=tmp_path,
        pipeline=_IdentityPipeline(),
        batch_size=2,
        cache_dataset=True,
        return_raw_ds=True,
        **domain_kwargs,
    )

    def snapshot():
        return [
            (
                sample["metadata"]["example_id"].numpy().decode("utf-8"),
                int(sample["label"].numpy()),
            )
            for sample in ds
        ]

    assert snapshot() == [("first", 0), ("last", 7)]
    assert snapshot() == [("first", 0), ("last", 7)]
    assert len(decoded_paths) == 2
    assert set(decoded_paths) == set(retained_paths)

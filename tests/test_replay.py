"""Replay from verified local inventories and committed batch positions."""

import numpy as np
import pytest
import tensorflow as tf

from justdata.core import (
    DataPipeline,
    InventoryRecord,
    InventorySource,
    MetadataSidecar,
    ReplayError,
    ReplayState,
    admit_inventory,
    count_real_examples,
    finalize_dataset,
    load_replay_epoch,
    open_inventory,
    register_pipeline,
    stable_int64_hash,
)


def _resolve(config, is_training):
    variant = config.get("variant", 0)
    return {
        "implementation": "tests/replay-pure-v1",
        "configuration": {"variant": variant},
        "stages": {
            "preprocess": {"active": True, "config": {}},
            "augment": {"active": is_training, "config": {"variant": variant}},
            "late_augment": {"active": is_training, "config": {}},
            "postprocess": {"active": True, "config": {}},
        },
        "model_input": {"output_key": "x", "dtype": "int32"},
    }


@register_pipeline("tests/replay-pure", config_resolver=_resolve)
def _pure_pipeline(variant=0, **kwargs):
    del kwargs

    def preprocess(sample):
        return sample

    def augment(sample, seed=None):
        return sample | {
            "x": sample["x"] + variant,
            "sample_random": tf.random.stateless_uniform([], seed=seed),
        }

    def late_augment(batch, num_classes=None, seed=None):
        del num_classes
        return batch | {
            "batch_random": tf.random.stateless_uniform(tf.shape(batch["x"]), seed=seed)
        }

    def postprocess(sample, num_classes=None):
        del num_classes
        return sample

    return preprocess, augment, late_augment, postprocess


def _opaque_config(config, is_training):
    resolved = _resolve(config, is_training)
    resolved["configuration"]["opaque"] = object()
    return resolved


register_pipeline("tests/replay-opaque", config_resolver=_opaque_config)(_pure_pipeline)


def _identity(sample, num_classes=None):
    return sample


def _positive(sample):
    return sample["x"] > 0


def _admitted(tmp_path, count=5, *, reverse=False):
    def reader(_payloads, metadata):
        value = metadata["value"]
        ignored = value == 0
        return {
            "x": tf.constant(value, tf.int32),
            "mask": tf.fill([2, 2], 255 if ignored else value % 3),
            "pixel_valid_mask": tf.fill([2, 2], not ignored),
        }

    records = [
        InventoryRecord(f"row-{index}", metadata={"value": index})
        for index in range(count)
    ]
    if reverse:
        records.reverse()
    return admit_inventory(
        {"local": InventorySource({"train": records}, reader)},
        {"local": ["train"]},
        inventory_id=f"replay-{count}",
        snapshot_dir=tmp_path / "snapshot",
        output_signature={
            "x": tf.TensorSpec([], tf.int32),
            "mask": tf.TensorSpec([2, 2], tf.int32),
            "pixel_valid_mask": tf.TensorSpec([2, 2], tf.bool),
        },
    )


def _load(admitted, *, dataset_type="train", batch_size=2, seed=17, **kwargs):
    return load_replay_epoch(
        admitted,
        dataset_type,
        batch_size,
        seed,
        pipeline=kwargs.pop("pipeline", DataPipeline("tests/replay-pure")),
        epoch=kwargs.pop("epoch", 0),
        callbacks_are_stateless=kwargs.pop("callbacks_are_stateless", True),
        **kwargs,
    )


def _rows(batches):
    return [
        batch["metadata"]["example_id"][index].decode()
        for batch in batches
        for index, valid in enumerate(batch["padding_mask"])
        if valid
    ]


def test_resume_skips_only_committed_batches_after_complete_pipeline(tmp_path):
    admitted = _admitted(tmp_path)
    full = _load(admitted, batch_size=2, map_parallel_calls=3)
    expected = list(full.batches.as_numpy_iterator())
    interrupted = _load(admitted, batch_size=2, map_parallel_calls=3)
    iterator = iter(interrupted.batches)
    next(iterator)  # committed
    next(iterator)  # read but not committed

    saved = ReplayState.from_json(interrupted.state.with_next_batch(1).to_json())
    restarted = _load(
        open_inventory(admitted.path),
        batch_size=2,
        map_parallel_calls=3,
        state=saved,
        as_numpy=True,
    )
    actual = list(restarted.batches)
    assert restarted.remaining_batches == 2
    assert restarted.remaining_examples == 3
    assert restarted.state == saved
    assert _rows(actual) == _rows(expected[1:])
    for resumed_batch, expected_batch in zip(actual, expected[1:], strict=True):
        tf.nest.map_structure(
            np.testing.assert_array_equal, resumed_batch, expected_batch
        )
    assert sorted(_rows(expected)) == [f"row-{index}" for index in range(5)]
    assert int(count_real_examples(actual[-1])) == 1
    np.testing.assert_array_equal(actual[-1]["padding_mask"], [True, False])

    completed = _load(admitted, map_parallel_calls=3, state=saved.with_next_batch(3))
    assert completed.remaining_batches == completed.remaining_examples == 0
    assert list(completed.batches) == []


def test_evaluation_keeps_partial_batch_and_ignored_pixels(tmp_path):
    admitted = _admitted(tmp_path)
    epoch = _load(admitted, dataset_type="validation", batch_size=4)
    batches = list(epoch.batches.as_numpy_iterator())
    assert epoch.remaining_batches == 2
    assert epoch.remaining_examples == 5
    assert _rows(batches) == [f"row-{index}" for index in range(5)]
    assert int(count_real_examples(batches[-1])) == 1
    np.testing.assert_array_equal(
        batches[-1]["padding_mask"], [True, False, False, False]
    )
    assert batches[0]["pixel_valid_mask"][0].sum() == 0
    assert batches[0]["mask"][0, 0, 0] == 255
    assert not np.any(
        batches[0]["pixel_valid_mask"]
        & batches[0]["padding_mask"][:, None, None]
        & (batches[0]["mask"] == 255)
    )
    assert batches[-1]["mask"][1:].sum() == 0
    with pytest.raises(ReplayError) as error:
        _load(admitted, dataset_type="validation", batch_size=4, drop_remainder=True)
    assert error.value.status == "unsupported_replay"
    assert error.value.reason == "evaluation_drop"


def test_replay_binds_inventory_config_seed_view_batch_and_runtime(tmp_path):
    admitted = _admitted(tmp_path)
    first = _load(admitted)
    state = first.state.with_next_batch(1)
    for change in (
        {"seed": 18},
        {"epoch": 1},
        {"view": 1},
        {"batch_size": 3},
        {"map_parallel_calls": 2},
        {"pipeline": DataPipeline("tests/replay-pure", variant=1)},
        {"state": ReplayState.from_dict(state.to_dict() | {"python_version": "0"})},
    ):
        with pytest.raises(ReplayError) as error:
            _load(admitted, **({"state": state} | change))
        assert error.value.status == "incompatible_replay"
    with pytest.raises(ReplayError, match="must advance"):
        state.with_next_batch(0)
    with pytest.raises(ReplayError, match="must advance"):
        state.with_next_batch(4)
    another = tmp_path / "another"
    another.mkdir()
    changed_inventory = _admitted(another, reverse=True)
    with pytest.raises(ReplayError) as error:
        _load(changed_inventory, state=state)
    assert "inventory_digest" in str(error.value)

    other_view = _load(admitted, view=1)
    assert other_view.state.config_digest != first.state.config_digest
    assert other_view.state.inventory_digest == first.state.inventory_digest


def test_immutable_sidecar_rejoins_real_rows_after_resume(tmp_path):
    admitted = _admitted(tmp_path)
    sidecar = MetadataSidecar.from_inventory(admitted)
    first = _load(
        admitted,
        metadata_mode="numeric_only",
        metadata_sidecar=sidecar,
        batch_size=3,
    )
    resumed = _load(
        open_inventory(admitted.path),
        metadata_mode="numeric_only",
        metadata_sidecar=sidecar,
        batch_size=3,
        state=first.state.with_next_batch(1),
    )
    batch = next(iter(resumed.batches))
    real_keys = batch["metadata"]["row_id"].numpy()[batch["padding_mask"].numpy()]
    assert len(real_keys) == 2
    for key in real_keys:
        assert int(key) in sidecar.records
    assert set(real_keys) == {
        stable_int64_hash("row-0"),
        stable_int64_hash("row-1"),
        stable_int64_hash("row-2"),
        stable_int64_hash("row-3"),
        stable_int64_hash("row-4"),
    } - set(next(iter(first.batches))["metadata"]["row_id"].numpy().tolist())


def test_unqualified_callbacks_and_persistent_cache_report_status(tmp_path):
    admitted = _admitted(tmp_path)
    with pytest.raises(ReplayError) as error:
        _load(admitted, callbacks_are_stateless=False)
    assert (error.value.status, error.value.reason) == (
        "unsupported_replay",
        "callbacks_unqualified",
    )
    with pytest.raises(ReplayError) as error:
        _load(admitted, cache_dataset=True, cache_path=str(tmp_path / "cache"))
    assert error.value.reason == "persistent_cache"
    with pytest.raises(ReplayError) as error:
        _load(admitted, cache_model_inputs=True)
    assert error.value.reason == "augmented_model_input_cache"
    with pytest.raises(ReplayError) as error:
        _load(admitted, sidecar_metadata_path=str(tmp_path / "sidecar.json"))
    assert error.value.reason == "streaming_sidecar"
    with pytest.raises(ReplayError) as error:
        _load(admitted, pipeline=DataPipeline("tests/replay-opaque"))
    assert (error.value.status, error.value.reason) == (
        "unsupported_replay",
        "configuration_unavailable",
    )
    with pytest.raises(TypeError, match="must be boolean"):
        _load(admitted, callbacks_are_stateless="yes")


def test_empty_inventory_and_explicit_training_drop(tmp_path):
    empty = _admitted(tmp_path, count=0)
    result = _load(empty, batch_size=4)
    assert result.remaining_batches == result.remaining_examples == 0
    assert list(result.batches) == []

    (tmp_path / "other").mkdir()
    other = _admitted(tmp_path / "other", count=5)
    dropped = _load(other, batch_size=4, drop_remainder=True)
    assert dropped.remaining_batches == 1
    assert dropped.remaining_examples == 4
    assert int(count_real_examples(next(iter(dropped.batches)))) == 4


def test_real_example_count_with_unknown_dataset_cardinality():
    source = tf.data.Dataset.from_tensor_slices({"x": [0, 1, 2, 3]})
    source = source.filter(_positive)

    batches, n_batches = finalize_dataset(
        source,
        postprocess_fn=_identity,
        num_classes=None,
        batch_size=4,
        prefetch=False,
    )
    assert n_batches is None
    batch = next(iter(batches))
    assert int(count_real_examples(batch)) == 3
    assert (
        int(count_real_examples({"padding_mask": batch["padding_mask"].numpy()})) == 3
    )

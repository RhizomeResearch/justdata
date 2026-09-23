"""Protected cache identity, completion, and bounded execution."""

import json
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest
import tensorflow as tf

from justdata.core import (
    CacheError,
    CachePolicy,
    inspect_cache,
    load_ds,
    register_pipeline,
)
from justdata.core.finalization import finalize_dataset


def _resolve(config, is_training):
    geometry = config.get("geometry", 1)
    augmentation_policy = config.get("augmentation_policy", "none")
    model_identity = config.get("model_identity", "model-a")
    return {
        "implementation": "tests/cache-v1",
        "configuration": {
            "offset": config.get("offset", 0),
            "geometry": geometry,
            "augmentation_policy": augmentation_policy,
            "model_identity": model_identity,
        },
        "stages": {
            "preprocess": {
                "active": True,
                "config": {"offset": config.get("offset", 0), "geometry": geometry},
            },
            "augment": {
                "active": is_training,
                "config": {"policy": augmentation_policy},
            },
            "late_augment": {"active": False, "config": {}},
            "postprocess": {"active": True, "config": {}},
        },
        "model_input": {
            "output_key": "x",
            "dtype": "int64",
            "model_identity": model_identity,
        },
    }


@register_pipeline("tests/cache-execution", config_resolver=_resolve)
def _pipeline(offset=0, geometry=1, **kwargs):
    del kwargs

    def preprocess(sample):
        return sample | {"x": sample["x"] * geometry + offset}

    def augment(sample, seed=None):
        return sample

    def postprocess(sample, num_classes=None):
        return sample

    return preprocess, augment, None, postprocess


def _source(values):
    return tf.data.Dataset.from_tensor_slices({"x": np.asarray(values, np.int64)})


def _load(source, pipeline, **options):
    with patch("justdata.core.loader.fetch_ds", return_value=source):
        return load_ds(
            "fixture",
            "validation",
            "validation",
            2,
            17,
            pipeline=pipeline,
            deterministic=True,
            map_parallel_calls=1,
            private_threadpool_size=1,
            max_intra_op_parallelism=1,
            prefetch=1,
            **options,
        )


def _policy(identity="input-v1", *, mode="eager", max_examples=8, max_bytes=100_000):
    return CachePolicy(
        input_identity=identity,
        max_examples=max_examples,
        max_bytes=max_bytes,
        materialization=mode,
        callbacks_are_deterministic=True,
    )


def test_eager_cache_reuse_and_changed_input_and_preprocessing(tmp_path):
    from justdata.core.registry import get_pipeline

    path = str(tmp_path / "prepared")
    pipeline = get_pipeline(pipeline_name="tests/cache-execution", offset=3)
    policy = _policy()
    ds, n_batches, config = _load(
        _source([1, 2, 3]),
        pipeline,
        cache_dataset=True,
        cache_path=path,
        cache_policy=policy,
        return_config=True,
    )
    assert n_batches == 2
    assert [int(v) for batch in ds for v in batch["x"].numpy()[:2]][:3] == [4, 5, 6]
    assert policy.status("preprocess")["state"] == "complete"
    assert config.to_dict()["execution"]["limits"]["prefetch"] == 1
    assert inspect_cache(path)["count"] == 3

    reused, _ = _load(
        _source([90, 90, 90]),
        pipeline,
        cache_dataset=True,
        cache_path=path,
        cache_policy=_policy(),
    )
    assert int(next(iter(reused))["x"][0]) == 4
    with pytest.raises(CacheError) as changed_input:
        _load(
            _source([1, 2, 3]),
            pipeline,
            cache_dataset=True,
            cache_path=path,
            cache_policy=_policy("input-v2"),
        )
    assert changed_input.value.code == "incompatible"
    altered = get_pipeline(pipeline_name="tests/cache-execution", offset=4)
    with pytest.raises(CacheError, match="identity"):
        _load(
            _source([1, 2, 3]),
            altered,
            cache_dataset=True,
            cache_path=path,
            cache_policy=_policy(),
        )
    for change in (
        {"geometry": 2},
        {"augmentation_policy": "flips"},
        {"model_identity": "model-b"},
    ):
        changed = get_pipeline(
            pipeline_name="tests/cache-execution", offset=3, **change
        )
        with pytest.raises(CacheError) as mismatch:
            _load(
                _source([1, 2, 3]),
                changed,
                cache_dataset=True,
                cache_path=path,
                cache_policy=_policy(),
            )
        assert mismatch.value.code == "incompatible"

    (tmp_path / "prepared" / "records.tfrecord").write_bytes(b"corrupt")
    with pytest.raises(CacheError) as corrupt:
        _load(
            _source([1, 2, 3]),
            pipeline,
            cache_dataset=True,
            cache_path=path,
            cache_policy=_policy(),
        )
    assert corrupt.value.code == "corrupt"


def test_lazy_cache_completion_and_interruption(tmp_path):
    from justdata.core.registry import get_pipeline

    pipeline = get_pipeline(pipeline_name="tests/cache-execution")
    path = str(tmp_path / "lazy")
    policy = _policy(mode="lazy")
    raw, _ = _load(
        _source([1, 2, 3]),
        pipeline,
        cache_dataset=True,
        cache_path=path,
        cache_policy=policy,
        return_raw_ds=True,
    )
    assert policy.status("preprocess")["state"] == "missing"
    assert [int(row["x"]) for row in raw] == [1, 2, 3]
    assert policy.status("preprocess")["state"] == "complete"
    assert [int(row["x"]) for row in raw] == [1, 2, 3]

    partial_path = str(tmp_path / "partial")
    partial, _ = _load(
        _source([1, 2, 3]),
        pipeline,
        cache_dataset=True,
        cache_path=partial_path,
        cache_policy=_policy(mode="lazy"),
        return_raw_ds=True,
    )
    iterator = iter(partial)
    next(iterator)
    del iterator
    assert inspect_cache(partial_path)["state"] == "incomplete"
    with pytest.raises(CacheError) as error:
        _load(
            _source([1, 2, 3]),
            pipeline,
            cache_dataset=True,
            cache_path=partial_path,
            cache_policy=_policy(),
        )
    assert error.value.code == "incomplete"


def test_lazy_cache_second_writer_reports_ownership(tmp_path):
    from justdata.core.registry import get_pipeline

    pipeline = get_pipeline(pipeline_name="tests/cache-execution")
    path = str(tmp_path / "concurrent")
    first, _ = _load(
        _source([1, 2, 3]),
        pipeline,
        cache_dataset=True,
        cache_path=path,
        cache_policy=_policy(mode="lazy"),
        return_raw_ds=True,
    )
    second_policy = _policy(mode="lazy")
    second, _ = _load(
        _source([1, 2, 3]),
        pipeline,
        cache_dataset=True,
        cache_path=path,
        cache_policy=second_policy,
        return_raw_ds=True,
    )
    first_iterator = iter(first)
    next(first_iterator)
    with pytest.raises(tf.errors.OpError):
        next(iter(second))
    assert second_policy.status("preprocess")["failure_code"] == "incomplete"
    del first_iterator


def test_cache_quota_and_model_input_stage(tmp_path):
    from justdata.core.registry import get_pipeline

    pipeline = get_pipeline(pipeline_name="tests/cache-execution")
    too_small = _policy(max_examples=2)
    with pytest.raises(CacheError) as error:
        _load(
            _source([1, 2, 3]),
            pipeline,
            cache_dataset=True,
            cache_path=str(tmp_path / "limited"),
            cache_policy=too_small,
        )
    assert error.value.code == "quota_exceeded"
    assert too_small.status("preprocess")["failure_code"] == "quota_exceeded"
    assert inspect_cache(tmp_path / "limited")["failure_code"] == "quota_exceeded"

    lazy_policy = _policy(mode="lazy", max_examples=2)
    lazy, _ = _load(
        _source([1, 2, 3]),
        pipeline,
        cache_dataset=True,
        cache_path=str(tmp_path / "limited-lazy"),
        cache_policy=lazy_policy,
        return_raw_ds=True,
    )
    with pytest.raises(tf.errors.OpError):
        list(lazy)
    assert lazy_policy.status("preprocess")["failure_code"] == "quota_exceeded"
    assert inspect_cache(tmp_path / "limited-lazy")["failure_code"] == "quota_exceeded"

    path = str(tmp_path / "model")
    ds, _ = _load(
        _source([1, 2, 3]),
        pipeline,
        cache_model_inputs=True,
        model_input_cache_path=path,
        cache_policy=_policy(),
    )
    assert [int(v) for batch in ds for v in batch["x"]] == [1, 2, 3, 0]
    assert inspect_cache(path)["state"] == "complete"

    with pytest.raises(ValueError, match="prepared dataset"):
        raw, tools = _load(
            _source([1, 2, 3]),
            pipeline,
            cache_model_inputs=True,
            model_input_cache_path=str(tmp_path / "unused"),
            cache_policy=_policy(mode="lazy"),
            return_raw_ds=True,
        )
        tools["finalize_fn"](raw.take(1))


@pytest.mark.parametrize("prefetch", [0, -1, 1.5, "2"])
def test_prefetch_rejects_invalid_limits(prefetch):
    with pytest.raises(ValueError, match="prefetch"):
        finalize_dataset(
            _source([1]),
            postprocess_fn=lambda x, num_classes=None: x,
            num_classes=None,
            batch_size=1,
            prefetch=prefetch,
        )


def test_bounded_prefetch_disables_injected_prefetch():
    batches, _ = finalize_dataset(
        _source([1, 2, 3]),
        postprocess_fn=lambda x, num_classes=None: x,
        num_classes=None,
        batch_size=2,
        prefetch=2,
        map_parallel_calls=1,
    )
    options = batches.options()
    assert options.experimental_optimization.inject_prefetch is False
    assert options.autotune.enabled is False
    assert [int(batch["padding_mask"].numpy().sum()) for batch in batches] == [2, 1]


def test_cpu_setup_after_import_in_fresh_process():
    code = """
import json
from tensorflow.python.eager import context
import justdata.core
import justdata.vision
import justdata.acoustic
import justdata.audio
assert context.context()._context_handle is None
from justdata.core import configure_tensorflow_cpu
report = configure_tensorflow_cpu(intra_op_threads=1, inter_op_threads=1)
print(json.dumps(report))
"""
    process = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )
    report = json.loads(process.stdout.strip().splitlines()[-1])
    assert report["logical_gpus"] == []
    assert report["logical_cpus"]
    assert report["intra_op_threads"] == 1


def test_cpu_setup_rejects_late_thread_configuration():
    code = """
import tensorflow as tf
from justdata.core import configure_tensorflow_cpu
tf.constant(1).numpy()
try:
    configure_tensorflow_cpu(intra_op_threads=2)
except RuntimeError:
    pass
else:
    raise AssertionError('late configuration was accepted')
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)

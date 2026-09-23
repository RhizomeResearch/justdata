"""Measure finite CPU input profiles with native process RSS.

uv run python benchmarks/benchmark_input_resources.py
uv run python benchmarks/benchmark_input_resources.py --gpu-preflight
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from justdata.core import (
    CacheError,
    CachePolicy,
    configure_tensorflow_cpu,
    get_pipeline,
    load_ds,
    register_pipeline,
    register_source_loader,
)


SHAPES = {
    "training": (512, 512),
    "square": (1024, 1024),
    "landscape": (576, 1024),
    "portrait": (1024, 576),
}
MAX_ADDITIONAL_RSS = 1 << 30
CACHE_BUDGET = 1 << 30


def _rss_bytes():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak * 1024 if sys.platform != "darwin" else peak


@register_source_loader("resource-fixture:")
def _source(name, splits, data_dir=None):
    del splits, data_dir
    _, profile, count = name.split(":")
    height, width = SHAPES[profile]

    def samples():
        for index in range(int(count)):
            yield {
                "image": np.full((height, width, 3), index % 255, np.uint8),
                "mask": np.full((height, width), index % 3, np.int32),
                "pixel_valid_mask": np.ones((height, width), np.bool_),
            }

    signature = {
        "image": tf.TensorSpec((height, width, 3), tf.uint8),
        "mask": tf.TensorSpec((height, width), tf.int32),
        "pixel_valid_mask": tf.TensorSpec((height, width), tf.bool),
    }
    return [tf.data.Dataset.from_generator(samples, output_signature=signature)]


def _resolve(config, is_training):
    return {
        "implementation": "benchmarks/input-resources-v1",
        "configuration": {"profile": config["profile"]},
        "stages": {
            "preprocess": {"active": True, "config": {}},
            "augment": {"active": is_training, "config": {}},
            "late_augment": {"active": False, "config": {}},
            "postprocess": {"active": True, "config": {}},
        },
        "model_input": {"image": "float32 NHWC", "mask": "int32 HW"},
    }


@register_pipeline("benchmarks/input-resources", config_resolver=_resolve)
def _pipeline(profile, **kwargs):
    del profile, kwargs

    def preprocess(sample):
        return sample | {"image": tf.cast(sample["image"], tf.float32) / 255.0}

    def augment(sample, seed=None):
        return sample

    def postprocess(sample, num_classes=None):
        return sample

    return preprocess, augment, None, postprocess


def _run_case(profile, records, prefetch, *, cache=False):
    report = configure_tensorflow_cpu(intra_op_threads=1, inter_op_threads=1)
    baseline = _rss_bytes()
    name = f"resource-fixture:{profile}:{records}"
    pipeline = get_pipeline(pipeline_name="benchmarks/input-resources", profile=profile)
    policy = None
    with tempfile.TemporaryDirectory() as directory:
        options = {}
        if cache:
            identity = hashlib.sha256(name.encode()).hexdigest()
            policy = CachePolicy(
                input_identity=identity,
                max_bytes=CACHE_BUDGET,
                max_examples=records,
                materialization="eager",
                callbacks_are_deterministic=True,
            )
            options = {
                "cache_dataset": True,
                "cache_path": str(Path(directory) / "prepared"),
                "cache_policy": policy,
            }
        start = time.perf_counter()
        ds, _, config = load_ds(
            name,
            "validation",
            "validation",
            2,
            0,
            pipeline=pipeline,
            return_config=True,
            deterministic=True,
            map_parallel_calls=2,
            private_threadpool_size=2,
            max_intra_op_parallelism=1,
            prefetch=prefetch,
            shuffle_buffer=8,
            **options,
        )
        real_rows = 0
        for batch in ds:
            real_rows += int(tf.reduce_sum(tf.cast(batch["padding_mask"], tf.int32)))
        elapsed = time.perf_counter() - start
        status = policy.status("preprocess") if policy is not None else None
    additional_rss = max(0, _rss_bytes() - baseline)
    result = {
        "profile": profile,
        "shape": SHAPES[profile],
        "records": records,
        "real_rows": real_rows,
        "prefetch": prefetch,
        "seconds": round(elapsed, 3),
        "rows_per_second": round(real_rows / elapsed, 2),
        "additional_peak_rss_bytes": additional_rss,
        "additional_rss_limit_bytes": MAX_ADDITIONAL_RSS,
        "cache": status,
        "devices": report,
        "limits": config.to_dict()["execution"]["limits"],
    }
    if real_rows != records or additional_rss > MAX_ADDITIONAL_RSS:
        raise RuntimeError(json.dumps(result))
    print(json.dumps(result), flush=True)


def _run_quota_probe():
    report = configure_tensorflow_cpu(intra_op_threads=1, inter_op_threads=1)
    with tempfile.TemporaryDirectory() as directory:
        policy = CachePolicy(
            input_identity="synthetic-quota-v1",
            max_bytes=1,
            max_examples=32,
            materialization="eager",
            callbacks_are_deterministic=True,
        )
        path = str(Path(directory) / "limited")
        try:
            load_ds(
                "resource-fixture:training:32",
                "validation",
                "validation",
                2,
                0,
                pipeline=get_pipeline(
                    pipeline_name="benchmarks/input-resources", profile="training"
                ),
                cache_dataset=True,
                cache_path=path,
                cache_policy=policy,
                map_parallel_calls=2,
                private_threadpool_size=2,
                max_intra_op_parallelism=1,
                prefetch=1,
            )
        except CacheError as error:
            if error.code != "quota_exceeded":
                raise
            print(
                json.dumps(
                    {
                        "profile": "quota_probe",
                        "failure_code": error.code,
                        "cache": policy.status("preprocess"),
                        "devices": report,
                    }
                ),
                flush=True,
            )
            return
        raise RuntimeError("Quota probe unexpectedly completed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case", nargs=4, metavar=("PROFILE", "RECORDS", "PREFETCH", "CACHE")
    )
    parser.add_argument("--gpu-preflight", action="store_true")
    parser.add_argument("--quota-probe", action="store_true")
    args = parser.parse_args()
    if args.gpu_preflight:
        report = configure_tensorflow_cpu(intra_op_threads=1, inter_op_threads=1)
        import jax
        import jax.numpy as jnp

        devices = jax.devices("gpu")
        with jax.default_device(devices[0]):
            result = (jnp.ones((2, 2)) + 1).block_until_ready()
        if result.device != devices[0]:
            raise RuntimeError("JAX computation was not placed on the selected GPU")
        print(
            json.dumps(
                {"tensorflow": report, "jax_gpu_devices": [str(d) for d in devices]}
            )
        )
        return
    if args.case:
        profile, records, prefetch, cache = args.case
        _run_case(profile, int(records), int(prefetch), cache=cache == "cache")
        return
    if args.quota_probe:
        _run_quota_probe()
        return
    for profile in SHAPES:
        for records in (32, 256):
            for prefetch in (1, 2):
                command = [
                    sys.executable,
                    __file__,
                    "--case",
                    profile,
                    str(records),
                    str(prefetch),
                    "stream",
                ]
                subprocess.run(command, check=True)
    subprocess.run(
        [sys.executable, __file__, "--case", "training", "32", "1", "cache"],
        check=True,
    )
    subprocess.run([sys.executable, __file__, "--quota-probe"], check=True)


if __name__ == "__main__":
    main()

"""CPU benchmarks without downloads; run from the repository root.

    uv run python benchmarks/benchmark_performance.py --repeats 5

Run the same script with PYTHONPATH=/path/to/baseline/src to compare a saved
source tree. Compare output hashes as well as timings. Construction and epoch
iteration are reported separately; warmup and hashing are outside timed runs.
Temporary allocation peaks use tracemalloc and exclude TensorFlow native memory.
"""

import argparse
import hashlib
import json
import platform
import statistics
import tempfile
import time
import tracemalloc
from pathlib import Path

import numpy as np
import tensorflow as tf


def _synchronize(result):
    return tf.nest.map_structure(
        lambda value: value.numpy() if tf.is_tensor(value) else value, result
    )


def _digest(result, fixture_root=None):
    digest = hashlib.sha256()
    for value in tf.nest.flatten(result):
        array = np.asarray(value)
        digest.update(str((array.shape, array.dtype.str)).encode())
        contents = (
            repr(array.tolist()).encode()
            if array.dtype.kind in "OUS"
            else array.tobytes()
        )
        if fixture_root is not None and array.dtype.kind in "OUS":
            contents = contents.replace(str(fixture_root).encode(), b"<fixture>")
        digest.update(contents)
    return digest.hexdigest()


def _measure(name, fn, repeats, *, calls=1, allocations=False, fingerprint=True):
    result = _synchronize(fn())
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(calls):
            _synchronize(fn())
        samples.append((time.perf_counter() - start) * 1000 / calls)
    report = {
        "case": name,
        "median_ms": statistics.median(samples),
        "samples_ms": samples,
    }
    if fingerprint:
        report["output_sha256"] = _digest(result)
    if allocations:
        tracemalloc.start()
        _synchronize(fn())
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        report["python_numpy_peak_mib"] = peak / 2**20
    print(json.dumps(report), flush=True)


def _options():
    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    options.threading.max_intra_op_parallelism = 1
    return options


def _consume(dataset):
    # Hash every batch outside the timer in _dataset_case; keep timed iteration
    # streaming and bounded to one batch of NumPy outputs.
    for _ in dataset.as_numpy_iterator():
        pass


def _dataset_case(name, build, repeats, *, fixture_root, batched=False):
    _measure(name + "/build", build, repeats, fingerprint=False)
    dataset = build().with_options(_options())
    if not batched:
        dataset = dataset.batch(128)
    _measure(name + "/epoch", lambda: _consume(dataset), repeats, fingerprint=False)
    digest = hashlib.sha256()
    for batch in dataset.as_numpy_iterator():
        digest.update(_digest(batch, fixture_root).encode())
    print(json.dumps({"case": name, "output_sha256": digest.hexdigest()}), flush=True)


def _records(count, path):
    return [
        {
            "path": str(path),
            "label": index % 10,
            "split": "eval",
            "clip_id": str(index),
            "source_id": "a",
            "start_time": 0.0,
            "end_time": 1.0,
            "dataset": "synthetic",
            "example_id": str(index),
            "filename": path.name,
        }
        for index in range(count)
    ]


def _source_benchmarks(root, repeats):
    from justdata.acoustic import dcase2025
    from justdata.acoustic import sources as audio_sources
    from justdata.vision import sources as vision_sources

    records = _records(10000, root / "audio.wav")
    _dataset_case(
        "local_records",
        lambda: audio_sources._records_to_dataset(records),
        repeats,
        fixture_root=root,
    )
    dcase_records = [
        record
        | {
            "scene_label": "airport",
            "scene_id": 0,
            "device": "A",
            "device_type": "real",
            "is_known_device": True,
            "city": "london",
            "location_id": "0001",
            "segment_id": record["clip_id"],
            "source_recording_id": "0001",
            "fold": "eval",
        }
        for record in records
    ]
    _dataset_case(
        "dcase_records",
        lambda: dcase2025._records_to_dataset(dcase_records),
        repeats,
        fixture_root=root,
    )

    image_path = root / "image.png"
    image = tf.reshape(tf.cast(tf.range(256 * 320 * 3), tf.uint8), [256, 320, 3])
    image_path.write_bytes(tf.io.encode_png(image).numpy())
    vision_records = [
        {
            "_path": str(image_path),
            "label": index % 10,
            "metadata": {
                "dataset": "synthetic",
                "record_id": "123",
                "archive": "images.zip",
                "split": "eval",
                "class_name": "class",
                "class_index": index % 10,
                "filename": image_path.name,
                "path": str(image_path),
                "example_id": str(index),
            },
        }
        for index in range(257)
    ]
    _dataset_case(
        "vision_decode",
        lambda: vision_sources._records_to_vision_dataset(vision_records),
        repeats,
        fixture_root=root,
    )
    return records[:257], vision_records


def _pipeline_benchmarks(root, audio_records, vision_records, repeats):
    from justdata.acoustic.adapters import acoustic_source_adapter
    from justdata.acoustic.sources import _records_to_dataset
    from justdata.core.loader import load_ds
    from justdata.core.registry import get_pipeline
    from justdata.core.sources import register_source_loader
    from justdata.vision.sources import _records_to_vision_dataset

    waveform = tf.sin(tf.range(32000, dtype=tf.float32) * 0.03)[:, None]
    (root / "audio.wav").write_bytes(tf.audio.encode_wav(waveform, 32000).numpy())

    @register_source_loader("benchmark:")
    def source(name, splits, data_dir):
        del splits, data_dir
        if name == "benchmark:audio":
            dataset = _records_to_dataset(audio_records)
            return [dataset.map(acoustic_source_adapter, num_parallel_calls=2)]
        return [_records_to_vision_dataset(vision_records)]

    for name, pipeline in (
        (
            "audio",
            get_pipeline(
                pipeline_name="acoustic/classification",
                preset="audio_default_32k_logmel64",
                modality="acoustic",
            ),
        ),
        (
            "vision",
            get_pipeline(
                pipeline_name="vision/classification",
                apply_presets=False,
                aug_kwargs={"image_size": 224, "enable": False},
                postproc_kwargs={
                    "image_size": 224,
                    "eval_view_config": {"image_size": 224},
                    "normalization_params": (
                        (0.485, 0.456, 0.406),
                        (0.229, 0.224, 0.225),
                    ),
                },
            ),
        ),
    ):

        def build(name=name, pipeline=pipeline):
            return load_ds(
                f"benchmark:{name}",
                "eval",
                "validation",
                batch_size=32,
                seed=12,
                num_classes=10,
                pipeline=pipeline,
                deterministic=True,
                map_parallel_calls=2,
                private_threadpool_size=2,
                max_intra_op_parallelism=1,
            )[0]

        _dataset_case(
            name + "_pipeline", build, repeats, fixture_root=root, batched=True
        )


def _operation_benchmarks(repeats):
    from justdata.acoustic._signal import _convolve_channels
    from justdata.acoustic.sources import _as_waveform_np
    from justdata.core.stats import _partial_state
    from justdata.vision.stages import apply_eval_views

    image = tf.reshape(tf.cast(tf.range(256 * 320 * 3), tf.uint8), [256, 320, 3])
    for count in (1, 5):
        view = tf.function(
            lambda image: apply_eval_views(
                {"image": image},
                config={"image_size": 224, "mode": "multi_crop", "num_crops": count},
            ),
            autograph=False,
        )
        _measure(f"eval_views_{count}", lambda: view(image), repeats, calls=200)

    rng = np.random.default_rng(12)
    ir = tf.constant([0.02, 0.08, 0.8, 0.08, 0.02])
    for channels in (1, 2):
        waveform = tf.constant(rng.normal(size=(4096, channels)), tf.float32)
        convolution = tf.function(
            lambda audio: _convolve_channels(audio, ir, False), autograph=False
        )
        _measure(
            f"convolution_{channels}_channels",
            lambda: convolution(waveform),
            repeats,
            calls=200,
        )

    array = rng.normal(size=(320000, 2)).astype(np.float32)
    _measure(
        "hf_waveform",
        lambda: _as_waveform_np(
            {"array": array, "sampling_rate": 32000},
            decode_mode="hf_native",
            fallback_sample_rate=None,
        ),
        repeats,
        calls=100,
        allocations=True,
    )
    observations = rng.normal(size=(32000, 128))
    _measure(
        "statistics", lambda: _partial_state(observations), repeats, allocations=True
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    import justdata.acoustic  # noqa: F401
    import justdata.vision  # noqa: F401

    print(
        json.dumps(
            {
                "tensorflow": tf.__version__,
                "numpy": np.__version__,
                "python": platform.python_version(),
                "device": "CPU",
                "threads": 2,
                "repeats": args.repeats,
            }
        ),
        flush=True,
    )
    _operation_benchmarks(args.repeats)
    with tempfile.TemporaryDirectory(prefix="justdata-benchmark-") as directory:
        root = Path(directory)
        audio_records, vision_records = _source_benchmarks(root, args.repeats)
        _pipeline_benchmarks(root, audio_records, vision_records, args.repeats)


if __name__ == "__main__":
    main()

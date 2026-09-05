"""Synthetic CPU training benchmarks; no downloads or persistent fixtures.

    uv run python benchmarks/benchmark_training.py --repeats 5

Use PYTHONPATH=/path/to/baseline/src to compare a saved source tree. Older
versions without the stateless-augmentation opt-in run that case serially.
"""

import argparse
import hashlib
import inspect
import json
import platform

import numpy as np
import tensorflow as tf

from benchmark_performance import _consume, _digest, _measure


def _epoch_case(name, dataset, repeats):
    _measure(name, lambda: _consume(dataset), repeats, fingerprint=False)
    digest = hashlib.sha256()
    for batch in dataset.as_numpy_iterator():
        digest.update(_digest(batch).encode())
    print(json.dumps({"case": name, "output_sha256": digest.hexdigest()}), flush=True)


def _audio_cases(repeats):
    from justdata.acoustic.augment.patchout import _drop_axis
    from justdata.acoustic.tasks import make_late_augmentations

    stage = tf.function(
        make_late_augmentations(
            spectrogram_augmentations={
                "time_mask": {"max_width": 20, "fill_value": "zero"},
                "frequency_mask": {"max_width": 12, "fill_value": "zero"},
            },
            model_layout="btfc",
        )
    )
    features = tf.reshape(tf.range(32 * 101 * 64, dtype=tf.float32), [32, 101, 64, 1])
    for length in (32000, 320000):
        batch = {
            "features": features,
            "waveform": tf.ones([32, length, 1]),
            "label": tf.range(32),
            "metadata": {"id": tf.strings.as_string(tf.range(32))},
        }
        _measure(
            f"spectrogram_with_waveform_{length}",
            lambda: stage(batch, seed=tf.constant([19, 31])),
            repeats,
            calls=10,
        )

    for size, count in ((128, 4), (12800, 1000)):
        values = tf.reshape(tf.range(size * 3, dtype=tf.float32), [size, 3])
        drop = tf.function(
            lambda x, seed: _drop_axis(x, axis=0, count=count, seed=seed),
            autograph=False,
        )
        _measure(
            f"patchout_{size}_{count}",
            lambda: drop(values, tf.constant([31, 9])),
            repeats,
            calls=50,
        )


def _vision_cases(repeats):
    from justdata.vision.augmentations.color import solarize
    from justdata.vision.augmentations.mixing import random_erasing
    from justdata.vision.tasks.classification import make_late_augmentations

    images = tf.random.stateless_uniform(
        [32, 224, 224, 3], [17, 9], minval=-2.0, maxval=3.0
    )
    labels = tf.one_hot(tf.range(32), 1000)
    seed = tf.constant([9, 17], tf.int64)
    for name, mixing in (
        ("mixup", {"cutmix_alpha": 0.0}),
        ("cutmix", {"mixup_alpha": 0.0}),
    ):
        stage = tf.function(make_late_augmentations(**mixing))
        for layout in ("nchw", "nhwc"):
            value = tf.transpose(images, [0, 3, 1, 2]) if layout == "nchw" else images
            batch = {"image": value, "label": labels}
            _measure(
                f"{name}_{layout}",
                lambda: stage(batch, num_classes=1000, seed=seed),
                repeats,
                calls=10,
            )
    for probability in (0.25, 1.0):
        _measure(
            f"erasing_{probability}",
            lambda: random_erasing(images, seed, p=probability),
            repeats,
            calls=10,
        )
    solarize_seeds = tf.random.split(seed, 32)
    solarize_images = (images + 2.0) * 51.0
    for probability in (0.0, 0.2, 1.0):
        apply = tf.function(
            lambda x: tf.map_fn(
                lambda pair: solarize(pair[0], pair[1], p=probability),
                (x, solarize_seeds),
                fn_output_signature=x.dtype,
            ),
            autograph=False,
        )
        _measure(
            f"solarize_{probability}", lambda: apply(solarize_images), repeats, calls=10
        )


def _loader_cases(repeats):
    from justdata.core.finalization import finalize_dataset
    from justdata.core.loader import load_ds
    from justdata.core.sources import register_source_loader
    from justdata.vision.tasks.classification import make_augmentations

    options = tf.data.Options()
    options.threading.private_threadpool_size = 4
    options.threading.max_intra_op_parallelism = 1
    raw = tf.data.Dataset.from_tensor_slices(
        {
            "features": tf.reshape(
                tf.cast(tf.range(10000 * 64), tf.float32), [10000, 64]
            ),
            "label": tf.range(10000) % 10,
            "metadata": {
                "id": tf.range(10000),
                "path": tf.fill([10000], "synthetic/path"),
            },
        }
    ).with_options(options)

    def postprocess(sample, num_classes=None):
        return sample | {"features": (sample["features"] - 123.0) / 17.0}

    for mode in ("full", "numeric_only", "none"):
        ds, _ = finalize_dataset(
            raw,
            postprocess_fn=postprocess,
            num_classes=10,
            batch_size=128,
            metadata_mode=mode,
            is_training=True,
            deterministic=True,
        )
        _epoch_case(f"metadata_{mode}", ds, repeats)

    image = tf.random.stateless_uniform([256, 320, 3], [4, 9], maxval=255.0)

    @register_source_loader("training-benchmark:")
    def source(name, splits, data_dir):
        return [
            tf.data.Dataset.range(512).map(
                lambda index: {"image": image, "label": index}
            )
        ]

    def identity(sample, **kwargs):
        return sample

    augment = make_augmentations(
        224,
        augment_type="color_jitter",
        cj_kwargs={
            "brightness": 0.4,
            "contrast": 0.4,
            "saturation": 0.4,
            "hue": 0.1,
            "p": 0.8,
            "p_grayscale": 0.2,
        },
    )
    raw, tools = load_ds(
        "training-benchmark:vision",
        "train",
        "train",
        32,
        seed=0,
        preprocess_fn=identity,
        augment_fn=augment,
        postprocess_fn=identity,
        late_augment_fn=identity,
        shuffle_buffer=512,
        deterministic=True,
        return_raw_ds=True,
        map_parallel_calls=4,
        private_threadpool_size=4,
        max_intra_op_parallelism=1,
    )
    finalize = tools["finalize_epoch"]
    available = "augment_is_stateless" in inspect.signature(finalize).parameters
    print(json.dumps({"parallel_stateless_available": available}), flush=True)
    for parallel in (False, True):
        kwargs = {"augment_is_stateless": parallel} if available else {}
        ds, _ = finalize(raw, seed=91, **kwargs)
        _epoch_case(
            f"training_epoch_{'parallel' if parallel else 'serial'}", ds, repeats
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
    print(
        json.dumps(
            {
                "tensorflow": tf.__version__,
                "numpy": np.__version__,
                "python": platform.python_version(),
                "device": "CPU",
                "threads": 2,
                "dataset_threads": 4,
                "repeats": args.repeats,
            }
        ),
        flush=True,
    )
    _audio_cases(args.repeats)
    _vision_cases(args.repeats)
    _loader_cases(args.repeats)


if __name__ == "__main__":
    main()

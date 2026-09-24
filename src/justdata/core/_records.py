"""Tensor records shared by protected caches and inventory snapshots.

Each sample is stored as one ``tf.train.Example`` whose features are the
serialized leaves of ``tf.nest.flatten(sample)``, keyed by leaf index. The byte
layout feeds persisted digests, so it must remain stable.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import tensorflow as tf


RECORDS_FILE = "records.tfrecord"


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def serialize_example(sample: Any) -> bytes:
    features = {
        str(index): tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(value).numpy()])
        )
        for index, value in enumerate(tf.nest.flatten(sample))
    }
    return tf.train.Example(
        features=tf.train.Features(feature=features)
    ).SerializeToString()


def parse_example_fn(signature: Any):
    """Return a graph function decoding one serialized example into ``signature``."""
    specs = tf.nest.flatten(signature)
    fields = {
        str(index): tf.io.FixedLenFeature([], tf.string) for index in range(len(specs))
    }

    def parse(serialized):
        values = tf.io.parse_single_example(serialized, fields)
        leaves = [
            tf.ensure_shape(
                tf.io.parse_tensor(values[str(index)], spec.dtype), spec.shape
            )
            for index, spec in enumerate(specs)
        ]
        return tf.nest.pack_sequence_as(signature, leaves)

    return parse


def read_examples(
    root: Path, signature: Any, count: int | None = None
) -> tf.data.Dataset:
    """Read ``root/records.tfrecord`` in order, asserting ``count`` when known."""
    ds = tf.data.TFRecordDataset(os.fspath(root / RECORDS_FILE))
    ds = ds.map(parse_example_fn(signature), num_parallel_calls=1, deterministic=True)
    if count is not None:
        ds = ds.apply(tf.data.experimental.assert_cardinality(count))
    return ds

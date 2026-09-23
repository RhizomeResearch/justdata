"""Process setup for TensorFlow input pipelines beside other accelerators."""

from __future__ import annotations

import tensorflow as tf


def configure_tensorflow_cpu(
    *, intra_op_threads: int | None = None, inter_op_threads: int | None = None
) -> dict[str, object]:
    """Hide TensorFlow GPUs before runtime initialization and report placement.

    Call before constructing tensors, datasets, or invoking other TensorFlow
    operations. A runtime that was initialized too early raises RuntimeError.
    """
    for name, value in (
        ("intra_op_threads", intra_op_threads),
        ("inter_op_threads", inter_op_threads),
    ):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ):
            raise ValueError(f"{name} must be a positive integer")
    physical_gpus = tf.config.list_physical_devices("GPU")
    tf.config.set_visible_devices([], "GPU")
    if intra_op_threads is not None:
        tf.config.threading.set_intra_op_parallelism_threads(intra_op_threads)
    if inter_op_threads is not None:
        tf.config.threading.set_inter_op_parallelism_threads(inter_op_threads)
    logical_gpus = tf.config.list_logical_devices("GPU")
    if logical_gpus:
        raise RuntimeError("TensorFlow GPU visibility could not be disabled")
    return {
        "tensorflow_version": tf.__version__,
        "physical_gpus": [device.name for device in physical_gpus],
        "logical_gpus": [],
        "logical_cpus": [
            device.name for device in tf.config.list_logical_devices("CPU")
        ],
        "intra_op_threads": tf.config.threading.get_intra_op_parallelism_threads(),
        "inter_op_threads": tf.config.threading.get_inter_op_parallelism_threads(),
    }


__all__ = ["configure_tensorflow_cpu"]

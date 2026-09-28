"""Require JAX GPU computations beside a CPU-only TensorFlow input pipeline.

uv run python benchmarks/gpu_preflight.py

Importing JustData imports TensorFlow, whose CUDA-enabled Linux wheel exposes
CUDA library stubs to the whole process. JAX therefore initializes its GPU
backend before JustData is imported, as the README prescribes.
"""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np


def _computations():
    rng = np.random.default_rng(0)
    left = rng.standard_normal((256, 256), dtype=np.float32)
    right = rng.standard_normal((256, 256), dtype=np.float32)
    images = rng.standard_normal((4, 32, 32, 8), dtype=np.float32)
    kernel = rng.standard_normal((3, 3, 8, 16), dtype=np.float32)
    return {
        "matmul": lambda: jnp.matmul(left, right),
        "conv": lambda: jax.lax.conv_general_dilated(
            images,
            kernel,
            (1, 1),
            "SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        ),
        "fft2": lambda: jnp.abs(jnp.fft.fft2(left)),
    }


def main():
    gpu = jax.devices("gpu")[0]
    from justdata.core import configure_tensorflow_cpu

    report = configure_tensorflow_cpu(intra_op_threads=1, inter_op_threads=1)
    cpu = jax.devices("cpu")[0]
    errors = {}
    with jax.default_matmul_precision("highest"):
        for name, compute in _computations().items():
            with jax.default_device(gpu):
                result = compute().block_until_ready()
            if result.devices() != {gpu}:
                raise RuntimeError(f"JAX {name} was not placed on the selected GPU")
            with jax.default_device(cpu):
                expected = np.asarray(compute())
            error = float(np.max(np.abs(np.asarray(result) - expected)))
            if error > 1e-4 * max(1.0, float(np.max(np.abs(expected)))):
                raise RuntimeError(f"JAX {name} on GPU differs from CPU by {error}")
            errors[name] = error
    print(
        json.dumps(
            {
                "tensorflow": report,
                "jax_gpu_device": str(gpu),
                "max_abs_error": errors,
            }
        )
    )


if __name__ == "__main__":
    main()

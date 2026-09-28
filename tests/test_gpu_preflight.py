"""Import-order contract of the JAX GPU preflight benchmark."""

import os
from pathlib import Path
import subprocess
import sys


PREFLIGHT = Path(__file__).resolve().parents[1] / "benchmarks" / "gpu_preflight.py"


def test_gpu_preflight_initializes_jax_before_tensorflow(tmp_path):
    # The CUDA-enabled TensorFlow wheel exposes CUDA stubs process-wide on
    # import, so JAX must initialize its GPU backend first. This stand-in stops
    # the preflight at backend initialization and reports the import state.
    fake_jax = tmp_path / "jax"
    fake_jax.mkdir()
    (fake_jax / "__init__.py").write_text(
        "import sys\n"
        "def devices(backend=None):\n"
        "    print(f'tensorflow_imported={\"tensorflow\" in sys.modules}')\n"
        "    raise SystemExit(0)\n"
    )
    (fake_jax / "numpy.py").write_text("")
    python_path = [str(tmp_path), os.environ.get("PYTHONPATH", "")]
    result = subprocess.run(
        [sys.executable, str(PREFLIGHT)],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, python_path))},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["tensorflow_imported=False"]

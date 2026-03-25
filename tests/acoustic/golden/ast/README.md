# AST Golden Fixtures

Run `generate_ast_golden.py` from the project environment after installing the
`golden` extra. The reference path uses the Python 3.12-compatible
Torch/Torchaudio versions locked by this repo.

The generated `.npz` files are intentionally small and contain:

- `waveform`: deterministic synthetic mono waveform with shape `[time, 1]`
- `features`: AST eval fbank tensor after target padding/cropping and
  `(x - mean) / (std * 2)` normalization, with shape `[time, 128]`

These fixtures are optional and used only by `pytest -m golden`.

# Golden Tests

Golden tests are optional compatibility tests against external reference
packages, feature extractors, fixtures, or converted checkpoints. They exist to
catch frontend drift that ordinary shape and smoke tests cannot see.

## Default CI

Default CI runs non-golden tests:

```bash
uv run pytest -m "not golden"
```

Golden tests are marked with `@pytest.mark.golden` or module-level
`pytestmark = pytest.mark.golden` and are excluded by default.

## Optional golden job

Run golden tests only when reference assets are available:

```bash
uv sync --extra golden
uv run pytest -m golden
```

The `golden` extra is intentionally separate from the default dependency set so
normal development and CI do not download heavyweight reference stacks.

## What belongs in a golden test

Use golden tests for:

- Same waveform -> same frontend tensor against EfficientAT, PaSST, CED, or PANNs references.
- Same frontend tensor -> same checkpoint logits after conversion.
- Reference fixtures that prove windowing, mel filters, log compression, layout, and normalization.

Do not use golden tests for basic shape, dtype, metadata, registry, or padding
behavior. Those belong in ordinary non-golden tests.

## Compatibility contract

When a golden fixture changes, update the relevant preset contract and document
the new preset hash. A changed hash is expected when frontend semantics change;
an unchanged hash with changed golden output indicates a bug.

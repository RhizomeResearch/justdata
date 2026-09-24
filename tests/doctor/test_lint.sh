#!/bin/sh
set -eu

uv run --locked --only-group lint ruff check .
uv run --locked --only-group lint ruff format --check .
uv run --locked --only-group lint mdformat --check --wrap 120 \
    ./*.md benchmarks docs examples src tests
# Type checking resolves imports against the locked development environment.
uv run --locked ty check src/

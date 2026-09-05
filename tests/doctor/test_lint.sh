#!/bin/sh
set -eu

uv run --locked --only-group lint ruff check .
uv run --locked --only-group lint ruff format --check .

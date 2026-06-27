#!/bin/sh
set -eu

work_dir="$(mktemp -d "${TMPDIR:-/tmp}/justdata-package-support.XXXXXX")"
trap 'rm -rf "$work_dir"' EXIT

wheel_dir="$work_dir/wheel"
uv build --wheel --out-dir "$wheel_dir"
wheel="$(find "$wheel_dir" -maxdepth 1 -name '*.whl' -print -quit)"

python - "$wheel" <<'PY'
import email
import sys
import zipfile

wheel = sys.argv[1]
with zipfile.ZipFile(wheel) as archive:
    metadata_path = next(
        name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
    )
    metadata = email.message_from_bytes(archive.read(metadata_path))

actual = metadata["Requires-Python"]
expected = ">=3.11,<3.14"
if actual.replace(" ", "") != expected:
    raise SystemExit(f"Requires-Python is {actual!r}, expected {expected!r}")
PY

for extra in base vision acoustic; do
    venv="$work_dir/$extra"
    uv venv --python "$(command -v python)" "$venv"

    if [ "$extra" = base ]; then
        requirement="$wheel"
    else
        requirement="$wheel[$extra]"
    fi

    uv pip install --python "$venv/bin/python" "$requirement"

    case "$extra" in
        base)
            "$venv/bin/python" -c 'import justdata; import justdata.core'
            ;;
        vision)
            "$venv/bin/python" -c 'import justdata; import justdata.core; import justdata.vision'
            ;;
        acoustic)
            "$venv/bin/python" -c 'import justdata; import justdata.core; import justdata.acoustic'
            ;;
    esac
done

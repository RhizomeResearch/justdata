#!/bin/sh
set -eu

work_dir="$(mktemp -d "${TMPDIR:-/tmp}/justdata-package-support.XXXXXX")"
trap 'rm -rf "$work_dir"' EXIT

dist_dir="$work_dir/dist"
uv build --out-dir "$dist_dir"
wheel="$(find "$dist_dir" -maxdepth 1 -name '*.whl' -print -quit)"
sdist="$(find "$dist_dir" -maxdepth 1 -name '*.tar.gz' -print -quit)"

python - "$wheel" "$sdist" <<'PY'
import email
import sys
import tarfile
import zipfile

wheel = sys.argv[1]
sdist = sys.argv[2]
with zipfile.ZipFile(wheel) as archive:
    wheel_names = archive.namelist()
    metadata_path = next(
        name for name in wheel_names if name.endswith(".dist-info/METADATA")
    )
    metadata = email.message_from_bytes(archive.read(metadata_path))

actual = metadata["Requires-Python"]
expected = ">=3.11,<3.14"
if actual.replace(" ", "") != expected:
    raise SystemExit(f"Requires-Python is {actual!r}, expected {expected!r}")

if metadata["License-Expression"] != "MIT":
    raise SystemExit(
        f"License-Expression is {metadata['License-Expression']!r}, expected 'MIT'"
    )
if "LICENSE" not in metadata.get_all("License-File", []):
    raise SystemExit("Wheel metadata does not declare LICENSE")
if not any(name.endswith("/licenses/LICENSE") for name in wheel_names):
    raise SystemExit("Wheel does not contain LICENSE")

with tarfile.open(sdist) as archive:
    if not any(name.endswith("/LICENSE") for name in archive.getnames()):
        raise SystemExit("Source distribution does not contain LICENSE")
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

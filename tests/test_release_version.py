import os
from pathlib import Path
import subprocess
import sys

import pytest


CHECKER = Path(__file__).parent / "doctor" / "check_release_version.py"


def run_check(tmp_path, tag, *, project_version="1.1.0", package_version="1.1.0"):
    (tmp_path / "pyproject.toml").write_text(
        f'[project]\nversion = "{project_version}"\n'
    )
    package = tmp_path / "src" / "justdata"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(f'__version__ = "{package_version}"\n')
    environment = dict(os.environ)
    environment.pop("CI_COMMIT_TAG", None)
    if tag is not None:
        environment["CI_COMMIT_TAG"] = tag
    return subprocess.run(
        [sys.executable, str(CHECKER)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("tag", ["1.1.0", "v1.1.0"])
def test_release_accepts_matching_tag(tmp_path, tag):
    result = run_check(tmp_path, tag)
    assert result.returncode == 0, result.stderr
    assert "Release version verified: 1.1.0" in result.stdout


@pytest.mark.parametrize("tag", [None, ""])
def test_release_requires_tag(tmp_path, tag):
    result = run_check(tmp_path, tag)
    assert result.returncode != 0
    assert "CI_COMMIT_TAG is required" in result.stderr


@pytest.mark.parametrize("tag", ["v1.0.0", "vv1.1.0", "release/1.1.0"])
def test_release_rejects_mismatched_tag(tmp_path, tag):
    result = run_check(tmp_path, tag)
    assert result.returncode != 0
    assert "Release version mismatch" in result.stderr


def test_release_rejects_package_version_drift(tmp_path):
    result = run_check(tmp_path, "v1.1.0", package_version="1.0.0")
    assert result.returncode != 0
    assert "justdata.__version__='1.0.0'" in result.stderr


def test_release_rejects_project_version_drift(tmp_path):
    result = run_check(tmp_path, "v1.1.0", project_version="1.0.0")
    assert result.returncode != 0
    assert "pyproject.toml='1.0.0'" in result.stderr

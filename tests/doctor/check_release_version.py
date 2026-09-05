"""Reject release tags that disagree with package metadata before publishing."""

import os
from pathlib import Path
import runpy
import tomllib


def main():
    tag = os.environ.get("CI_COMMIT_TAG", "")
    if not tag:
        raise SystemExit("CI_COMMIT_TAG is required for publishing")

    with Path("pyproject.toml").open("rb") as stream:
        project_version = tomllib.load(stream)["project"]["version"]
    package_version = runpy.run_path("src/justdata/__init__.py")["__version__"]
    tag_version = tag.removeprefix("v")

    if tag_version != project_version or tag_version != package_version:
        raise SystemExit(
            f"Release version mismatch: tag={tag!r}, "
            f"pyproject.toml={project_version!r}, "
            f"justdata.__version__={package_version!r}"
        )
    print(f"Release version verified: {project_version} (tag {tag})")


if __name__ == "__main__":
    main()

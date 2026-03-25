import sys

from justdata.acoustic import *  # noqa: F403
from justdata import acoustic as _acoustic


for _name in ("configs", "schema", "registry", "presets", "pipelines", "tasks"):
    sys.modules[f"{__name__}.{_name}"] = getattr(_acoustic, _name)


__all__ = _acoustic.__all__

import sys

from justdata.acoustic import *  # noqa: F403
from justdata import acoustic as _acoustic


for _name in (
    "adapters",
    "configs",
    "datasets",
    "decoding",
    "pipelines",
    "frontends",
    "layouts",
    "normalization",
    "postprocessing",
    "presets",
    "registry",
    "resampling",
    "schema",
    "sources",
    "tasks",
):
    sys.modules[f"{__name__}.{_name}"] = getattr(_acoustic, _name)


__all__ = _acoustic.__all__

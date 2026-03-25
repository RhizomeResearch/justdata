import sys

from justdata.acoustic import *  # noqa: F403
from justdata import acoustic as _acoustic


for _name in (
    "adapters",
    "batching",
    "configs",
    "datasets",
    "decoding",
    "pipelines",
    "frontends",
    "jax",
    "labels",
    "layouts",
    "metadata",
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

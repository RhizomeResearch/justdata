import sys
from types import ModuleType

from justdata import acoustic as _acoustic
from justdata.acoustic import *  # noqa: F403


for _name in _acoustic.__all__:
    _value = getattr(_acoustic, _name)
    if isinstance(_value, ModuleType):
        sys.modules.setdefault(f"{__name__}.{_name}", _value)

_acoustic_prefix = f"{_acoustic.__name__}."
for _canonical_name, _module in tuple(sys.modules.items()):
    if _canonical_name.startswith(_acoustic_prefix):
        _alias_name = f"{__name__}.{_canonical_name.removeprefix(_acoustic_prefix)}"
        sys.modules.setdefault(_alias_name, _module)


__all__ = _acoustic.__all__

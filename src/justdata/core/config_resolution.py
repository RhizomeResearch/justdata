from __future__ import annotations

import copy
import inspect
from collections.abc import Collection, Mapping
from typing import Any, Callable


def is_positive_int(value: Any) -> bool:
    """Return whether ``value`` is a positive ``int``, excluding ``bool``."""
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def reject_unknown_keys(
    config: Mapping[str, Any], allowed: set[str], *, path: str
) -> None:
    unknown = sorted(set(config) - allowed)
    if unknown:
        name = unknown[0]
        raise ValueError(f"Unknown configuration key: {path}.{name}")


def resolve_callable_config(
    fn: Callable,
    supplied: Mapping[str, Any] | None,
    *,
    path: str,
    omit: Collection[str] | None = None,
) -> dict[str, Any]:
    """Validate keyword configuration and expand a callable's declared defaults."""
    if supplied is None:
        supplied = {}
    if not isinstance(supplied, Mapping):
        raise TypeError(f"{path} must be a mapping or None")

    omitted = omit or set()
    signature = inspect.signature(fn)
    parameters = {
        name: parameter
        for name, parameter in signature.parameters.items()
        if name not in omitted
        and parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    reject_unknown_keys(supplied, set(parameters), path=path)

    resolved = {}
    for name, parameter in parameters.items():
        if name in supplied:
            resolved[name] = copy.deepcopy(supplied[name])
        elif parameter.default is not inspect.Parameter.empty:
            resolved[name] = copy.deepcopy(parameter.default)
        else:
            raise ValueError(f"Missing required configuration key: {path}.{name}")
    return resolved


__all__ = ["is_positive_int", "reject_unknown_keys", "resolve_callable_config"]

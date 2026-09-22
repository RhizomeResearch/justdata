from __future__ import annotations

import copy
import dataclasses
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


def _json_value(value: Any, *, path: str = "config") -> Any:
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)

    if isinstance(value, Mapping):
        result = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string key: {key!r}")
            result[key] = _json_value(child, path=f"{path}.{key}")
        return result

    if isinstance(value, (tuple, list)):
        return [
            _json_value(child, path=f"{path}[{index}]")
            for index, child in enumerate(value)
        ]

    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return value

    raise TypeError(
        f"{path} contains unsupported value {value!r} of type {type(value).__name__}"
    )


def canonical_config_json(config: Mapping[str, Any]) -> str:
    """Encode executed configuration using the versioned canonical JSON form."""
    plain = _json_value(config)
    return json.dumps(
        plain,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


@dataclass(frozen=True)
class ExecutedConfig:
    """Immutable canonical snapshot of one configured JustData execution."""

    _canonical_json: str

    @classmethod
    def from_dict(cls, config: Mapping[str, Any]) -> ExecutedConfig:
        if not isinstance(config, Mapping):
            raise TypeError("ExecutedConfig requires a mapping")
        return cls(canonical_config_json(copy.deepcopy(config)))

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self._canonical_json)

    def to_json(self) -> str:
        return self._canonical_json

    def to_bytes(self) -> bytes:
        return self._canonical_json.encode("utf-8")


__all__ = ["ExecutedConfig", "canonical_config_json"]

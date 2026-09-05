from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from typing import Any


def _normalize_augment_specs(
    augmentations: Mapping[str, Any] | Sequence[Any] | str | None,
    *,
    canonical_name: Callable[[str], str] | None = None,
    mapping_name: Callable[[Any], str] | None = None,
) -> list[dict[str, Any]]:
    """Parse specs while keeping explicit-name and mapping-key aliases separate."""
    if augmentations is None:
        return []
    if isinstance(augmentations, str):
        name = canonical_name(augmentations) if canonical_name else augmentations
        return [{"name": name}]
    if isinstance(augmentations, Mapping):
        if "name" in augmentations:
            spec = dict(augmentations)
            name = str(spec["name"])
            spec["name"] = canonical_name(name) if canonical_name else name
            return [spec]

        specs = []
        for name, value in augmentations.items():
            if value is None or value is False:
                continue
            if mapping_name is not None:
                name = mapping_name(name)
            else:
                name = canonical_name(str(name)) if canonical_name else str(name)
            if value is True:
                specs.append({"name": name})
            elif isinstance(value, Mapping):
                specs.append({"name": name, **dict(value)})
            else:
                specs.append({"name": name, "config": value})
        return specs

    specs = []
    for value in augmentations:
        specs.extend(
            _normalize_augment_specs(
                value, canonical_name=canonical_name, mapping_name=mapping_name
            )
        )
    return specs


def _config_data(
    config: Any,
    overrides: Mapping[str, Any],
    *,
    defaults: Mapping[str, Any] | None = None,
    config_cls: type | None = None,
) -> dict[str, Any]:
    data = dict(defaults) if defaults is not None else {}
    if config is not None:
        if dataclasses.is_dataclass(config):
            data.update(dataclasses.asdict(config))
        elif isinstance(config, Mapping):
            data.update(config)
        else:
            expected = config_cls.__name__ if config_cls is not None else "dataclass"
            raise TypeError(f"config must be a {expected}, mapping, or None")
    data.update({key: value for key, value in overrides.items() if value is not None})
    return data

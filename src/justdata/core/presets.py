import copy
import dataclasses
import hashlib
import json
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict

_PRESETS: Dict[str, Dict[str, Any]] = {}
_PRESET_LOCK = threading.Lock()


def _plain_value(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)

    if isinstance(value, Mapping):
        return {str(k): _plain_value(v) for k, v in value.items()}

    if isinstance(value, (tuple, list)):
        return [_plain_value(v) for v in value]

    return value


def canonical_preset_json(config: Any) -> str:
    return json.dumps(_plain_value(config), sort_keys=True, separators=(",", ":"))


def preset_hash(config: Any) -> str:
    return hashlib.sha256(canonical_preset_json(config).encode("utf-8")).hexdigest()[
        :16
    ]


@dataclass(frozen=True)
class ResolvedPreset(Mapping[str, Any]):
    name: str
    modality: str
    config: Any

    def to_dict(self) -> Dict[str, Any]:
        plain = _plain_value(self.config)
        return copy.deepcopy(plain)

    def to_json(self) -> str:
        return canonical_preset_json(self.config)

    def hash(self) -> str:
        return preset_hash(self.config)

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self):
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


def register_preset(dataset: str, config: Any, *, modality: str = "core"):
    key = dataset.lower()
    with _PRESET_LOCK:
        presets = _PRESETS.setdefault(modality, {})
        if key in presets:
            raise ValueError(
                f"Preset '{key}' already registered for modality '{modality}'"
            )
        presets[key] = copy.deepcopy(config)


def _find_preset(dataset: str, *, modality: str = "core") -> tuple[str | None, Any]:
    presets = _PRESETS.get(modality, {})
    dataset = dataset.lower()

    if dataset in presets:
        return dataset, presets[dataset]

    for key, val in sorted(presets.items(), key=lambda kv: len(kv[0]), reverse=True):
        if key != "_default" and dataset.startswith(key):
            return key, val

    if "_default" in presets:
        return "_default", presets["_default"]

    return None, {}


def get_dataset_presets(dataset: str, *, modality: str = "core") -> Dict[str, Any]:
    _name, config = _find_preset(dataset, modality=modality)
    return copy.deepcopy(config)


def get_resolved_preset(dataset: str, *, modality: str = "core") -> ResolvedPreset:
    name, config = _find_preset(dataset, modality=modality)
    return ResolvedPreset(
        name=name or dataset.lower(), modality=modality, config=copy.deepcopy(config)
    )


def merge_with_presets(
    dataset: str, user_kwargs: Dict[str, Any], *, modality: str = "core"
) -> Dict[str, Any]:
    """
    Smartly merge user-provided kwargs with modality-specific presets.

    If a user kwarg matches the modality default, it is treated as an
    unmodified default and the dataset-specific preset takes precedence.
    """

    dataset_presets = copy.deepcopy(get_dataset_presets(dataset, modality=modality))
    default_presets = get_dataset_presets("_default", modality=modality)

    def smart_merge(base, user, default):
        result = base.copy()
        for k, v in user.items():
            if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                result[k] = smart_merge(
                    base[k], v, default.get(k, {}) if isinstance(default, dict) else {}
                )
            else:
                default_val = default.get(k) if isinstance(default, dict) else None
                val_to_compare = tuple(v) if isinstance(v, list) else v
                def_to_compare = (
                    tuple(default_val) if isinstance(default_val, list) else default_val
                )

                if (
                    isinstance(val_to_compare, tuple)
                    and len(val_to_compare) > 0
                    and isinstance(val_to_compare[0], list)
                ):
                    val_to_compare = tuple(
                        tuple(x) if isinstance(x, list) else x for x in val_to_compare
                    )
                if (
                    isinstance(def_to_compare, tuple)
                    and len(def_to_compare) > 0
                    and isinstance(def_to_compare[0], list)
                ):
                    def_to_compare = tuple(
                        tuple(x) if isinstance(x, list) else x for x in def_to_compare
                    )

                if (
                    k not in default
                    or val_to_compare != def_to_compare
                    or k not in result
                ):
                    result[k] = v
        return result

    return smart_merge(dataset_presets, user_kwargs, default_presets)

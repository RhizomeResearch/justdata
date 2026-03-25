import copy
import threading
from typing import Any, Dict

_PRESETS: Dict[str, Dict[str, Dict[str, Any]]] = {}
_PRESET_LOCK = threading.Lock()


def register_preset(dataset: str, config: Dict[str, Any], *, modality: str = "core"):
    with _PRESET_LOCK:
        _PRESETS.setdefault(modality, {})[dataset.lower()] = config


def get_dataset_presets(dataset: str, *, modality: str = "core") -> Dict[str, Any]:
    presets = _PRESETS.get(modality, {})
    dataset = dataset.lower()

    if dataset in presets:
        return presets[dataset]

    for key, val in sorted(presets.items(), key=lambda kv: len(kv[0]), reverse=True):
        if key != "_default" and dataset.startswith(key):
            return val

    return presets.get("_default", {})


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

                if k not in default or val_to_compare != def_to_compare or k not in result:
                    result[k] = v
        return result

    return smart_merge(dataset_presets, user_kwargs, default_presets)

from typing import Any, Dict

from justdata.core.presets import (
    get_dataset_presets as _get_dataset_presets,
    merge_with_presets as _merge_with_presets,
    register_preset as _register_preset,
)


def register_preset(dataset: str, config: Dict[str, Any]):
    _register_preset(dataset, config, modality="acoustic")


def get_dataset_presets(dataset: str) -> Dict[str, Any]:
    return _get_dataset_presets(dataset, modality="acoustic")


def merge_with_presets(dataset: str, user_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return _merge_with_presets(dataset, user_kwargs, modality="acoustic")

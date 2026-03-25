from typing import Any, Dict

from justdata.acoustic.configs import AudioPreset
from justdata.core.presets import (
    ResolvedPreset,
    get_dataset_presets as _get_dataset_presets,
    get_resolved_preset as _get_resolved_preset,
    merge_with_presets as _merge_with_presets,
    register_preset as _register_preset,
)


def register_preset(dataset: str, config: Dict[str, Any] | AudioPreset):
    if isinstance(config, AudioPreset):
        config = config.to_dict()
    _register_preset(dataset, config, modality="acoustic")


def get_dataset_presets(dataset: str) -> Dict[str, Any]:
    return _get_dataset_presets(dataset, modality="acoustic")


def get_resolved_preset(dataset: str) -> ResolvedPreset:
    return _get_resolved_preset(dataset, modality="acoustic")


def merge_with_presets(dataset: str, user_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return _merge_with_presets(dataset, user_kwargs, modality="acoustic")

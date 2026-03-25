from justdata.core.adapters import get_adapter, register_adapter
from justdata.core.loader import fetch_ds, load_ds
from justdata.core.presets import (
    get_dataset_presets,
    merge_with_presets,
    register_preset,
)
from justdata.core.registry import (
    DataPipeline,
    DatasetInfo,
    get_dataset_info,
    get_pipeline,
    get_pipeline_for_dataset,
    get_task_for_dataset,
    register_dataset,
    register_pipeline,
)
from justdata.core.sources import (
    get_source_loader,
    register_default_source_loader,
    register_source_loader,
)

__all__ = [
    "DataPipeline",
    "DatasetInfo",
    "fetch_ds",
    "get_adapter",
    "get_dataset_info",
    "get_dataset_presets",
    "get_pipeline",
    "get_pipeline_for_dataset",
    "get_source_loader",
    "get_task_for_dataset",
    "load_ds",
    "merge_with_presets",
    "register_adapter",
    "register_dataset",
    "register_default_source_loader",
    "register_pipeline",
    "register_preset",
    "register_source_loader",
]

__version__ = "0.1.0"

from justdata.adapters import get_adapter, register_adapter
from justdata.loader import create_minic_datasets, fetch_ds, load_ds
from justdata.presets import get_dataset_presets, merge_with_presets, register_preset
from justdata.registry import (
    DataPipeline,
    get_pipeline,
    get_pipeline_for_dataset,
    get_task_for_dataset,
    register_dataset,
    register_pipeline,
)

__all__ = [
    "__version__",
    # Loader
    "load_ds",
    "fetch_ds",
    "create_minic_datasets",
    # Pipeline registry
    "DataPipeline",
    "get_pipeline",
    "get_pipeline_for_dataset",
    "register_pipeline",
    # Dataset registry
    "get_task_for_dataset",
    "register_dataset",
    # Presets
    "get_dataset_presets",
    "merge_with_presets",
    "register_preset",
    # Adapters
    "get_adapter",
    "register_adapter",
]

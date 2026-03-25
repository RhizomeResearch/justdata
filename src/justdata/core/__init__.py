from justdata.core.adapters import get_adapter, register_adapter
from justdata.core.filters import (
    filter_by_metadata,
    groupby_metadata,
    metadata_filter_predicate,
)
from justdata.core.loader import fetch_ds, load_ds
from justdata.core.metadata import apply_metadata_mode, numeric_metadata
from justdata.core.presets import (
    ResolvedPreset,
    get_resolved_preset,
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
    has_pipeline,
    list_pipelines,
    register_dataset,
    register_pipeline,
)
from justdata.core.sources import (
    get_source_loader,
    register_default_source_loader,
    register_source_loader,
)
from justdata.core.stats import compute_feature_stats, make_stats_iterator

__all__ = [
    "DataPipeline",
    "DatasetInfo",
    "ResolvedPreset",
    "apply_metadata_mode",
    "compute_feature_stats",
    "filter_by_metadata",
    "fetch_ds",
    "get_adapter",
    "get_dataset_info",
    "get_dataset_presets",
    "get_pipeline",
    "get_pipeline_for_dataset",
    "get_resolved_preset",
    "get_source_loader",
    "get_task_for_dataset",
    "groupby_metadata",
    "has_pipeline",
    "list_pipelines",
    "load_ds",
    "make_stats_iterator",
    "merge_with_presets",
    "metadata_filter_predicate",
    "numeric_metadata",
    "register_adapter",
    "register_dataset",
    "register_default_source_loader",
    "register_pipeline",
    "register_preset",
    "register_source_loader",
]

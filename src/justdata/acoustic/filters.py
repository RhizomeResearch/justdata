from justdata.core.filters import (
    filter_by_metadata,
    groupby_metadata,
    metadata_filter_predicate,
)


filter = filter_by_metadata
groupby = groupby_metadata


__all__ = [
    "filter",
    "filter_by_metadata",
    "groupby",
    "groupby_metadata",
    "metadata_filter_predicate",
]

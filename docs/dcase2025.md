# DCASE 2025 Task 1

`justdata.acoustic.dcase2025` provides source loading, metadata parsing,
split-safety checks, source/target dataset helpers, and DCASE metrics for Task 1
acoustic scene classification.

## Direct EfficientAT 1-second view

```python
import justdata.acoustic
from justdata.core.registry import get_pipeline
from justdata.core.loader import load_ds

pipeline = get_pipeline(
    dataset="dcase2025_task1",
    preset="dcase2025_task1_efficientat_32k_1s",
)

ds, n = load_ds(
    dataset_names_arg=["dcase2025:task1"],
    splits_arg={"dcase2025:task1": ["dev_train_25"]},
    dataset_type="train",
    batch_size=64,
    seed=0,
    pipeline=pipeline,
    num_classes=10,
    cache_dataset=False,
    data_dir="/path/to/dcase",
    metadata_mode="numeric_only",
    as_numpy=True,
)
```

## Split safety

Statistics are allowed on `dev_train_25` and blocked on `dev_test` and `eval` by
default. This prevents accidental target leakage:

```python
from justdata.acoustic.dcase2025 import assert_stats_allowed

assert_stats_allowed("dev_train_25")
```

Use `allow_override=True` only when a protocol explicitly permits it.

## Source stats for DCASE device A

```python
from justdata.acoustic.dcase2025 import make_source_dataset
from justdata.acoustic.stats import compute_feature_stats
from justdata.core.registry import get_pipeline

pipeline = get_pipeline(
    dataset="dcase2025_task1",
    preset="dcase2025_task1_efficientat_32k_1s",
    output_key="inputs",
)

source_ds, n = make_source_dataset(
    split="dev_train_25",
    source_domain={"device": "A"},
    preset="dcase2025_task1_efficientat_32k_1s",
    pipeline=pipeline,
    data_dir="/path/to/dcase",
)

stats = compute_feature_stats(
    source_ds,
    axes=("time",),
    groupby="device",
    feature_key="inputs",
)
```

`source_domain` and `target_domain` are evaluated on raw manifest records before
waveform decoding. For additional raw-record selection, pass
`source_filter_fn`; it may use only manifest fields. The generic `filter_fn`
keeps its separate behavior and runs later, after adaptation, preprocessing,
and preprocessing-cache lookup, so it may inspect canonical or preprocessed
fields.

In executable scripts, destructure `make_source_dataset(...)` as
`source_ds, n = make_source_dataset(...)`. If the selected preset writes model
features under `features`, pass a pipeline with `output_key="inputs"` or change
`feature_key` accordingly.

## Metadata modes

Use `metadata_mode="numeric_only"` for JAX-friendly batches. String metadata can
be written to a sidecar JSONL file with `sidecar_metadata_path`.

```python
ds, n = load_ds(
    dataset_names_arg=["dcase2025:task1"],
    splits_arg={"dcase2025:task1": ["dev_train_25"]},
    dataset_type="validation",
    batch_size=64,
    seed=0,
    pipeline=pipeline,
    cache_dataset=False,
    data_dir="/path/to/dcase",
    metadata_mode="numeric_only",
    sidecar_metadata_path="dcase_metadata.jsonl",
)
```

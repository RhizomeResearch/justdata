import justdata.acoustic  # noqa: F401
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

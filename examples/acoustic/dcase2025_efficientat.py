import justdata.acoustic  # noqa: F401
from justdata.core.loader import load_ds
from justdata.core.registry import get_pipeline


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
    metadata_mode="numeric_only",
    as_numpy=True,
)

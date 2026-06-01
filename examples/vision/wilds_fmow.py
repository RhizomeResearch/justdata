import justdata.vision  # noqa: F401
from justdata.core.loader import load_ds
from justdata.core.registry import get_pipeline


wilds_ds = "wilds:fmow?split_scheme=time_after_2016"
pipeline = get_pipeline(dataset="wilds:fmow")

ds, n = load_ds(
    dataset_names_arg=[wilds_ds],
    splits_arg={wilds_ds: ["train"]},
    dataset_type="train",
    batch_size=32,
    seed=0,
    pipeline=pipeline,
    num_classes=62,
    data_dir="data",
    metadata_mode="numeric_only",
)

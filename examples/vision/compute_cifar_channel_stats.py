import justdata.vision  # noqa: F401
from justdata.core.loader import load_ds
from justdata.core.registry import get_pipeline
from justdata.core.stats import compute_feature_stats


pipeline = get_pipeline(dataset="cifar10")

ds, n = load_ds(
    dataset_names_arg=["cifar10"],
    splits_arg={"cifar10": ["train"]},
    dataset_type="validation",
    batch_size=128,
    seed=0,
    pipeline=pipeline,
    num_classes=10,
    metadata_mode="numeric_only",
)

stats = compute_feature_stats(
    ds,
    axes=(0, 2, 3),
    groupby=None,
    feature_key="image",
)

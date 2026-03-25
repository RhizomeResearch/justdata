import justdata.vision  # noqa: F401
from justdata.core.registry import get_pipeline
from justdata.vision.minic import create_minic_datasets


pipeline = get_pipeline(dataset="cifar10")

corrupted_datasets, n = create_minic_datasets(
    corruption_types=["noise", "blur"],
    severity=3,
    dataset_names_arg=["cifar10"],
    splits_arg={"cifar10": ["test"]},
    dataset_type="validation",
    batch_size=128,
    seed=0,
    pipeline=pipeline,
    num_classes=10,
    metadata_mode="numeric_only",
)

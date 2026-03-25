import justdata.acoustic  # noqa: F401
from justdata.acoustic.corruptions.datasets import create_audio_corruption_datasets
from justdata.core.registry import get_pipeline


pipeline = get_pipeline(
    dataset="dcase2025_task1",
    preset="dcase2025_task1_efficientat_32k_1s",
)

corrupted_datasets, n = create_audio_corruption_datasets(
    corruption_types=["additive_white_noise", "clipping"],
    severity=3,
    base_dataset="dcase2025:task1",
    preset="dcase2025_task1_efficientat_32k_1s",
    split="dev_test",
    pipeline=pipeline,
    batch_size=64,
    seed=0,
    num_classes=10,
    metadata_mode="numeric_only",
)

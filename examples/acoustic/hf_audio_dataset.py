import justdata.acoustic  # noqa: F401
from justdata.core.loader import load_ds
from justdata.core.registry import get_pipeline


pipeline = get_pipeline(dataset="speech_commands")

ds, n = load_ds(
    dataset_names_arg=["hf_audio:speech_commands"],
    splits_arg={"hf_audio:speech_commands": ["train"]},
    dataset_type="train",
    batch_size=32,
    seed=0,
    pipeline=pipeline,
    num_classes=35,
    metadata_mode="numeric_only",
)

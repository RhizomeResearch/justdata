import matplotlib.pyplot as plt

import justdata.vision  # noqa: F401
from justdata.core.loader import load_ds
from justdata.core.registry import get_pipeline


DATASET = "zenodo:8027520?file=imagenet16.zip"
NUM_CLASSES = 16
PLOT_SAMPLES = 16
PLOT_PATH = "imagenet16_samples.png"


def _decode(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def plot_samples(ds, *, output_path=PLOT_PATH, n=PLOT_SAMPLES):
    samples = list(ds.take(n).as_numpy_iterator())
    cols = 4
    rows = (len(samples) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(8, 8))
    axes = axes.reshape(-1)

    for ax, sample in zip(axes, samples):
        ax.imshow(sample["image"])
        ax.set_title(_decode(sample["metadata"]["class_name"]), fontsize=9)
        ax.axis("off")

    for ax in axes[len(samples) :]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.show()


pipeline = get_pipeline(dataset=DATASET)

train_ds, train_n = load_ds(
    dataset_names_arg=[DATASET],
    splits_arg={DATASET: ["train"]},
    dataset_type="train",
    batch_size=128,
    seed=0,
    pipeline=pipeline,
    num_classes=NUM_CLASSES,
    cache_dataset=False,
    metadata_mode="numeric_only",
)

val_ds, val_n = load_ds(
    dataset_names_arg=[DATASET],
    splits_arg={DATASET: ["val"]},
    dataset_type="validation",
    batch_size=128,
    seed=0,
    pipeline=pipeline,
    num_classes=NUM_CLASSES,
    cache_dataset=False,
    metadata_mode="numeric_only",
)

raw_val_ds, _ = load_ds(
    dataset_names_arg=[DATASET],
    splits_arg={DATASET: ["val"]},
    dataset_type="validation",
    batch_size=PLOT_SAMPLES,
    seed=0,
    pipeline=pipeline,
    num_classes=NUM_CLASSES,
    cache_dataset=False,
    return_raw_ds=True,
)

plot_samples(raw_val_ds)

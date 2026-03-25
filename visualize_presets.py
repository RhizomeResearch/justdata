"""Visualization script for all justdata augmentation presets.

Produces one PNG per preset showing training samples (top) and validation
samples (bottom), plus a dedicated DINOv2 multi-crop figure.

Usage:
    uv run python visualize_presets.py
"""

import copy

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from justdata.core.loader import load_ds
from justdata.core.registry import get_pipeline, get_pipeline_for_dataset
from justdata.vision.presets import get_dataset_presets


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_norm_params(preset_name: str):
    preset = get_dataset_presets(preset_name)
    return preset.get("postproc_kwargs", {}).get(
        "normalization_params",
        ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    )


def _denorm(images: np.ndarray, mean, std) -> np.ndarray:
    """Reverse normalization; converts NCHW → NHWC if needed."""
    # Batch: NCHW → NHWC
    if images.ndim == 4 and images.shape[1] <= 4 and images.shape[-1] > 4:
        images = np.transpose(images, [0, 2, 3, 1])
    return np.clip(images * np.array(std) + np.array(mean), 0, 1)


def _build_pipeline(preset_name: str, num_classes: int, is_training: bool):
    """Build the 4-function pipeline from a registered preset name."""
    kwargs = copy.deepcopy(get_dataset_presets(preset_name))
    pp = kwargs.setdefault("postproc_kwargs", {})
    pp["is_training"] = is_training
    pp["num_classes"] = num_classes
    return get_pipeline("vision/classification", **kwargs)


def _fetch(
    dataset_name,
    split,
    num_classes,
    preproc,
    aug,
    laug,
    postproc,
    batch_size,
    dataset_type="train",
):
    ds, _ = load_ds(
        dataset_names_arg=dataset_name,
        splits_arg=split,
        dataset_type=dataset_type,
        batch_size=batch_size,
        seed=42,
        preprocess_fn=preproc,
        augment_fn=aug,
        late_augment_fn=laug,
        postprocess_fn=postproc,
        num_classes=num_classes,
        cache_dataset=False,
    )
    return next(iter(ds))


# ---------------------------------------------------------------------------
# Supervised Learning visualization
# ---------------------------------------------------------------------------


def visualize_sl_preset(
    preset_name: str,
    dataset_name: str,
    split: str,
    num_classes: int,
    title: str,
    filename: str,
    rows: int = 4,
    cols: int = 5,
):
    """Two-block figure: top block = training, bottom block = validation."""
    norm = _get_norm_params(preset_name)
    n = rows * cols

    # Training pipeline
    preproc, aug, laug, postproc_tr = _build_pipeline(preset_name, num_classes, True)
    batch_tr = _fetch(dataset_name, split, num_classes, preproc, aug, laug, postproc_tr, n, "train")
    imgs_tr = _denorm(batch_tr["image"].numpy(), *norm)

    # Validation pipeline (augmentation disabled by dataset_type != "train")
    _, _, _, postproc_val = _build_pipeline(preset_name, num_classes, False)
    batch_val = _fetch(dataset_name, split, num_classes, preproc, aug, laug, postproc_val, n, "val")
    imgs_val = _denorm(batch_val["image"].numpy(), *norm)

    fig, axes = plt.subplots(rows * 2 + 1, cols, figsize=(cols * 2.2, rows * 4.6))
    fig.suptitle(title, fontsize=10, y=1.005)

    # Spacer row index between train and val blocks
    spacer = rows

    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            axes[r, c].imshow(imgs_tr[idx])
            axes[r, c].axis("off")
            axes[spacer + 1 + r, c].imshow(imgs_val[idx])
            axes[spacer + 1 + r, c].axis("off")

    # Spacer row (invisible) with section labels
    for c in range(cols):
        axes[spacer, c].axis("off")
    axes[0, 0].set_title("Train", fontsize=9, loc="left")
    axes[spacer + 1, 0].set_title("Val", fontsize=9, loc="left")

    plt.tight_layout()
    plt.savefig(filename, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Saved → {filename}")


# ---------------------------------------------------------------------------
# DINOv2 SSL visualization
# ---------------------------------------------------------------------------


def visualize_dinov2(
    dataset_name: str,
    split: str,
    num_classes: int,
    filename: str,
    n_samples: int = 4,
    n_local_show: int = 4,
):
    """Grid of [Global1 | Global2 | Local1 … Local4] per sample."""
    norm = _get_norm_params("dinov2")
    preproc, aug, laug, postproc = _build_pipeline("dinov2", num_classes, True)
    batch = _fetch(dataset_name, split, num_classes, preproc, aug, laug, postproc, n_samples, "train")

    n_global = 2
    n_cols = n_global + n_local_show

    fig, axes = plt.subplots(
        n_samples, n_cols, figsize=(n_cols * 2.4, n_samples * 2.4)
    )
    fig.suptitle("DINOv2 SSL — Multi-Crop Asymmetric Pipeline", fontsize=12)

    col_titles = [
        "Global 1\n(blur p=1.0, solar p=0.0)",
        "Global 2\n(blur p=0.1, solar p=0.2)",
    ] + [f"Local {i+1}\n(blur p=0.5)" for i in range(n_local_show)]

    for s in range(n_samples):
        for g in range(n_global):
            # global_crops: [B, 2, C, H, W] after CHW permute in postproc
            crop = batch["global_crops"][s, g].numpy()
            crop = np.transpose(crop, [1, 2, 0])
            crop = np.clip(crop * np.array(norm[1]) + np.array(norm[0]), 0, 1)
            ax = axes[s, g]
            ax.imshow(crop)
            ax.axis("off")
            if s == 0:
                ax.set_title(col_titles[g], fontsize=8)

        for lc in range(n_local_show):
            # local_crops: [B, 8, C, h, w] after CHW permute
            crop = batch["local_crops"][s, lc].numpy()
            crop = np.transpose(crop, [1, 2, 0])
            crop = np.clip(crop * np.array(norm[1]) + np.array(norm[0]), 0, 1)
            ax = axes[s, n_global + lc]
            ax.imshow(crop)
            ax.axis("off")
            if s == 0:
                ax.set_title(col_titles[n_global + lc], fontsize=8)

    plt.tight_layout()
    plt.savefig(filename, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Saved → {filename}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tf.config.set_visible_devices([], "GPU")

    # --- CIFAR-10 ---
    visualize_sl_preset(
        preset_name="cifar",
        dataset_name="cifar10",
        split="train[:5%]",
        num_classes=10,
        title=(
            "CIFAR-10 | TrivialAugment + RandomCrop(32, pad=4, zeros) + HFlip\n"
            "Mixup α=0.8 / CutMix α=1.0 | RE p=0.25 | norm=(0.4914…, 0.2023…)"
        ),
        filename="viz_cifar10.png",
    )

    # --- CIFAR-100 ---
    visualize_sl_preset(
        preset_name="cifar100",
        dataset_name="cifar100",
        split="train[:5%]",
        num_classes=100,
        title=(
            "CIFAR-100 | TrivialAugment + RandomCrop(32, pad=4, zeros) + HFlip\n"
            "Mixup α=0.8 / CutMix α=1.0 | RE p=0.25 | norm=(0.5071…, 0.2675…)"
        ),
        filename="viz_cifar100.png",
    )

    # --- ImageNet default (ViT / ConvNeXt modern branch) ---
    visualize_sl_preset(
        preset_name="_default",
        dataset_name="imagenette",
        split="train[:5%]",
        num_classes=10,
        title=(
            "ImageNet Default — ViT / ConvNeXt | RandAugment(n=2, m=9) + RRC(bicubic) + HFlip\n"
            "Mixup α=0.8 / CutMix α=1.0 | RE p=0.25 | val: resize-256 → crop-224 (0.875)"
        ),
        filename="viz_imagenet_default.png",
    )

    # --- ImageNet ResNet legacy (ColorJitter, no RandAugment) ---
    visualize_sl_preset(
        preset_name="imagenet_resnet",
        dataset_name="imagenette",
        split="train[:5%]",
        num_classes=10,
        title=(
            "ImageNet ResNet Legacy | ColorJitter(b=0.4, c=0.4, s=0.4, h=0.1) + RRC(bicubic) + HFlip\n"
            "Mixup α=0.2 / CutMix α=1.0 | RE p=0.25 | val: resize-256 → crop-224"
        ),
        filename="viz_imagenet_resnet.png",
    )

    # --- RSB A1 — Heavy ---
    visualize_sl_preset(
        preset_name="imagenet_a1",
        dataset_name="imagenette",
        split="train[:5%]",
        num_classes=10,
        title=(
            "ImageNet RSB-A1 Heavy (ResNet-152, 600ep, BCE) | RandAugment(n=2, m=7) + RRC(bicubic)\n"
            "Mixup α=0.2 / CutMix α=1.0 | RE p=0.35 | val: direct resize-224 (crop_pct=1.0)"
        ),
        filename="viz_imagenet_a1.png",
    )

    # --- RSB A2 — Moderate ---
    visualize_sl_preset(
        preset_name="imagenet_a2",
        dataset_name="imagenette",
        split="train[:5%]",
        num_classes=10,
        title=(
            "ImageNet RSB-A2 Moderate (ResNet-50, 300ep, BCE) | RandAugment(n=2, m=6) + RRC(bicubic)\n"
            "Mixup α=0.2 / CutMix α=1.0 | RE p=0.25 | val: direct resize-224 (crop_pct=1.0)"
        ),
        filename="viz_imagenet_a2.png",
    )

    # --- RSB A3 — Light (FixRes 160 → 224) ---
    visualize_sl_preset(
        preset_name="imagenet_a3",
        dataset_name="imagenette",
        split="train[:5%]",
        num_classes=10,
        title=(
            "ImageNet RSB-A3 Light (ResNet-50 fast, 100ep, CE) | RandAugment(n=2, m=6) + RRC(bicubic, 160px)\n"
            "Mixup α=0.1 / CutMix α=1.0 | RE disabled | val: FixRes resize-236 → crop-224"
        ),
        filename="viz_imagenet_a3.png",
    )

    # --- DINOv2 SSL ---
    visualize_dinov2(
        dataset_name="imagenette",
        split="train[:5%]",
        num_classes=10,
        filename="viz_dinov2.png",
    )

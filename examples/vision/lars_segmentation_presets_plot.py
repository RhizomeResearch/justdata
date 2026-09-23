"""Render five LaRS training samples under all three segmentation presets.

Matplotlib is only needed to run this example; it is not a JustData dependency.
First create matching semantic and panoptic train inventories with --limit 5
using lars_local_inventory.py, then pass their output directories here.
"""

import argparse
import json
from itertools import islice
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.patches import Patch

import justdata.vision  # noqa: F401 - register vision pipelines and presets
from justdata.core import get_pipeline, open_inventory


PRESETS = ("segmentation_a1", "segmentation_a2", "segmentation_a3")
SEMANTIC_COLORS = {
    0: (210, 70, 65),  # obstacle
    1: (45, 115, 210),  # water
    2: (105, 205, 225),  # sky
    255: (30, 30, 30),  # ignore
}


def _source(directory, labels):
    info = json.loads((directory / "source.json").read_text(encoding="utf-8"))
    if info["split"] != "train" or info["labels"] != labels:
        raise ValueError(f"{directory} must contain a {labels} train inventory")
    if info["selected_count"] < 5:
        raise ValueError(f"{directory} needs at least five selected samples")
    return info, open_inventory(directory / "snapshot").dataset


def _pipeline(name, preset, panoptic_info):
    overrides = (
        {"geometry_kwargs": {"class_values": (0, 1, 2), "ignore_value": 255}}
        if name == "vision/segmentation"
        else {
            "panoptic_kwargs": {
                "class_values": tuple(
                    sorted(category["id"] for category in panoptic_info["categories"])
                ),
                "thing_class_values": tuple(
                    sorted(
                        category["id"]
                        for category in panoptic_info["categories"]
                        if category["isthing"]
                    )
                ),
                "void_value": 0,
                "max_segments": panoptic_info["max_segments"],
            }
        }
    )
    # Show the source RGB domain directly, without undoing normalization.
    overrides["postproc_kwargs"] = {"normalize_image": False, "permute_image": False}
    pre, augment, _, post = get_pipeline(
        name, preset=preset, overrides=overrides
    ).build(is_training=True)
    return lambda sample, seed: post(augment(pre(sample), seed))


def _semantic_rgb(mask):
    rgb = np.empty((*mask.shape, 3), dtype=np.uint8)
    for label, color in SEMANTIC_COLORS.items():
        rgb[mask == label] = color
    return rgb


def _panoptic_rgb(mask, segments, category_colors):
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for segment_id, category_id, valid in zip(
        segments["segment_ids"].numpy(),
        segments["category_ids"].numpy(),
        segments["valid_mask"].numpy(),
        strict=True,
    ):
        if valid:
            rgb[mask == segment_id] = category_colors[int(category_id)]
    # Draw segment boundaries so two instances of the same category remain distinct.
    edges = np.zeros(mask.shape, dtype=bool)
    edges[1:, :] |= mask[1:, :] != mask[:-1, :]
    edges[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    rgb[edges & (mask != 0)] = 0
    return rgb


def plot(semantic_dir: Path, panoptic_dir: Path, output: Path, seed: int):
    if output.suffix.lower() != ".png":
        raise ValueError("output must be a .png file")
    if output.exists():
        raise FileExistsError(output)
    semantic_info, semantic_ds = _source(semantic_dir, "semantic")
    panoptic_info, panoptic_ds = _source(panoptic_dir, "panoptic")
    if semantic_info["selected_ids_sha256"] != panoptic_info["selected_ids_sha256"]:
        raise ValueError("Semantic and panoptic inventories select different images")
    categories = sorted(panoptic_info["categories"], key=lambda row: row["id"])
    palette = plt.get_cmap("tab20")(np.linspace(0, 1, len(categories)))
    category_colors = {
        row["id"]: tuple(np.asarray(color[:3]) * 255)
        for row, color in zip(categories, palette, strict=True)
    }
    pipelines = {
        preset: (
            _pipeline("vision/segmentation", preset, panoptic_info),
            _pipeline("vision/panoptic_segmentation", preset, panoptic_info),
        )
        for preset in PRESETS
    }
    fig, axes = plt.subplots(5, 9, figsize=(25, 15), squeeze=False)
    fig.suptitle("LaRS training views: RGB, semantic labels, panoptic labels")
    for row_index, (semantic_row, panoptic_row) in enumerate(
        islice(zip(semantic_ds, panoptic_ds, strict=True), 5)
    ):
        example_id = semantic_row["metadata"]["example_id"].numpy()
        if example_id != panoptic_row["metadata"]["example_id"].numpy():
            raise ValueError("Semantic and panoptic rows have different image IDs")
        sample_seed = tf.constant([seed, row_index], tf.int32)
        for preset_index, preset in enumerate(PRESETS):
            semantic_view = pipelines[preset][0](semantic_row, sample_seed)
            panoptic_view = pipelines[preset][1](panoptic_row, sample_seed)
            image = semantic_view["image"].numpy()
            if not np.array_equal(image, panoptic_view["image"].numpy()):
                raise ValueError("Semantic and panoptic views are not aligned")
            panels = (
                np.clip(image, 0, 255).astype(np.uint8),
                _semantic_rgb(semantic_view["mask"].numpy()),
                _panoptic_rgb(
                    panoptic_view["panoptic_mask"].numpy(),
                    panoptic_view["segments"],
                    category_colors,
                ),
            )
            for panel_index, panel in enumerate(panels):
                ax = axes[row_index, 3 * preset_index + panel_index]
                ax.imshow(panel, interpolation="nearest")
                ax.set_axis_off()
                if row_index == 0:
                    ax.set_title(
                        f"{preset.upper()}\n"
                        + ("RGB", "semantic", "panoptic")[panel_index]
                    )
                if preset_index == panel_index == 0:
                    ax.text(
                        -0.08,
                        0.5,
                        example_id.decode().rsplit("/", 1)[-1],
                        rotation=90,
                        va="center",
                        ha="right",
                        transform=ax.transAxes,
                    )
    handles = [
        Patch(facecolor=np.asarray(color) / 255, label=f"semantic: {name}")
        for name, color in zip(
            ("obstacle", "water", "sky", "ignore"),
            SEMANTIC_COLORS.values(),
            strict=True,
        )
    ]
    handles.extend(
        Patch(
            facecolor=np.asarray(category_colors[row["id"]]) / 255,
            label=f"panoptic: {row['name']}",
        )
        for row in categories
    )
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8)
    fig.tight_layout(rect=(0, 0.09, 1, 0.97))
    fig.savefig(output, dpi=110, bbox_inches="tight", format="png")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic-inventory", type=Path, required=True)
    parser.add_argument("--panoptic-inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    tf.config.set_visible_devices([], "GPU")
    plot(args.semantic_inventory, args.panoptic_inventory, args.output, args.seed)

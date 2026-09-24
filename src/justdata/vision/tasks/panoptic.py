"""Panoptic data stages with recorded, paired geometry."""

import tensorflow as tf

from justdata.vision.encodings.panoptic_targets import (
    validate_panoptic_sample,
    with_panoptic_targets,
)
from justdata.vision.geometry import replay_panoptic_geometry, sample_panoptic_geometry
from justdata.vision.stages import (
    finish_recorded_view,
    normalize_image_format,
    recorded_view_augmentations,
)


def make_panoptic_preprocessing(
    *,
    class_values,
    thing_class_values,
    void_value,
    max_segments,
    keep_original_annotations=False,
):
    def preprocessing(sample):
        sample = validate_panoptic_sample(
            normalize_image_format(sample),
            class_values=class_values,
            thing_class_values=thing_class_values,
            void_value=void_value,
            max_segments=max_segments,
        )
        if keep_original_annotations:
            sample = sample | {
                "original_panoptic_mask": sample["panoptic_mask"],
                "original_segments": sample["segments"],
            }
            if "annotation_valid_mask" in sample:
                sample["original_annotation_valid_mask"] = sample[
                    "annotation_valid_mask"
                ]
        return sample

    return preprocessing


def make_panoptic_augmentations(
    config, geometry, *, color_jitter_kwargs=None, photometric_kwargs=None
):
    def replay_view(sample, seed):
        record = sample_panoptic_geometry(
            tf.shape(sample["image"])[:2],
            geometry,
            void_value=config["void_value"],
            is_training=True,
            seed=seed,
        )
        return replay_panoptic_geometry(sample, record, **config)

    return recorded_view_augmentations(
        replay_view,
        color_jitter_kwargs=color_jitter_kwargs,
        photometric_kwargs=photometric_kwargs,
    )


def make_panoptic_postprocessing(
    config,
    geometry,
    *,
    is_training=False,
    normalize_image=True,
    normalization_params=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    permute_image=True,
    emit_panoptic_targets=False,
):
    def postprocessing(sample, **kwargs):
        if not is_training:
            record = sample_panoptic_geometry(
                tf.shape(sample["image"])[:2],
                geometry,
                void_value=config["void_value"],
                is_training=False,
            )
            sample = replay_panoptic_geometry(sample, record, **config)
        sample = finish_recorded_view(
            sample,
            geometry,
            is_training=is_training,
            label_keys=("panoptic_mask",),
            normalize_image=normalize_image,
            normalization_params=normalization_params,
            permute_image=permute_image,
        )
        if emit_panoptic_targets:
            sample = with_panoptic_targets(sample, **config)
        return sample

    return postprocessing


__all__ = [
    "make_panoptic_preprocessing",
    "make_panoptic_augmentations",
    "make_panoptic_postprocessing",
]

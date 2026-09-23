"""Panoptic data stages with recorded, paired geometry."""

import tensorflow as tf

from justdata.vision.encodings.panoptic_targets import (
    validate_panoptic_sample,
    with_panoptic_targets,
)
from justdata.vision.geometry import sample_panoptic_geometry
from justdata.vision.panoptic_geometry import replay_panoptic_geometry
from justdata.vision.stages import normalize_image_format
from justdata.vision.transforms import nhwc_to_nchw, normalize


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
    from justdata.vision.augmentations.color import (
        color_jitter,
        photometric_distortion,
    )

    def augmentations(sample, seed):
        seeds = tf.random.experimental.stateless_split(seed, 2)
        if photometric_kwargs is not None:
            sample = sample | {
                "image": photometric_distortion(
                    sample["image"], seeds[1], **photometric_kwargs
                )
            }
        record = sample_panoptic_geometry(
            tf.shape(sample["image"])[:2],
            geometry,
            void_value=config["void_value"],
            is_training=True,
            seed=seeds[0],
        )
        sample = replay_panoptic_geometry(sample, record, **config)
        if color_jitter_kwargs is not None:
            sample = sample | {
                "image": color_jitter(sample["image"], seeds[1], **color_jitter_kwargs)
            }
        return sample

    return augmentations


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
        else:
            size = (
                (geometry.train_crop_size + geometry.patch_size - 1)
                // geometry.patch_size
                * geometry.patch_size
            )
            sample["image"].set_shape([size, size, 3])
            for key in (
                "panoptic_mask",
                "source_valid_mask",
                "pixel_valid_mask",
                "annotation_valid_mask",
            ):
                if key in sample:
                    sample[key].set_shape([size, size])
        image = tf.cast(sample["image"], tf.float32)
        if normalize_image:
            image = normalize(image, *normalization_params)
        if permute_image:
            image = nhwc_to_nchw(image)
        sample = sample | {"image": image}
        if emit_panoptic_targets:
            sample = with_panoptic_targets(sample, **config)
        return sample

    return postprocessing


__all__ = [
    "make_panoptic_preprocessing",
    "make_panoptic_augmentations",
    "make_panoptic_postprocessing",
]

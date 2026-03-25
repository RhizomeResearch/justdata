from typing import Literal

import tensorflow as tf

from justdata.augmentations.composed import (
    create_global_crops,
    create_local_crops,
)
from justdata.augmentations.mixing import (
    mixup_cutmix,
    random_erasing,
)
from justdata.augmentations.registry import get_augment_strategy, get_crop_strategy
from justdata.stages import normalize_image_format, resize_and_normalize
from justdata.transforms import nhwc_to_nchw


def make_preprocessing(image_key: str = "image"):
    def preprocessing(sample):
        return normalize_image_format(sample, image_key=image_key)

    return preprocessing


def make_augmentations(
    image_size: int,
    enable: bool = True,
    local_crops: bool = True,
    n_global_crops: int = 2,
    n_local_crops: int = 8,
    mode: Literal["ssl", "sl"] = "sl",
    ra_kwargs: dict = None,
    ta_kwargs: dict = None,
    cj_kwargs: dict = None,
    gc_kwargs: dict = None,
    lc_kwargs: dict = None,
    crop_type: str = "random_resized",
    interpolation: str = "bilinear",
    padding: int = 4,
    pad_mode: str = "REFLECT",
    augment_type: str = "rand_augment",
):
    if not enable:

        def no_op(sample, seed):
            return sample

        return no_op

    crop_fn = get_crop_strategy(crop_type)
    aug_fn = get_augment_strategy(augment_type)

    ra_kwargs = ra_kwargs or {}
    ta_kwargs = ta_kwargs or {}
    cj_kwargs = cj_kwargs or {}
    gc_kwargs = gc_kwargs or {}
    lc_kwargs = lc_kwargs or {}

    gc_kwargs.setdefault("size", image_size)
    gc_kwargs.setdefault("scale", (0.4, 1.0))
    lc_kwargs.setdefault("size", int(image_size * 0.425))
    lc_kwargs.setdefault("scale", (0.05, 0.4))

    # Select augmentation kwargs based on strategy
    if augment_type == "rand_augment":
        _aug_kwargs = ra_kwargs
    elif augment_type == "trivial_augment":
        _aug_kwargs = ta_kwargs
    elif augment_type == "color_jitter":
        _aug_kwargs = cj_kwargs
    else:
        _aug_kwargs = {}

    def augmentations(sample, seed):
        seeds = tf.random.split(seed, 2)
        image = sample["image"]

        res = {}
        if "label" in sample:
            res["label"] = sample["label"]

        if mode == "ssl":
            res["global_crops"] = create_global_crops(
                image, crops_number=n_global_crops, seed=seeds[0], **gc_kwargs
            )
            if local_crops:
                res["local_crops"] = create_local_crops(
                    image, crops_number=n_local_crops, seed=seeds[1], **lc_kwargs
                )
        elif mode == "sl":
            cropped_image = crop_fn(
                image,
                size=image_size,
                seed=seeds[0],
                padding=padding,
                pad_mode=pad_mode,
                interpolation=interpolation,
            )
            aug_image = aug_fn(cropped_image, seeds[1], **_aug_kwargs)
            res["image"] = aug_image
        else:
            raise ValueError(f"Unknown mode `{mode}`. Expected one of ['sl', 'ssl']")

        # Pass through any other keys
        for k, v in sample.items():
            if k not in res and k not in ("image", "label"):
                res[k] = v

        return res

    return augmentations


def make_late_augmentations(
    enable: bool = True,
    mixup_alpha: float = 0.8,
    cutmix_alpha: float = 1.0,
    prob: float = 1.0,
    switch_prob: float = 0.5,
    label_smoothing: float = 0.1,
    bce_target: bool = False,
    permute_image: bool = True,
    mode: Literal["ssl", "sl"] = "sl",
    random_erasing_prob: float = 0.0,
):
    def late_augmentations(sample, num_classes, seed):
        if (
            not enable
            or mode == "ssl"
            or "image" not in sample
            or "label" not in sample
        ):
            return sample

        images = sample["image"]
        labels = sample["label"]

        seeds = tf.random.split(seed, 2)

        input_is_nchw = tf.logical_and(
            tf.shape(images)[1] <= 4,
            tf.shape(images)[-1] > 4,
        )
        images = tf.cond(
            input_is_nchw, lambda: tf.transpose(images, [0, 2, 3, 1]), lambda: images
        )

        # Random erasing is applied per-sample before batch mixing (tensor domain)
        if random_erasing_prob > 0:
            images = random_erasing(images, seed=seeds[0], p=random_erasing_prob)

        if mixup_alpha > 0 or cutmix_alpha > 0:
            if num_classes is None:
                raise ValueError("`num_classes` must be provided for mixup/cutmix.")
            images, labels = mixup_cutmix(
                images=images,
                labels=labels,
                seed=seeds[1],
                num_classes=num_classes,
                mixup_alpha=mixup_alpha,
                cutmix_alpha=cutmix_alpha,
                prob=prob,
                switch_prob=switch_prob,
                label_smoothing=label_smoothing,
                bce_target=bce_target,
            )

        should_permute = tf.constant(permute_image)
        should_permute = tf.logical_or(should_permute, input_is_nchw)
        images = tf.cond(should_permute, lambda: nhwc_to_nchw(images), lambda: images)

        result = {k: v for k, v in sample.items() if k not in ("image", "label")}
        result["image"] = images
        result["label"] = labels

        return result

    return late_augmentations


def make_postprocessing(
    image_size: int,
    train_image_size: int | None = None,
    is_training: bool = False,
    normalize_image: bool = True,
    normalization_params: tuple | None = (
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225),
    ),
    permute_image: bool = True,
    one_hot_labels: bool = False,
    num_classes: int | None = None,
    label_smoothing: float = 0.0,
    image_keys: list[str] = None,
    label_keys: list[str] = None,
    val_resize_size: int | None | Literal["auto"] = "auto",
):
    if image_keys is None:
        image_keys = ["image", "images", "global_crops", "local_crops"]
    if label_keys is None:
        label_keys = ["label"]

    if val_resize_size == "auto":
        val_resize_buffer = int(image_size / 0.875)
    else:
        val_resize_buffer = val_resize_size

    effective_image_size = (
        train_image_size if (is_training and train_image_size) else image_size
    )

    def postprocessing(sample, num_classes=num_classes):
        sample = resize_and_normalize(
            sample,
            image_keys=image_keys,
            image_size=effective_image_size,
            resize_size=val_resize_buffer if not is_training else None,
            normalize_image=normalize_image,
            normalization_params=normalization_params,
            permute=permute_image,
        )

        res = dict(sample)
        for key in label_keys:
            if key not in sample:
                continue
            label = sample[key]

            if one_hot_labels:
                if num_classes is None:
                    raise ValueError(
                        "`num_classes` must be provided when `one_hot_labels=True`"
                    )
                if label_smoothing > 0:
                    off_value = label_smoothing / float(num_classes)
                    on_value = 1.0 - label_smoothing + off_value
                    label = tf.one_hot(
                        tf.cast(label, tf.int32),
                        num_classes,
                        on_value=on_value,
                        off_value=off_value,
                    )
                else:
                    label = tf.one_hot(tf.cast(label, tf.int32), num_classes)

            res[key] = label

        return res

    return postprocessing

import tensorflow as tf

from justdata.vision.augmentations.registry import (
    get_augment_strategy,
    get_crop_strategy,
)
from justdata.vision.stages import normalize_image_format, resize_and_normalize
from justdata.vision.transforms import (
    normalize,
    pad_to_patch_multiple,
    resize_short_side,
)


def make_preprocessing(image_key="image", mask_key="mask"):
    def preprocessing(sample):
        return normalize_image_format(sample, image_key=image_key)

    return preprocessing


def make_augmentations(
    image_size: int,
    enable: bool = True,
    crop_type: str = "random_resized",
    augment_type: str = "rand_augment",
    ra_kwargs: dict = None,
    ta_kwargs: dict = None,
    padding: int = 4,
    pad_mode: str = "REFLECT",
    mask_fill_value: int = 255,
):
    if not enable:

        def no_aug(sample, seed):
            return sample

        return no_aug

    crop_fn = get_crop_strategy(crop_type)
    aug_fn = get_augment_strategy(augment_type)

    ra_kwargs = dict(ra_kwargs or {})
    ta_kwargs = dict(ta_kwargs or {})

    def augmentations(sample, seed):
        seeds = tf.random.split(seed, 2)

        # Ensure mask has a channel dim for crop ops (crop_to_bounding_box requires rank 3+)
        mask = sample["mask"]
        mask_was_2d = tf.equal(tf.rank(mask), 2)
        mask = tf.cond(mask_was_2d, lambda: tf.expand_dims(mask, -1), lambda: mask)

        # Apply same spatial crop to both image and mask, using nearest interpolation for the mask
        image = crop_fn(
            sample["image"],
            size=image_size,
            seed=seeds[0],
            padding=padding,
            pad_mode=pad_mode,
        )
        mask = crop_fn(
            mask,
            size=image_size,
            seed=seeds[0],
            padding=padding,
            pad_mode=pad_mode,
            interpolation="nearest",
        )

        # Restore original mask rank
        mask = tf.cond(mask_was_2d, lambda: tf.squeeze(mask, -1), lambda: mask)
        sample = sample | {"image": image, "mask": mask}

        # Automatic policies apply geometric operations to image and mask together.
        aug_kwargs_dict = ra_kwargs if augment_type == "rand_augment" else ta_kwargs
        if augment_type in {
            "rand_augment",
            "trivial_augment",
            "trivial_augment_wide",
        }:
            image, mask = aug_fn(
                sample["image"],
                seed=seeds[1],
                segmentation_mask=sample["mask"],
                segmentation_fill_value=mask_fill_value,
                **aug_kwargs_dict,
            )
            return sample | {"image": image, "mask": mask}

        image = aug_fn(sample["image"], seed=seeds[1], **aug_kwargs_dict)

        return sample | {"image": image}

    return augmentations


def make_late_augmentations(**kwargs):
    def late_augmentations(sample, num_classes=None, seed=None):
        return sample

    return late_augmentations


def make_postprocessing(
    image_size: int,
    is_training: bool = False,
    normalize_image: bool = True,
    normalization_params: tuple | None = (
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225),
    ),
    permute_image: bool = True,
    patch_align: bool = False,
    patch_size: int = 14,
    val_resize_size: int | None = None,
):
    """Build postprocessing for segmentation.

    Args:
        patch_align: If ``True``, pad images/masks to the nearest multiple
            of ``patch_size`` instead of resizing to a fixed square.  This
            is the recommended validation strategy for ViT-based dense
            prediction (avoids dropping boundary pixels).
        val_resize_size: When ``patch_align`` is ``True``, resize the
            shorter side to this value before padding.  Defaults to
            ``image_size`` if not set.
    """

    def postprocessing(sample, **kwargs):
        if patch_align and not is_training:
            # Dense ViT evaluation: resize shorter side, pad to patch multiple
            target = val_resize_size if val_resize_size is not None else image_size
            image = tf.cast(sample["image"], tf.float32)
            image = resize_short_side(image, target_size=target, method="bicubic")
            if normalize_image:
                if normalization_params is None:
                    raise ValueError(
                        "`normalization_params` needs to be provided if "
                        "`normalize_image` is True"
                    )
                image = normalize(image, *normalization_params)
            image = pad_to_patch_multiple(image, patch_size=patch_size)
            if permute_image:
                from justdata.vision.transforms import nhwc_to_nchw

                image = nhwc_to_nchw(image)
            sample = sample | {"image": image}

            if "mask" in sample:
                mask = sample["mask"]
                mask = tf.cond(
                    tf.equal(tf.rank(mask), 2),
                    lambda: tf.expand_dims(mask, -1),
                    lambda: mask,
                )
                mask.set_shape([None, None, None])
                mask = resize_short_side(
                    tf.cast(mask, tf.float32),
                    target_size=target,
                    method="nearest",
                )
                mask = pad_to_patch_multiple(mask, patch_size=patch_size)
                mask = tf.squeeze(mask, -1)
                sample = sample | {"mask": mask}

            return sample

        # Standard fixed-size postprocessing
        sample = resize_and_normalize(
            sample,
            image_keys=["image"],
            image_size=image_size,
            resize_size=None,
            normalize_image=normalize_image,
            normalization_params=normalization_params,
            permute=permute_image,
        )

        if "mask" in sample:
            mask = sample["mask"]
            mask = tf.cond(
                tf.equal(tf.rank(mask), 2),
                lambda: tf.expand_dims(mask, -1),
                lambda: mask,
            )
            mask.set_shape([None, None, None])
            mask = tf.image.resize(mask, [image_size, image_size], method="nearest")
            mask = tf.squeeze(mask, -1)
            sample = sample | {"mask": mask}

        return sample

    return postprocessing

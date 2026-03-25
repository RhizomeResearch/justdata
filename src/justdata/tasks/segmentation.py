import tensorflow as tf

from justdata.augmentations.registry import get_augment_strategy, get_crop_strategy
from justdata.stages import co_transform, normalize_image_format, resize_and_normalize


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
):
    if not enable:

        def no_aug(sample, seed):
            return sample

        return no_aug

    crop_fn = get_crop_strategy(crop_type)
    aug_fn = get_augment_strategy(augment_type)

    ra_kwargs = ra_kwargs or {}
    ta_kwargs = ta_kwargs or {}

    def augmentations(sample, seed):
        seeds = tf.random.split(seed, 2)

        # Co-transform: same spatial crop applied to both image and mask
        sample = co_transform(
            sample,
            image_key="image",
            mask_key="mask",
            transform_fn=lambda img, s: crop_fn(
                img, size=image_size, seed=s, padding=padding, pad_mode=pad_mode
            ),
            seed=seeds[0],
        )

        # Color augmentation on image only (not mask)
        aug_kwargs_dict = ra_kwargs if augment_type == "rand_augment" else ta_kwargs
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
):
    def postprocessing(sample):
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
            mask = tf.image.resize(mask, [image_size, image_size], method="nearest")
            mask = tf.cond(
                tf.equal(tf.shape(mask)[-1], 1),
                lambda: tf.squeeze(mask, -1),
                lambda: mask,
            )
            sample = sample | {"mask": mask}

        return sample

    return postprocessing

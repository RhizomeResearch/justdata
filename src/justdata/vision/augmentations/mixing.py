from typing import Tuple

import tensorflow as tf

from justdata.core.label_mixing import (
    LabelMixMode,
    blend_prepared_labels,
    prepare_labels_for_mixing,
)


@tf.function
def mixup_cutmix(
    images: tf.Tensor,
    labels: tf.Tensor,
    seed: tf.Tensor,
    num_classes: int,
    mixup_alpha: float = 0.8,
    cutmix_alpha: float = 1.0,
    prob: float = 1.0,
    switch_prob: float = 0.5,
    label_smoothing: float = 0.1,
    bce_target: bool = False,
    label_mode: LabelMixMode = "single_label",
) -> Tuple[tf.Tensor, tf.Tensor]:
    if mixup_alpha > 0 and cutmix_alpha == 0:
        switch_prob = -1.0
    elif mixup_alpha == 0 and cutmix_alpha > 0:
        switch_prob = 1.0

    batch_size = tf.shape(images)[0]
    seeds = tf.random.split(seed, 5)

    augment_cond = tf.less(
        tf.random.stateless_uniform([], seed=seeds[0], minval=0.0, maxval=1.0), prob
    )

    random_vals = tf.random.stateless_uniform([batch_size], seed=seeds[1])
    shuffled_indices = tf.argsort(random_vals)

    def gather_shuffled(tensor):
        return tf.gather(tensor, shuffled_indices)

    def _sample_from_beta(alpha, shape, s):
        splits = tf.unstack(tf.random.split(s, 2))
        s1, s2 = splits[0], splits[1]
        sample_a = tf.random.stateless_gamma(shape, alpha=alpha, seed=s1)
        sample_b = tf.random.stateless_gamma(shape, alpha=alpha, seed=s2)
        return sample_a / (sample_a + sample_b + 1e-8)

    def _prepare_labels(lbls):
        return prepare_labels_for_mixing(
            lbls,
            label_mode=label_mode,
            num_classes=num_classes,
            label_smoothing=label_smoothing if label_mode == "single_label" else 0.0,
        )

    def _update_labels(imgs, lbls, lam):
        labels_1 = _prepare_labels(lbls)
        labels_2 = _prepare_labels(gather_shuffled(lbls))
        lam = tf.reshape(lam, [-1, 1])
        new_labels = blend_prepared_labels(labels_1, labels_2, lam, bce_target=bce_target)
        return imgs, new_labels

    def _mixup(imgs, lbls):
        lam = _sample_from_beta(mixup_alpha, [batch_size], seeds[2])
        lam = tf.reshape(lam, [-1, 1, 1, 1])
        imgs_f = tf.cast(imgs, tf.float32)
        shuffled_f = tf.cast(gather_shuffled(imgs), tf.float32)
        mixed_imgs = lam * imgs_f + (1.0 - lam) * shuffled_f
        mixed_imgs = tf.cast(mixed_imgs, imgs.dtype)
        return _update_labels(mixed_imgs, lbls, tf.reshape(lam, [batch_size]))

    def _cutmix(imgs, lbls):
        lam = _sample_from_beta(cutmix_alpha, [batch_size], seeds[2])
        image_height = tf.cast(tf.shape(imgs)[1], tf.float32)
        image_width = tf.cast(tf.shape(imgs)[2], tf.float32)

        ratio = tf.math.sqrt(1 - lam)
        cut_height = tf.cast(ratio * image_height, tf.int32)
        cut_width = tf.cast(ratio * image_width, tf.int32)

        s_loc = tf.random.split(seeds[3], 2)

        center_y = tf.random.stateless_uniform(
            [batch_size],
            minval=0,
            maxval=tf.cast(image_height, tf.int32),
            dtype=tf.int32,
            seed=s_loc[0],
        )
        center_x = tf.random.stateless_uniform(
            [batch_size],
            minval=0,
            maxval=tf.cast(image_width, tf.int32),
            dtype=tf.int32,
            seed=s_loc[1],
        )

        min_y = tf.clip_by_value(
            center_y - cut_height // 2, 0, tf.cast(image_height, tf.int32)
        )
        min_x = tf.clip_by_value(
            center_x - cut_width // 2, 0, tf.cast(image_width, tf.int32)
        )
        max_y = tf.clip_by_value(
            center_y + cut_height // 2, 0, tf.cast(image_height, tf.int32)
        )
        max_x = tf.clip_by_value(
            center_x + cut_width // 2, 0, tf.cast(image_width, tf.int32)
        )

        y_coords = tf.range(tf.cast(image_height, tf.int32))[tf.newaxis, :, tf.newaxis]
        x_coords = tf.range(tf.cast(image_width, tf.int32))[tf.newaxis, tf.newaxis, :]

        mask = tf.cast(
            ~(
                (y_coords >= min_y[:, tf.newaxis, tf.newaxis])
                & (y_coords < max_y[:, tf.newaxis, tf.newaxis])
                & (x_coords >= min_x[:, tf.newaxis, tf.newaxis])
                & (x_coords < max_x[:, tf.newaxis, tf.newaxis])
            ),
            dtype=tf.float32,
        )

        mask = mask[..., tf.newaxis]
        imgs_f = tf.cast(imgs, tf.float32)
        shuffled_f = tf.cast(gather_shuffled(imgs), tf.float32)
        mixed_imgs = mask * imgs_f + (1.0 - mask) * shuffled_f
        mixed_imgs = tf.cast(mixed_imgs, imgs.dtype)

        bbox_area = tf.cast((max_y - min_y) * (max_x - min_x), tf.float32)
        lam_adjusted = 1.0 - bbox_area / (image_height * image_width)

        return _update_labels(mixed_imgs, lbls, lam_adjusted)

    def apply_augment():
        if mixup_alpha > 0 and cutmix_alpha > 0:
            is_cutmix = (
                tf.random.stateless_uniform([], seed=seeds[4], maxval=1.0) < switch_prob
            )
            return tf.cond(
                is_cutmix,
                lambda: _cutmix(images, labels),
                lambda: _mixup(images, labels),
            )
        elif mixup_alpha > 0:
            return _mixup(images, labels)
        else:
            return _cutmix(images, labels)

    return tf.cond(
        augment_cond,
        apply_augment,
        lambda: (images, _prepare_labels(labels)),
    )


@tf.function
def random_erasing(
    images: tf.Tensor,
    seed: tf.Tensor,
    p: float = 0.25,
    scale: Tuple[float, float] = (0.02, 0.33),
    ratio: Tuple[float, float] = (0.3, 3.3),
    replace: float = 0.0,
) -> tf.Tensor:
    batch_size = tf.shape(images)[0]
    seeds = tf.random.split(seed, batch_size)
    seeds = tf.stack(seeds)

    def apply_erasing(args):
        img, s = args
        s_parts = tf.random.split(s, 5)
        apply_cond = tf.random.stateless_uniform([], seed=s_parts[0]) < p

        def do_erase():
            height = tf.cast(tf.shape(img)[0], tf.float32)
            width = tf.cast(tf.shape(img)[1], tf.float32)
            area = height * width

            target_area = (
                tf.random.stateless_uniform(
                    [], minval=scale[0], maxval=scale[1], seed=s_parts[1]
                )
                * area
            )
            aspect_ratio = tf.random.stateless_uniform(
                [], minval=ratio[0], maxval=ratio[1], seed=s_parts[2]
            )

            h = tf.cast(
                tf.math.round(tf.math.sqrt(target_area * aspect_ratio)), tf.int32
            )
            w = tf.cast(
                tf.math.round(tf.math.sqrt(target_area / aspect_ratio)), tf.int32
            )

            h = tf.clip_by_value(h, 1, tf.cast(height, tf.int32))
            w = tf.clip_by_value(w, 1, tf.cast(width, tf.int32))

            y = tf.random.stateless_uniform(
                [],
                minval=0,
                maxval=tf.cast(height, tf.int32) - h + 1,
                dtype=tf.int32,
                seed=s_parts[3],
            )
            x = tf.random.stateless_uniform(
                [],
                minval=0,
                maxval=tf.cast(width, tf.int32) - w + 1,
                dtype=tf.int32,
                seed=s_parts[4],
            )

            mask = tf.ones((h, w, tf.shape(img)[-1]), dtype=img.dtype)
            paddings = [
                [y, tf.cast(height, tf.int32) - (y + h)],
                [x, tf.cast(width, tf.int32) - (x + w)],
                [0, 0],
            ]
            mask = tf.pad(mask, paddings, constant_values=0)

            return tf.where(mask == 1, tf.cast(replace, img.dtype), img)

        return tf.cond(apply_cond, do_erase, lambda: img)

    return tf.map_fn(apply_erasing, (images, seeds), fn_output_signature=images.dtype)

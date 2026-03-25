from typing import List, Optional, Tuple, Union

import tensorflow as tf

from justdata.augmentations.registry import register_augment_strategy
from justdata.utils import (
    _autocontrast,
    _brightness,
    _color,
    _contrast,
    _cutout,
    _equalize,
    _grayscale,
    _invert,
    _posterize,
    _randomly_negate_tensor,
    _rotate_with_bboxes,
    _sharpness,
    _shear_with_bboxes,
    _shear_x,
    _shear_y,
    _solarize_add,
    _solarize_val,
    _translate_bbox,
    _translate_x,
    _translate_y,
    _wrapped_rotate,
)


@tf.function
def rand_augment(
    image: tf.Tensor,
    seed: tf.Tensor,
    bboxes: Optional[tf.Tensor] = None,
    num_layers: int = 2,
    magnitude: float = 10.0,
    cutout_const: float = 40.0,
    translate_const: float = 100.0,
    magnitude_std: float = 0.0,
    prob_to_apply: Optional[float] = None,
    exclude_ops: Optional[List[str]] = None,
) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
    input_image_type = image.dtype
    if input_image_type != tf.uint8:
        image = tf.clip_by_value(image, 0.0, 255.0)
        image = tf.cast(image, dtype=tf.uint8)

    replace_value = 128

    def _rotate_level_to_arg(level, seed):
        level = (level / 10) * 30.0
        level = _randomly_negate_tensor(level, seed)
        return (level,)

    def _enhance_level_to_arg(level, seed):
        return ((level / 10) * 1.8 + 0.1,)

    def _shear_level_to_arg(level, seed):
        level = (level / 10) * 0.3
        level = _randomly_negate_tensor(level, seed)
        return (level,)

    def _translate_level_to_arg(level, translate_const, seed):
        level = (level / 10) * float(translate_const)
        level = _randomly_negate_tensor(level, seed)
        return (level,)

    def _mult_to_arg(level, multiplier=1.0):
        return (tf.cast((level / 10) * multiplier, tf.int32),)

    def level_to_arg(name, level, seed):
        if name in ["AutoContrast", "Equalize", "Invert", "Grayscale"]:
            return ()
        if name == "Posterize":
            num_bits = tf.cast(8 - (level / 10) * 4, tf.int32)
            return (num_bits,)
        if name == "Solarize":
            threshold = tf.cast(256 * (1.0 - level / 10), tf.int32)
            return (threshold,)
        if name == "SolarizeAdd":
            return _mult_to_arg(level, 110)
        if name == "Cutout":
            return _mult_to_arg(level, cutout_const)
        if name in ["Color", "Contrast", "Brightness", "Sharpness"]:
            return _enhance_level_to_arg(level, seed)
        if name in ["ShearX", "ShearY", "ShearX_BBox", "ShearY_BBox"]:
            return _shear_level_to_arg(level, seed)
        if name in [
            "TranslateX",
            "TranslateY",
            "TranslateX_BBox",
            "TranslateY_BBox",
        ]:
            return _translate_level_to_arg(level, translate_const, seed)
        if name in ["Rotate", "Rotate_BBox"]:
            return _rotate_level_to_arg(level, seed)
        return ()

    def wrap_func(func, name):
        needs_bbox = "BBox" in name
        needs_seed = name == "Cutout"
        is_replace = name in [
            "Rotate",
            "TranslateX",
            "ShearX",
            "ShearY",
            "TranslateY",
            "Cutout",
            "Rotate_BBox",
            "ShearX_BBox",
            "ShearY_BBox",
            "TranslateX_BBox",
            "TranslateY_BBox",
        ]

        def wrapped(img, boxes, args, seed):
            call_args = [img]
            if needs_bbox:
                call_args.append(boxes)
            call_args.extend(list(args))
            if is_replace:
                call_args.append(replace_value)
            if needs_seed:
                call_args.append(seed)

            res = func(*call_args)
            if needs_bbox:
                return res
            return res, boxes

        return wrapped

    NAME_TO_FUNC = {
        "AutoContrast": _autocontrast,
        "Equalize": _equalize,
        "Invert": _invert,
        "Rotate": _wrapped_rotate,
        "Posterize": _posterize,
        "Solarize": _solarize_val,
        "SolarizeAdd": _solarize_add,
        "Color": _color,
        "Contrast": _contrast,
        "Brightness": _brightness,
        "Sharpness": _sharpness,
        "ShearX": _shear_x,
        "ShearY": _shear_y,
        "TranslateX": _translate_x,
        "TranslateY": _translate_y,
        "Cutout": _cutout,
        "Grayscale": _grayscale,
        "Rotate_BBox": _rotate_with_bboxes,
        "ShearX_BBox": lambda i, b, lvl, r: _shear_with_bboxes(
            i, b, lvl, r, shear_horizontal=True
        ),
        "ShearY_BBox": lambda i, b, lvl, r: _shear_with_bboxes(
            i, b, lvl, r, shear_horizontal=False
        ),
        "TranslateX_BBox": lambda i, b, p, r: _translate_bbox(
            i, b, p, r, shift_horizontal=True
        ),
        "TranslateY_BBox": lambda i, b, p, r: _translate_bbox(
            i, b, p, r, shift_horizontal=False
        ),
    }

    available_ops = [
        "AutoContrast",
        "Equalize",
        "Invert",
        "Rotate",
        "Posterize",
        "Solarize",
        "Color",
        "Contrast",
        "Brightness",
        "Sharpness",
        "ShearX",
        "ShearY",
        "TranslateX",
        "TranslateY",
        "Cutout",
        "SolarizeAdd",
    ]

    if bboxes is not None:
        box_aware_ops = {
            "Rotate": "Rotate_BBox",
            "ShearX": "ShearX_BBox",
            "ShearY": "ShearY_BBox",
            "TranslateX": "TranslateX_BBox",
            "TranslateY": "TranslateY_BBox",
        }
        available_ops = [box_aware_ops.get(op, op) for op in available_ops]

    if exclude_ops:
        available_ops = [op for op in available_ops if op not in exclude_ops]

    aug_image = image
    aug_bboxes = bboxes
    seed_layers = tf.random.split(seed, num_layers)

    for i in range(num_layers):
        layer_seed = seed_layers[i]
        seeds = tf.random.split(layer_seed, 4)

        op_idx = tf.random.stateless_uniform(
            [], maxval=len(available_ops), dtype=tf.int32, seed=seeds[0]
        )

        should_apply = tf.constant(True)
        if prob_to_apply is not None:
            should_apply = (
                tf.random.stateless_uniform([], seed=seeds[1]) < prob_to_apply
            )

        level = magnitude
        if magnitude_std > 0:
            level += tf.random.stateless_normal([], seed=seeds[2]) * magnitude_std
        level = tf.clip_by_value(level, 0.0, 10)

        branch_fns = []
        op_seeds = tf.random.split(seeds[3], len(available_ops))
        for j, op_name in enumerate(available_ops):
            func = NAME_TO_FUNC[op_name]
            wrapped_fn = wrap_func(func, op_name)

            def create_branch_fn(f, seed_val, name, im=aug_image, bx=aug_bboxes):
                def branch_fn():
                    ar = level_to_arg(name, level, seed_val)
                    return f(im, bx, ar, seed_val)

                return branch_fn

            branch_fns.append((j, create_branch_fn(wrapped_fn, op_seeds[j], op_name)))

        def apply_op():
            return tf.switch_case(
                branch_index=op_idx,
                branch_fns=branch_fns,
                default=lambda: (aug_image, aug_bboxes),
            )

        aug_image, aug_bboxes = tf.cond(
            should_apply, apply_op, lambda: (aug_image, aug_bboxes)
        )

    aug_image = tf.cast(aug_image, dtype=input_image_type)

    if bboxes is None:
        return aug_image
    return aug_image, aug_bboxes


@tf.function
def trivial_augment(
    image: tf.Tensor,
    seed: tf.Tensor,
    bboxes: Optional[tf.Tensor] = None,
    cutout_const: float = 40.0,
    translate_const: float = 100.0,
    exclude_ops: Optional[List[str]] = None,
) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
    seeds = tf.random.split(seed, 2)
    magnitude = tf.random.stateless_uniform([], minval=0.0, maxval=10.0, seed=seeds[0])
    return rand_augment(
        image=image,
        seed=seeds[1],
        bboxes=bboxes,
        num_layers=1,
        magnitude=magnitude,
        cutout_const=cutout_const,
        translate_const=translate_const,
        exclude_ops=exclude_ops,
    )


@register_augment_strategy("rand_augment")
def _aug_rand_augment(image, seed, **kwargs):
    return rand_augment(image, seed, **kwargs)


@register_augment_strategy("trivial_augment")
def _aug_trivial_augment(image, seed, **kwargs):
    return trivial_augment(image, seed, **kwargs)


@register_augment_strategy("none")
def _aug_none(image, seed, **kwargs):
    return image

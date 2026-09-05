from typing import List, Optional, Tuple, Union

import tensorflow as tf

from justdata.vision.augmentations.registry import register_augment_strategy
from justdata.vision.utils import (
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
    _rotate,
    _rotate_with_bboxes,
    _sharpness,
    _shear_with_bboxes,
    _shear_x,
    _shear_y,
    _solarize_add,
    _solarize_val,
    _translate_bbox,
    _transform,
    _translate,
    _translate_x,
    _translate_y,
)

# 31-bin scale: magnitude m ∈ {0, …, 30}, B = max index = 30
_B = 30.0

# Ops outside the strict 14-op RA space; not in the default pool for any algorithm
_NON_RA_OPS = ["Invert", "Cutout", "SolarizeAdd", "Grayscale"]

RAND_AUGMENT_OPS = (
    "Identity",
    "AutoContrast",
    "Equalize",
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
)
RAND_AUGMENT_SPATIAL_OPS = (
    "Rotate",
    "ShearX",
    "ShearY",
    "TranslateX",
    "TranslateY",
)


def _identity(image):
    return image


def _apply_segmentation_geometric_op(
    image,
    mask,
    name,
    args,
    image_replace,
    mask_replace,
):
    def apply(value, interpolation, replace):
        if name == "Rotate":
            return _rotate(
                value,
                args[0],
                replace=replace,
                interpolation=interpolation,
            )
        if name in ("TranslateX", "TranslateY"):
            translations = [-args[0], 0] if name == "TranslateX" else [0, -args[0]]
            return _translate(
                value,
                translations,
                replace=replace,
                interpolation=interpolation,
            )
        if name in ("ShearX", "ShearY"):
            transforms = (
                [1.0, args[0], 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
                if name == "ShearX"
                else [1.0, 0.0, 0.0, args[0], 1.0, 0.0, 0.0, 0.0]
            )
            return _transform(
                value,
                transforms=transforms,
                interpolation=interpolation,
                fill_mode="CONSTANT",
                fill_value=tf.cast(replace, tf.float32),
            )
        raise ValueError(f"Unsupported segmentation geometric operation: {name}")

    return (
        apply(image, interpolation="BILINEAR", replace=image_replace),
        apply(mask, interpolation="NEAREST", replace=mask_replace),
    )


@tf.function
def rand_augment(
    image: tf.Tensor,
    seed: tf.Tensor,
    bboxes: Optional[tf.Tensor] = None,
    num_layers: int = 2,
    magnitude: float = 9.0,
    cutout_const: float = 40.0,
    translate_const: float = 100.0,
    magnitude_std: float = 0.0,
    prob_to_apply: Optional[float] = None,
    exclude_ops: Optional[List[str]] = None,
    rotate_max: float = 30.0,
    shear_max: float = 0.3,
    enhance_max: float = 0.9,
    posterize_max_bits: int = 4,
    segmentation_mask: Optional[tf.Tensor] = None,
    segmentation_fill_value: int = 255,
) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
    """RandAugment on the 31-bin (m ∈ {0…30}, B=30) scale.

    Physical mappings per the spec:
    - Rotate:    (m/B) * rotate_max * s          default ±30°
    - Translate: (m/B) * translate_const * s
    - Shear:     (m/B) * shear_max * s            default ±0.3
    - Enhance:   1 + (m/B) * enhance_max * s      default MaxDelta=0.9
    - Posterize: 8 - round(m * posterize_max_bits / B)  default min 4 bits
    - Solarize:  255 * (1 - m/B)

    When ``segmentation_mask`` is provided, geometric operations reuse the
    image's sampled parameters with bilinear image interpolation and nearest
    mask interpolation. Vacated mask pixels use ``segmentation_fill_value``.
    """
    if bboxes is not None and segmentation_mask is not None:
        raise ValueError("RandAugment cannot transform bboxes and a mask together")

    input_image_type = image.dtype
    if input_image_type != tf.uint8:
        image = tf.clip_by_value(image, 0.0, 255.0)
        image = tf.cast(image, dtype=tf.uint8)

    replace_value = 128

    def _rotate_level_to_arg(level, seed):
        level = (level / _B) * rotate_max
        level = _randomly_negate_tensor(level, seed)
        return (level,)

    def _enhance_level_to_arg(level, seed):
        # v' = 1.0 + (m/B) * MaxDelta * s,  s ∈ {-1, +1}
        delta = (level / _B) * enhance_max
        delta = _randomly_negate_tensor(delta, seed)
        return (1.0 + delta,)

    def _shear_level_to_arg(level, seed):
        level = (level / _B) * shear_max
        level = _randomly_negate_tensor(level, seed)
        return (level,)

    def _translate_level_to_arg(level, t_const, seed):
        level = (level / _B) * t_const
        level = _randomly_negate_tensor(level, seed)
        return (level,)

    def _mult_to_arg(level, multiplier=1.0):
        return (tf.cast((level / _B) * multiplier, tf.int32),)

    def level_to_arg(name, level, seed):
        if name in ["Identity", "AutoContrast", "Equalize", "Invert", "Grayscale"]:
            return ()
        if name == "Posterize":
            # 8 - round(m / (B / MaxBits)) = 8 - round(m * MaxBits / B)
            num_bits = 8 - tf.cast(
                tf.round((level / _B) * float(posterize_max_bits)), tf.int32
            )
            return (num_bits,)
        if name == "Solarize":
            # Inverts pixels >= threshold; threshold decreases with magnitude
            threshold = tf.cast(255.0 * (1.0 - level / _B), tf.int32)
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
            if segmentation_mask is not None and name in RAND_AUGMENT_SPATIAL_OPS:
                return _apply_segmentation_geometric_op(
                    img,
                    boxes,
                    name,
                    args,
                    replace_value,
                    segmentation_fill_value,
                )

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
        "Identity": _identity,
        "AutoContrast": _autocontrast,
        "Equalize": _equalize,
        "Invert": _invert,
        "Rotate": _rotate,
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

    # Strict 14-op RA space: 3 magnitude-independent + 11 magnitude-dependent
    available_ops = list(RAND_AUGMENT_OPS)

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
    if not available_ops:
        raise ValueError("RandAugment requires at least one non-excluded operation")

    aug_image = image
    aug_bboxes = segmentation_mask if segmentation_mask is not None else bboxes
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
        level = tf.clip_by_value(level, 0.0, _B)

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

    if bboxes is None and segmentation_mask is None:
        return aug_image
    return aug_image, aug_bboxes


@tf.function
def trivial_augment(
    image: tf.Tensor,
    seed: tf.Tensor,
    bboxes: Optional[tf.Tensor] = None,
    translate_const: Optional[float] = None,
    exclude_ops: Optional[List[str]] = None,
    segmentation_mask: Optional[tf.Tensor] = None,
    segmentation_fill_value: int = 255,
) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
    """TrivialAugment: one op from the strict 14-op RA space, m ~ U{0,…,30}.

    Standard bounds. translate_const defaults to (150/331)*image_width if not
    provided, matching the proportional RA-space definition. Segmentation
    masks follow the paired geometric behavior documented by ``rand_augment``.
    """
    seeds = tf.random.split(seed, 2)
    # Discrete uniform magnitude: m ~ U{0, 1, …, 30}
    magnitude = tf.cast(
        tf.random.stateless_uniform(
            [], minval=0, maxval=31, dtype=tf.int32, seed=seeds[0]
        ),
        tf.float32,
    )

    # Standard translate bound: proportional to image width
    if translate_const is None:
        translate_const = tf.cast(tf.shape(image)[1], tf.float32) * (150.0 / 331.0)

    # Strict 14-op RA space: exclude ops not in the pool
    ta_exclude = list(_NON_RA_OPS)
    if exclude_ops:
        ta_exclude = ta_exclude + [op for op in exclude_ops if op not in ta_exclude]

    return rand_augment(
        image=image,
        seed=seeds[1],
        bboxes=bboxes,
        num_layers=1,
        magnitude=magnitude,
        translate_const=translate_const,
        exclude_ops=ta_exclude,
        segmentation_mask=segmentation_mask,
        segmentation_fill_value=segmentation_fill_value,
    )


@tf.function
def trivial_augment_wide(
    image: tf.Tensor,
    seed: tf.Tensor,
    bboxes: Optional[tf.Tensor] = None,
    exclude_ops: Optional[List[str]] = None,
    segmentation_mask: Optional[tf.Tensor] = None,
    segmentation_fill_value: int = 255,
) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
    """TrivialAugmentWide: one op from the strict 14-op RA space, m ~ U{0,…,30}.

    Wide bounds per the spec:
    - Rotate:    ±135°   (vs standard ±30°)
    - Translate: ±32 px fixed  (vs proportional; note: standard may exceed wide
                               for images wider than ~71 px)
    - Shear:     ±0.99   (vs standard ±0.3)
    - Enhance:   MaxDelta=0.99  (vs standard 0.9)
    - Posterize: min 2 bits  (vs standard min 4 bits)

    Segmentation masks follow the paired geometric behavior documented by
    ``rand_augment``.
    """
    seeds = tf.random.split(seed, 2)
    # Discrete uniform magnitude: m ~ U{0, 1, …, 30}
    magnitude = tf.cast(
        tf.random.stateless_uniform(
            [], minval=0, maxval=31, dtype=tf.int32, seed=seeds[0]
        ),
        tf.float32,
    )

    ta_exclude = list(_NON_RA_OPS)
    if exclude_ops:
        ta_exclude = ta_exclude + [op for op in exclude_ops if op not in ta_exclude]

    return rand_augment(
        image=image,
        seed=seeds[1],
        bboxes=bboxes,
        num_layers=1,
        magnitude=magnitude,
        translate_const=32.0,  # Wide: fixed 32 px (not image-proportional)
        rotate_max=135.0,  # Wide: ±135°
        shear_max=0.99,  # Wide: ±0.99
        enhance_max=0.99,  # Wide: MaxDelta=0.99
        posterize_max_bits=6,  # Wide: min 2 bits kept (8 − 6 = 2)
        exclude_ops=ta_exclude,
        segmentation_mask=segmentation_mask,
        segmentation_fill_value=segmentation_fill_value,
    )


@register_augment_strategy("rand_augment")
def _aug_rand_augment(image, seed, **kwargs):
    return rand_augment(image, seed, **kwargs)


@register_augment_strategy("trivial_augment")
def _aug_trivial_augment(image, seed, **kwargs):
    return trivial_augment(image, seed, **kwargs)


@register_augment_strategy("trivial_augment_wide")
def _aug_trivial_augment_wide(image, seed, **kwargs):
    return trivial_augment_wide(image, seed, **kwargs)


@register_augment_strategy("none")
def _aug_none(image, seed, **kwargs):
    return image

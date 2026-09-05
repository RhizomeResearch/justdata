from collections.abc import Mapping

import tensorflow as tf

from justdata.acoustic._random import _seed_tensor as _normalize_seed
from justdata.acoustic.registry import (
    list_audio_spectrogram_augments,
    list_audio_waveform_augments,
    register_audio_batch_augment,
    register_audio_corruption,
    register_audio_decoder,
    register_audio_postprocessor,
    register_audio_resampler,
    register_audio_spectrogram_augment,
    register_audio_waveform_augment,
)
from justdata.acoustic.configs import (
    AudioPreprocessConfig,
    LabelTransformConfig,
    SegmentStrategyConfig,
)
from justdata.acoustic.labels import transform_label
from justdata.acoustic.postprocessing import make_model_input_stage, preset_info
from justdata.acoustic.preprocessing import make_preprocessing as _make_preprocessing
from justdata.acoustic.schema import FEATURES, LABEL, METADATA
from justdata.acoustic.stages import make_segment_stage


def _identity_sample(sample, *args, **kwargs):
    return sample


def _identity_batch(batch, num_classes=None, seed=None, **kwargs):
    return batch


def _seed_tensor(seed):
    return None if seed is None else _normalize_seed(seed)


@register_audio_decoder("identity")
@register_audio_resampler("identity")
@register_audio_waveform_augment("none")
@register_audio_spectrogram_augment("none")
@register_audio_batch_augment("none")
@register_audio_postprocessor("identity")
@register_audio_corruption("identity")
def identity(sample, *args, **kwargs):
    return sample


def make_preprocessing(config: AudioPreprocessConfig | dict | None = None, **kwargs):
    if config is None and not kwargs:
        return _identity_sample
    if config is None:
        config = AudioPreprocessConfig(**kwargs)
    else:
        config = AudioPreprocessConfig.from_dict(config)
    return _make_preprocessing(config)


def make_augmentations(
    segment_config: SegmentStrategyConfig | dict | None = None,
    waveform_augmentations=None,
    spectrogram_augmentations=None,
    train_augment: dict | None = None,
    is_training: bool = True,
    augment_eval: bool = False,
    spectrogram_key: str = FEATURES,
    spectrogram_layout: str | None = None,
    **kwargs,
):
    import justdata.acoustic.augment  # noqa: F401
    from justdata.acoustic.augment import make_waveform_augmentation_stage

    if spectrogram_augmentations is not None:
        raise ValueError(
            "spectrogram_augmentations require post-frontend features; "
            "pass them to make_late_augmentations instead"
        )

    if waveform_augmentations is None and train_augment:
        if "waveform" in train_augment:
            waveform_augmentations = train_augment["waveform"]
        else:
            known_waveform_augments = set(list_audio_waveform_augments()) - {"none"}
            waveform_augmentations = {
                key: value
                for key, value in train_augment.items()
                if key in known_waveform_augments
            }

    stages = []
    if segment_config is not None:
        stages.append(make_segment_stage(segment_config, is_training=is_training))
    if waveform_augmentations:
        stages.append(
            make_waveform_augmentation_stage(
                waveform_augmentations,
                is_training=is_training,
                augment_eval=augment_eval,
            )
        )
    if not stages:
        return _identity_sample

    def augment(sample, seed=None):
        result = sample
        if seed is None:
            stage_seeds = [None] * len(stages)
        else:
            stage_seeds = tf.unstack(tf.random.split(_seed_tensor(seed), len(stages)))
        for stage, stage_seed in zip(stages, stage_seeds):
            result = stage(result, seed=stage_seed)
        return result

    return augment


def make_late_augmentations(**kwargs):
    import justdata.acoustic.augment  # noqa: F401
    from justdata.acoustic.augment import (
        make_batch_augmentation_stage,
        make_spectrogram_augmentation_stage,
        normalize_spectrogram_augment_specs,
    )

    known_batch_augments = {
        "batch_mixstyle",
        "cutmix",
        "cutmix_spec",
        "mixstyle",
        "mixup",
        "wavmix",
    }
    batch_augmentations = kwargs.pop("batch_augmentations", None)
    train_augment = kwargs.pop("train_augment", None)
    is_training = kwargs.pop("is_training", True)
    augment_eval = kwargs.pop("augment_eval", False)
    input_key = kwargs.pop("input_key", None)
    input_kind = kwargs.pop("input_kind", None)
    label_mode = kwargs.pop("label_mode", None)
    label_transform = kwargs.pop("label_transform", None)
    spectrogram_augmentations = kwargs.pop("spectrogram_augmentations", None)
    spectrogram_key = kwargs.pop("spectrogram_key", FEATURES)
    spectrogram_layout = kwargs.pop("spectrogram_layout", None)
    model_layout = kwargs.pop("model_layout", None)

    if spectrogram_augmentations is None and train_augment:
        if "spectrogram" in train_augment:
            spectrogram_augmentations = train_augment["spectrogram"]
        else:
            known_spectrogram_augments = set(list_audio_spectrogram_augments()) - {
                "none"
            }
            spectrogram_augmentations = {
                ("passt_patchout" if key == "patchout" else key): value
                for key, value in train_augment.items()
                if (
                    key in known_spectrogram_augments
                    and key not in known_batch_augments
                )
                or key == "patchout"
            }

    if batch_augmentations is None and train_augment:
        if "batch" in train_augment:
            batch_augmentations = train_augment["batch"]
        else:
            batch_augmentations = {
                key: value
                for key, value in train_augment.items()
                if key in known_batch_augments
            }

    if batch_augmentations is None:
        batch_augmentations = {
            key: kwargs.pop(key) for key in tuple(kwargs) if key in known_batch_augments
        }

    spectrogram_specs = normalize_spectrogram_augment_specs(spectrogram_augmentations)
    if not spectrogram_specs and not batch_augmentations:
        return _identity_batch

    sample_layouts = {"tf", "tfc", "cft"}
    batch_to_sample_layout = {"btf": "tf", "btfc": "tfc", "bcft": "cft"}
    layout_source = spectrogram_layout or model_layout
    if not spectrogram_specs:
        sample_spectrogram_layout = None
    elif layout_source in sample_layouts or layout_source is None:
        sample_spectrogram_layout = layout_source
    elif layout_source in batch_to_sample_layout:
        sample_spectrogram_layout = batch_to_sample_layout[layout_source]
    else:
        raise ValueError(
            "Spectrogram augmentation layout must be one of 'tf', 'tfc', "
            "'cft', 'btf', 'btfc', or 'bcft'"
        )

    batch_spectrogram_layout = model_layout or spectrogram_layout

    label_config = (
        LabelTransformConfig.from_dict(label_transform)
        if label_transform is not None
        else None
    )
    label_transform_dict = label_config.to_dict() if label_config is not None else None
    spectrogram_stage = make_spectrogram_augmentation_stage(
        spectrogram_specs,
        is_training=is_training,
        augment_eval=augment_eval,
        feature_key=spectrogram_key,
        layout=sample_spectrogram_layout,
    )
    batch_stage = make_batch_augmentation_stage(
        batch_augmentations,
        is_training=is_training,
        augment_eval=augment_eval,
        input_key=input_key,
        input_kind=input_kind,
        label_mode=label_mode,
        label_transform=label_transform_dict,
        spectrogram_layout=batch_spectrogram_layout,
    )

    def mapped_output_signature(batch):
        signature = tf.nest.map_structure(
            lambda value: tf.TensorSpec(value.shape[1:], value.dtype), batch
        )
        feature = batch[spectrogram_key]
        signature[spectrogram_key] = tf.TensorSpec(
            [None] * (feature.shape.rank - 1), feature.dtype
        )

        def patchout_debug_enabled(spec):
            config = spec.get("config")
            config_debug = (
                config.get("debug", False) if isinstance(config, Mapping) else False
            )
            return spec["name"] == "passt_patchout" and bool(
                spec.get("debug", config_debug)
            )

        patchout_debug = any(map(patchout_debug_enabled, spectrogram_specs))
        if patchout_debug:
            metadata_signature = dict(signature.get(METADATA, {}))
            metadata_signature["patchout"] = {
                "structured_frequency": tf.TensorSpec([None], tf.int32),
                "structured_time": tf.TensorSpec([None], tf.int32),
                "unstructured": tf.TensorSpec([None], tf.int32),
            }
            signature[METADATA] = metadata_signature
        return signature

    def late_augmentations(batch, num_classes=None, seed=None, **_kwargs):
        effective_num_classes = num_classes
        if effective_num_classes is None and label_config is not None:
            effective_num_classes = label_config.num_classes

        result = batch
        if spectrogram_specs:
            if spectrogram_key not in batch:
                raise ValueError(
                    f"Spectrogram augmentations require batch key {spectrogram_key!r}"
                )
            base_seed = _seed_tensor(seed)
            if base_seed is None:
                base_seed = tf.constant([0, 0], dtype=tf.int32)
            spectrogram_seed, batch_seed = tf.unstack(tf.random.split(base_seed, 2))
            row_seeds = tf.random.split(
                spectrogram_seed, tf.shape(batch[spectrogram_key])[0]
            )
            result = tf.map_fn(
                lambda values: spectrogram_stage(values[0], seed=values[1]),
                (batch, row_seeds),
                fn_output_signature=mapped_output_signature(batch),
            )
        else:
            batch_seed = seed

        if batch_augmentations:
            result = batch_stage(
                result,
                num_classes=effective_num_classes,
                seed=batch_seed,
            )
        return result

    return late_augmentations


def make_postprocessing(
    segment_config: SegmentStrategyConfig | dict | None = None,
    is_training: bool = False,
    frontend: dict | None = None,
    layout: str | None = None,
    dtype: str = "float32",
    output_key: str | None = None,
    static_shape: tuple[int | None, ...] | None = None,
    input_duration: float | None = None,
    target_sample_rate: int | None = None,
    preprocess: dict | None = None,
    label_transform: LabelTransformConfig | dict | None = None,
    **kwargs,
):
    original_segment_config = segment_config
    if segment_config is None:
        segment_keys = set(SegmentStrategyConfig.__dataclass_fields__)
        if segment_keys.intersection(kwargs):
            segment_config = {
                key: kwargs.pop(key) for key in tuple(kwargs) if key in segment_keys
            }
            original_segment_config = segment_config
        elif frontend is None:
            return _identity_sample

    stages = []
    if segment_config is not None:
        stages.append(make_segment_stage(segment_config, is_training=is_training))

    if frontend is not None:
        if layout is None:
            raise ValueError("Acoustic frontend postprocessing requires a layout")
        shape_preset = preset_info(
            frontend=frontend,
            layout=layout,
            dtype=dtype,
            output_key=output_key,
            static_shape=static_shape,
            input_duration=input_duration,
            target_sample_rate=target_sample_rate,
            preprocess=preprocess,
            segment=original_segment_config,
        )
        stages.append(
            make_model_input_stage(
                frontend,
                layout=layout,
                dtype=dtype,
                output_key=output_key,
                preset=shape_preset,
            )
        )

    label_config = (
        LabelTransformConfig.from_dict(label_transform)
        if label_transform is not None
        else None
    )

    if not stages and label_config is None:
        return _identity_sample

    def postprocessing(sample, num_classes=None):
        result = sample
        for stage in stages:
            result = stage(result)
        if label_config is not None and LABEL in result:
            label, metadata = transform_label(
                result[LABEL],
                label_config,
                metadata=result.get(METADATA),
            )
            result = dict(result)
            result[LABEL] = label
            result[METADATA] = metadata
        return result

    return postprocessing

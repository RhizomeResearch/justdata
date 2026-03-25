import tensorflow as tf

from justdata.acoustic.registry import (
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
from justdata.acoustic.schema import LABEL, METADATA
from justdata.acoustic.stages import make_segment_stage


def _identity_sample(sample, *args, **kwargs):
    return sample


def _identity_batch(batch, num_classes=None, seed=None, **kwargs):
    return batch


def _seed_tensor(seed):
    if seed is None:
        return None
    seed = tf.cast(tf.convert_to_tensor(seed), tf.int32)
    if seed.shape.rank == 0:
        return tf.stack([seed, tf.constant(0, dtype=tf.int32)])
    if seed.shape.rank == 1 and seed.shape[0] == 1:
        return tf.stack([seed[0], tf.constant(0, dtype=tf.int32)])
    return seed[:2]


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
    train_augment: dict | None = None,
    is_training: bool = True,
    augment_eval: bool = False,
    **kwargs,
):
    import justdata.acoustic.augment  # noqa: F401
    from justdata.acoustic.augment import make_waveform_augmentation_stage

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
    return _identity_batch


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
                key: kwargs.pop(key)
                for key in tuple(kwargs)
                if key in segment_keys
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

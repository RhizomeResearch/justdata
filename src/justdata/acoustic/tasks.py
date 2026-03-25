from justdata.acoustic.registry import (
    register_audio_batch_augment,
    register_audio_corruption,
    register_audio_decoder,
    register_audio_postprocessor,
    register_audio_resampler,
    register_audio_spectrogram_augment,
    register_audio_waveform_augment,
)
from justdata.acoustic.configs import AudioPreprocessConfig, SegmentStrategyConfig
from justdata.acoustic.postprocessing import make_model_input_stage, preset_info
from justdata.acoustic.preprocessing import make_preprocessing as _make_preprocessing
from justdata.acoustic.stages import make_segment_stage


def _identity_sample(sample, *args, **kwargs):
    return sample


def _identity_batch(batch, num_classes=None, seed=None, **kwargs):
    return batch


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
    **kwargs,
):
    if segment_config is None:
        return _identity_sample

    segment_stage = make_segment_stage(segment_config, is_training=True)

    def augment(sample, seed=None):
        return segment_stage(sample, seed=seed)

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

    if not stages:
        return _identity_sample

    def postprocessing(sample, num_classes=None):
        result = sample
        for stage in stages:
            result = stage(result)
        return result

    return postprocessing

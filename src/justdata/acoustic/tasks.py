from justdata.acoustic.registry import (
    register_audio_batch_augment,
    register_audio_corruption,
    register_audio_decoder,
    register_audio_frontend,
    register_audio_normalization,
    register_audio_postprocessor,
    register_audio_resampler,
    register_audio_spectrogram_augment,
    register_audio_waveform_augment,
)
from justdata.acoustic.configs import AudioPreprocessConfig, SegmentStrategyConfig
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


@register_audio_frontend("raw_waveform")
@register_audio_frontend("stft_magnitude")
@register_audio_frontend("mel_power")
@register_audio_frontend("logmel")
@register_audio_frontend("kaldi_fbank")
@register_audio_frontend("mfcc")
@register_audio_frontend("pcen_mel")
def frontend_placeholder(sample, *args, **kwargs):
    return sample


@register_audio_normalization("none")
@register_audio_normalization("dataset_mean_std")
@register_audio_normalization("per_clip_mean_std")
@register_audio_normalization("per_frequency_mean_std")
@register_audio_normalization("kaldi_cmvn")
@register_audio_normalization("checkpoint_mean_std")
@register_audio_normalization("affine")
def normalization_placeholder(sample, *args, **kwargs):
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
    **kwargs,
):
    if segment_config is None:
        segment_keys = set(SegmentStrategyConfig.__dataclass_fields__)
        if segment_keys.intersection(kwargs):
            segment_config = {
                key: kwargs.pop(key)
                for key in tuple(kwargs)
                if key in segment_keys
            }
        else:
            return _identity_sample

    return make_segment_stage(segment_config, is_training=is_training)

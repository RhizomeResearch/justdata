from justdata.acoustic.registry import (
    register_audio_batch_augment,
    register_audio_channel_strategy,
    register_audio_corruption,
    register_audio_decoder,
    register_audio_eval_view_strategy,
    register_audio_frontend,
    register_audio_normalization,
    register_audio_postprocessor,
    register_audio_resampler,
    register_audio_segment_strategy,
    register_audio_spectrogram_augment,
    register_audio_waveform_augment,
)


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


@register_audio_channel_strategy("keep")
@register_audio_channel_strategy("mono_mean")
@register_audio_channel_strategy("mono_left")
@register_audio_channel_strategy("mono_right")
def channel_placeholder(sample, *args, **kwargs):
    return sample


@register_audio_segment_strategy("random_crop")
@register_audio_segment_strategy("center_crop")
@register_audio_segment_strategy("full")
@register_audio_segment_strategy("sliding")
@register_audio_segment_strategy("pad_or_crop")
def segment_placeholder(sample, *args, **kwargs):
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


@register_audio_eval_view_strategy("center_crop")
@register_audio_eval_view_strategy("full")
@register_audio_eval_view_strategy("sliding")
@register_audio_eval_view_strategy("multi_crop")
def eval_view_placeholder(sample, *args, **kwargs):
    return sample


def make_preprocessing(**kwargs):
    return _identity_sample


def make_augmentations(**kwargs):
    return _identity_sample


def make_late_augmentations(**kwargs):
    return _identity_batch


def make_postprocessing(**kwargs):
    return _identity_sample

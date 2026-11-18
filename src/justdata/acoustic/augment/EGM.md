# `justdata.acoustic.augment`

## Overview

Stateless, seed-driven TensorFlow audio augmentation primitives, configuration dataclasses, and `tf.data` stage builders
covering per-sample waveform augmentations, per-sample spectrogram augmentations, PaSST spectrogram patchout, and
batch-level mixing augmentations. All transforms rely only on the base `tensorflow` dependency and internal `justdata`
modules.

## Interface

- `[__init__.py:__all__]` Re-exports 53 public symbols across `configs.py`, `waveform.py`, `spectrogram.py`,
  `patchout.py`, and `batch.py`.
- `[configs.py:AdditiveNoiseConfig]` Frozen dataclass `(prob=0.5, snr_db_min=5.0, snr_db_max=30.0, noise_kind='white')`.
- `[configs.py:AudioBatchMixStyleConfig]` Frozen dataclass `(prob, mix: Literal['global', 'frequency', 'channel'])`.
- `[configs.py:AudioCutMixSpecConfig]` Frozen dataclass
  `(alpha, prob, axes: Literal['time', 'frequency', 'time_frequency'])`.
- `[configs.py:AudioMixupConfig]` Frozen dataclass
  `(alpha, prob, label_mode: Literal['single_label', 'multi_label', 'event_frames'])`.
- `[configs.py:AudioWavMixConfig]` Frozen dataclass `(alpha, prob)`.
- `[configs.py:CodecSimulationConfig]` Frozen dataclass
  `(prob=0.2, codec: Literal['mulaw', 'alaw', 'mp3_proxy']='mulaw', bitrate=None)`.
- `[configs.py:DynamicRangeCompressionConfig]` Frozen dataclass
  `(prob=0.3, threshold_db_min=-30.0, threshold_db_max=-6.0, ratio_min=2.0, ratio_max=8.0)`.
- `[configs.py:RIRConvolutionConfig]` Frozen dataclass
  `(prob=0.3, rir_dataset=None, normalize_ir=True, compensate_delay=True)`.
- `[configs.py:RandomClippingConfig]` Frozen dataclass `(prob=0.2, min_threshold=0.2, max_threshold=1.0)`.
- `[configs.py:RandomGainConfig]` Frozen dataclass `(prob=0.5, min_db=-12.0, max_db=12.0)`.
- `[configs.py:RandomTimeShiftConfig]` Frozen dataclass
  `(prob=0.5, max_shift_seconds=0.5, mode: Literal['roll', 'zero']='roll')`.
- `[configs.py:SpeedPerturbConfig]` Frozen dataclass `(prob=0.3, rates=(0.9, 1.0, 1.1))`.
- `[waveform.py:make_colored_noise]` `(shape, seed, kind='white') -> tf.Tensor` generating stateless unit-RMS `'white'`,
  `'pink'`, or `'brown'` `tf.float32` noise.
- `[waveform.py:random_gain]`, `[waveform.py:time_shift]`, `[waveform.py:random_crop]`, `[waveform.py:additive_noise]`,
  `[waveform.py:colored_noise]`, `[waveform.py:polarity_inversion]`, `[waveform.py:random_clipping]`,
  `[waveform.py:dynamic_range_compression]`, `[waveform.py:rir_convolution]`, `[waveform.py:speed_perturb]`,
  `[waveform.py:codec_simulation]` Registered per-sample waveform augmentations accepting `[T]` or `[T, C]` tensors.
- `[waveform.py:normalize_waveform_augment_specs]`, `[waveform.py:apply_waveform_augmentations]`,
  `[waveform.py:make_waveform_augmentation_stage]` Spec normalizer, sample-dict applicator, and pre-normalized stage
  builder for waveform augmentations.
- `[spectrogram.py:SpectrogramLayout]` and `[spectrogram.py:BatchSpectrogramLayout]` Module-level `Literal` layout type
  aliases (`Literal['tf', 'tfc', 'cft']` and `Literal['tf', 'tfc', 'cft', 'btf', 'btfc', 'bcft']`).
- `[spectrogram.py:frequency_mask]`, `[spectrogram.py:time_mask]`, `[spectrogram.py:spectrogram_time_roll]`,
  `[spectrogram.py:mel_bin_shift]`, `[spectrogram.py:time_frequency_erasing]`, `[spectrogram.py:mixstyle]`,
  `[spectrogram.py:frequency_mixstyle]`, `[spectrogram.py:random_eq]` Registered spectrogram augmentations preserving
  input dtype and layout.
- `[spectrogram.py:normalize_spectrogram_augment_specs]`, `[spectrogram.py:apply_spectrogram_augmentations]`,
  `[spectrogram.py:make_spectrogram_augmentation_stage]` Spec normalizer (aliasing mapping key `{'patchout': ...}` to
  `'passt_patchout'`), sample-dict applicator, and stage builder for spectrogram augmentations.
- `[patchout.py:structured_patchout_frequency]`, `[patchout.py:structured_patchout_time]`,
  `[patchout.py:unstructured_patchout]`, `[patchout.py:passt_patchout]` Structured frequency/time bin dropping,
  unstructured token patchout, and combined registered `'passt_patchout'` transform accepting a `tf.Tensor` or sample
  `Mapping`.
- `[batch.py:audio_mixup]` (`[batch.py:mixup]`), `[batch.py:audio_wavmix]` (`[batch.py:wavmix]`),
  `[batch.py:audio_cutmix_spec]` (`[batch.py:cutmix_spec]`), `[batch.py:audio_batch_mixstyle]`
  (`[batch.py:batch_mixstyle]`) Registered batch-level augmentations (`'mixup'`, `'wavmix'`, `'cutmix_spec'`,
  `'batch_mixstyle'`).
- `[batch.py:normalize_batch_augment_specs]`, `[batch.py:apply_batch_augmentations]`,
  `[batch.py:make_batch_augmentation_stage]` Spec normalizer (aliasing `'cutmix'` to `'cutmix_spec'` and `'mixstyle'` to
  `'batch_mixstyle'`), batch sample-dict applicator, and stage builder.

## Invariants

- `[waveform.py:_enabled]`, `[spectrogram.py:_enabled]`, `[batch.py:_enabled]`, `[patchout.py:passt_patchout]`
  Evaluation safety: every waveform, spectrogram, patchout, and batch transform acts as an identity no-op whenever
  `not (is_training or augment_eval)`.
- `[configs.py:_ensure_probability]` and `[configs.py:_ensure_order]` All 12 `*Config` dataclasses validate
  `0.0 <= prob <= 1.0` in `__post_init__`, enforce ordered min/max bounds (`RandomGainConfig`, `AdditiveNoiseConfig`,
  `RandomClippingConfig`, `DynamicRangeCompressionConfig`), positive `alpha > 0` (`AudioMixupConfig`,
  `AudioCutMixSpecConfig`, `AudioWavMixConfig`), positive clipping thresholds and speed rates, compression ratios
  `ratio_min >= 1.0` / `ratio_max >= 1.0` (with signed/zero dB thresholds in `DynamicRangeCompressionConfig`), and
  closed `Literal` mode sets, raising `ValueError` on violations.
- `[_common.py:_config_data]` Raises `TypeError` if `config` is neither `None`, a dataclass, nor a `Mapping`; parameter
  precedence is `defaults < config < explicit non-None overrides`.
- `[waveform.py:_as_waveform_tc]` Framing/spectral waveform helpers (`time_shift`, `random_crop`, `rir_convolution`,
  `speed_perturb`, `codec_simulation`, and `'pink'`/`'brown'` `make_colored_noise`) require static rank 1 (`[T]`) or
  rank 2 (`[T, C]`) (raising `ValueError` otherwise) and restore original rank and input `dtype` on return (whereas
  elementwise `random_gain`, `polarity_inversion`, `random_clipping`, `dynamic_range_compression`, and `'white'` noise
  preserve input shape/dtype directly); `[waveform.py:apply_waveform_augmentations]` recomputes `result[DURATION]` via
  `infer_duration` when `DURATION` is present and `sample_rate` is known.
- `[waveform.py:time_shift]`, `[waveform.py:random_crop]`, `[waveform.py:rir_convolution]` Raise `ValueError` when
  required parameters are missing (`sample_rate` when `shift_samples` or `target_length` is unset; `target_length` and
  `crop_duration` both `None`; `rir_dataset` set without an explicit `rir` tensor).
- `[spectrogram.py:_as_tfc]` and `[spectrogram.py:_as_btfc]` `_as_tfc` enforces static rank matching for single-sample
  layouts (`'tf'` rank 2, `'tfc'`/`'cft'` rank 3), while `_as_btfc` accepts both batch layouts (`'btf'` rank 3,
  `'btfc'`/`'bcft'` rank 4) and single-sample `'tf'`/`'tfc'`/`'cft'` (prepending a size-1 batch axis), raising
  `ValueError` otherwise; `_restore_from_tfc`/`_restore_from_btfc` restore input layout while individual transforms
  restore `dtype`.
- `[patchout.py:unstructured_patchout]` and `[patchout.py:_drop_axis]` `unstructured_patchout` flattens `[T, F, C]` to
  `[T * F - dropped, C]` (squeezing channel for rank-2 input) when active and `n_patches` is not statically `0`;
  `_drop_axis` caps pairwise boolean masks at `65536` elements before switching to `tf.tensor_scatter_nd_update`;
  `[patchout.py:passt_patchout]` records dropped index tensors under `result[METADATA]['patchout']` only when
  `debug=True`.
- `[batch.py:_stateless_shuffled_range]` Permutes batch indices via stable `tf.argsort` over stateless uniform draws and
  rolls by `1` if the entire permutation equals `[0, ..., size - 1]` for `batch > 1` (avoiding the identity permutation,
  though partial fixed points remain possible for `batch >= 3`); `[batch.py:audio_batch_mixstyle]` and
  `[spectrogram.py:_mixstyle_impl]` no-op via `tf.cond` when `batch <= 1`.
- `[batch.py:audio_cutmix_spec]` Enforces `input_kind == 'spectrogram'` and matching static time dimension
  `x.shape[1] == y.shape[1]` for rank-3 `'event_frames'` labels, mixing labels by exact area ratio
  `1 - (time_width * freq_width) / (time * freq)` except `('event_frames', axes='time')`, which frame-splices via
  `[batch.py:_spliced_event_labels]`.
- `[batch.py:apply_batch_augmentations]` Skips specs whose resolved `input_key` is absent from `sample` and skips
  label-mixing transforms (`'mixup'`, `'cutmix_spec'`, `'wavmix'`) when `LABEL` is missing or labels are text
  (`label_mode == 'text'` or `dtype == tf.string`).

## Side Effects

- `[__init__.py:__all__]` Importing `justdata.acoustic.augment` imports `waveform`, `spectrogram`, `patchout`, and
  `batch`, executing module-level `@register_audio_*` decorators under `justdata.acoustic.registry._REGISTRY_LOCK`.
- `[waveform.py:random_gain]`, `[waveform.py:time_shift]`, `[waveform.py:random_crop]`, `[waveform.py:additive_noise]`,
  `[waveform.py:colored_noise]`, `[waveform.py:polarity_inversion]`, `[waveform.py:random_clipping]`,
  `[waveform.py:dynamic_range_compression]`, `[waveform.py:rir_convolution]`, `[waveform.py:speed_perturb]`,
  `[waveform.py:codec_simulation]` Register 11 transforms in the global waveform augmentation registry at import time.
- `[spectrogram.py:frequency_mask]`, `[spectrogram.py:time_mask]`, `[spectrogram.py:spectrogram_time_roll]`,
  `[spectrogram.py:mel_bin_shift]`, `[spectrogram.py:time_frequency_erasing]`, `[spectrogram.py:mixstyle]`,
  `[spectrogram.py:frequency_mixstyle]`, `[spectrogram.py:random_eq]`, `[patchout.py:passt_patchout]` Register 9
  transforms in the global spectrogram augmentation registry at import time.
- `[batch.py:audio_mixup]`, `[batch.py:audio_wavmix]`, `[batch.py:audio_cutmix_spec]`, `[batch.py:audio_batch_mixstyle]`
  Register 4 transforms (`'mixup'`, `'wavmix'`, `'cutmix_spec'`, `'batch_mixstyle'`) in the global batch augmentation
  registry at import time.

## Verification

- Run: `uv run pytest tests/acoustic`

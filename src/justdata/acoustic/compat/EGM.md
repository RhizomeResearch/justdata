# `justdata.acoustic.compat`

## Overview

Frontend-golden reference presets, frontend configurations, and pipeline builders reproducing the feature-extraction and
training recipes of external audio model families—AST (`YuanGongND/ast`), CED, EfficientAT / DyMN, PANNs (`cnn14`), and
PaSST—across AudioSet, ESC-50, SpeechCommands, and DCASE 2025 Task 1.

## Interface

- `[__init__.py:__all__]` Re-exports 41 public symbols from `ast.py`, `ced.py`, `efficientat.py`, `panns.py`, and
  `passt.py`.
- `[ast.py:AST_REFERENCE_COMMIT]`, `[ast.py:AST_MIX_WAVEFORM]`, `[ast.py:AST_MIX_LABEL]`, `[ast.py:AST_MIX_LAMBDA]`,
  `[ast.py:AST_RECIPE_CONFIGS]` Reference commit constant, mixup sample-dict keys, and recipe dictionary (`'audioset'`,
  `'esc50'`, `'speechcommands'`).
- `[ast.py:ast_frontend]`, `[ast.py:ast_pad_or_crop_fbank]`, `[ast.py:ast_normalize_fbank]`,
  `[ast.py:ast_frequency_mask]`, `[ast.py:ast_time_mask]`, `[ast.py:ast_specaugment]`,
  `[ast.py:ast_add_noise_and_roll]`, `[ast.py:ast_mix_waveforms]`, `[ast.py:ast_pipeline]` AST `FrontendConfig` builder,
  fbank padding/normalization/SpecAugment/noise-roll/mixup helpers, and
  `(preprocess_fn, augment_fn, late_augment_fn, postprocess_fn)` pipeline builder.
- `[ast.py:ast_audioset_16k_10s_fbank128]`, `[ast.py:ast_esc50_16k_5s_fbank128]`,
  `[ast.py:ast_speechcommands_16k_1s_fbank128]` 16 kHz 128-bin AST `AudioPreset` factories (`layout='btf'`,
  `preset_version=2`, `compatibility_status='frontend_golden'`).
- `[ced.py:ced_frontend]`, `[ced.py:ced_tiny_16k_logmel64]`, `[ced.py:ced_mini_16k_logmel64]`,
  `[ced.py:ced_small_16k_logmel64]`, `[ced.py:ced_base_16k_logmel64]`, `[ced.py:dcase2025_task1_ced_16k_1s]` 16 kHz
  64-bin HTK `torchaudio` dB (`top_db=120.0`) CED `FrontendConfig` and AudioSet / DCASE 2025 Task 1 `AudioPreset`
  factories.
- `[efficientat.py:DCASE_CLASSES]`, `[efficientat.py:efficientat_frontend]`, `[efficientat.py:efficientat_logmel]`
  Re-exported DCASE 10-class tuple, 32 kHz 128-bin `FrontendConfig` builder, and registered `'efficientat_logmel'` audio
  frontend (`preemphasis=0.97`, `frame_length=n_fft`).
- `[efficientat.py:efficientat_32k_10s_logmel128]`, `[efficientat.py:dymn_32k_10s_logmel128]`,
  `[efficientat.py:dcase2025_task1_efficientat_32k_1s]`, `[efficientat.py:dcase2025_task1_dymn_32k_1s]`,
  `[efficientat.py:dcase2025_task1_efficientat_32k_1s_zero_pad_to_10s]`,
  `[efficientat.py:dcase2025_task1_efficientat_32k_1s_repeat_to_10s]` 32 kHz `layout='bcft'` EfficientAT and DyMN
  `AudioPreset` factories.
- `[panns.py:panns_frontend]`, `[panns.py:panns_cnn14_32k_10s_logmel64]`, `[panns.py:panns_cnn14_16k_10s_logmel64]`
  Slaney/librosa dB log-mel `FrontendConfig` builder and 32 kHz / 16 kHz CNN14 `AudioPreset` factories (`layout='btf'`).
- `[passt.py:PATCH_GRID]`, `[passt.py:PATCHOUT]`, `[passt.py:passt_frontend]`, `[passt.py:passt_32k_10s_logmel128]`,
  `[passt.py:dcase2025_task1_passt_32k_1s]`, `[passt.py:dcase2025_task1_passt_32k_1s_zero_pad_to_10s]`,
  `[passt.py:dcase2025_task1_passt_32k_1s_repeat_to_10s]` PaSST patch-grid/patchout constants, 32 kHz
  `'efficientat_logmel'` `FrontendConfig` builder, and `layout='bcft'` AudioSet / DCASE 2025 Task 1 `AudioPreset`
  factories.

## Invariants

- `[ast.py:_metadata_for_recipe]`, `[ast.py:_preprocess]`, `[ast.py:_recipe_preset]`, `[ced.py:_preset]`,
  `[efficientat.py:_preset]`, `[panns.py:_preset]`, `[passt.py:_preset]` Every preset in `compat/` sets
  `preset_version=2`, `compatibility_status='frontend_golden'`, `channel_strategy='mono_mean'`, and
  `normalize_waveform='none'` (frontend-level golden parity only, not checkpoint/logits parity).
- `[ast.py:_target_length]` Requires either `ast['target_length']` or a non-`None` static time dimension in
  `static_shape` under `layout in ('btf', 'btfc', 'bft', 'bcft')`, raising `ValueError` otherwise.
- `[ast.py:ast_pad_or_crop_fbank]` and `[ast.py:ast_normalize_fbank]` Require rank-3 `[T, F, C]` tensors for
  zero-right-padding/cropping along axis `0`, cast to `tf.float32`, and normalize by `(fbank - mean) / (2.0 * std)`.
- `[ast.py:ast_mix_waveforms]` Casts both waveforms to `tf.float32`, expands `[T]` to `[T, 1]`, subtracts global DC mean
  before mixing, pads/crops partner length to match `waveform`, mixes with `lam`, and DC-centers the mixed waveform.
- `[ast.py:ast_pipeline]` Uses stateless TensorFlow PRNGs (`_seed_tensor` + `tf.random.split`); when
  `postproc_kwargs.get('is_training', False)` is `True`, runs the stochastic feature stage (mixup, `ast_specaugment`,
  optional `ast_add_noise_and_roll`) in `augment_fn` and label transform in `postprocess_fn`; when `False`, places
  `identity_sample` in `augment_fn` and deterministic feature extraction in `postprocess_fn`; `late_augment_fn` is
  always `identity_batch`.
- `[efficientat.py:efficientat_logmel]` Raises `ValueError('EfficientAT frontend requires STFT and mel configs')` when
  `config.stft is None` or `config.mel is None`, and computes STFT power with `preemphasis=0.97` and
  `frame_length=config.stft.n_fft`.
- `[passt.py:passt_frontend]` References `FrontendConfig(name='efficientat_logmel')`; resolving that frontend requires
  `justdata.acoustic.compat` or `justdata.acoustic.compat.efficientat` to have been imported so `'efficientat_logmel'`
  is registered.

## Side Effects

- `[__init__.py:__all__]` and `[efficientat.py:efficientat_logmel]` Importing `justdata.acoustic.compat` loads parent
  `justdata.acoustic` and `justdata.acoustic.frontends` (registering the 8 built-in frontends) and registers
  `'efficientat_logmel'` into `justdata.acoustic.registry._AUDIO_FRONTENDS` under `_REGISTRY_LOCK`.
- `[ast.py:ast_pipeline]` Mutates a non-empty caller-passed `preproc_kwargs` dict in place via
  `preproc_kwargs.setdefault('config', preprocess)` when `preprocess is not None`.

## Verification

- Run: `uv run pytest tests/acoustic`

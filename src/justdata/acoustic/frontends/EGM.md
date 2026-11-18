# `justdata.acoustic.frontends`

## Overview

Deterministic TensorFlow spectral and waveform feature-extraction frontends—covering raw waveforms, STFT magnitude/power
spectrograms, mel filterbanks, log/dB/PCEN spectral compression, MFCCs, Kaldi-compatible filterbanks, and AST Kaldi
filterbanks—producing canonical `[T, F, C]` (or `[T, C]`) `tf.float32` tensors.

## Interface

- `[__init__.py:__all__]` Re-exports 16 public symbols: `ast_kaldi_fbank`, `compress_log`, `compute_num_frames`,
  `ensure_waveform_tc`, `kaldi_fbank`, `logmel`, `mel_power`, `mel_spectrogram`, `mel_weight_matrix`, `mfcc`,
  `pcen_compression`, `pcen_mel`, `raw_waveform`, `reflect_or_constant_pad`, `stft_magnitude`, and
  `stft_power_spectrogram`.
- `[stft.py:ensure_waveform_tc]` `(audio: tf.Tensor) -> tf.Tensor` casting `[T]` or `[T, C]` waveforms to `tf.float32`
  `[T, C]`.
- `[stft.py:reflect_or_constant_pad]` `(audio: tf.Tensor, pad: int, pad_mode: str) -> tf.Tensor` symmetrically padding
  axis `0` in `'constant'` or `'reflect'` mode.
- `[stft.py:compute_num_frames]`
  `(num_samples: int, *, n_fft: int, win_length: int, hop_length: int, center: bool) -> int` computing STFT frame
  counts.
- `[stft.py:stft_power_spectrogram]`
  `(audio: tf.Tensor, config: STFTConfig | dict, *, preemphasis: float = 0.0, frame_length: int | None = None) -> tf.Tensor`
  returning `tf.float32` `[T, F, C]` spectrograms.
- `[stft.py:raw_waveform]` and `[stft.py:stft_magnitude]` Registered audio frontends `'raw_waveform'` (`[T, C]`) and
  `'stft_magnitude'` (`[T, F, C]`), both applying `apply_feature_normalization`.
- `[mel.py:mel_weight_matrix]` `(config: FrontendConfig | dict) -> tf.Tensor` building `tf.float32`
  `[n_fft // 2 + 1, n_mels]` filterbanks across `filterbank_impl` backends (`'tf'`, `'librosa'`,
  `'torchaudio_or_kaldi_compatible'`, `'kaldi_compatible'`, `'efficientat_kaldi'`, `'torchaudio'`).
- `[mel.py:mel_spectrogram]` and `[mel.py:mel_power]` Mel projection `[T, M, C]` via `tf.einsum('tfc,fm->tmc')` and
  registered `'mel_power'` frontend with feature normalization.
- `[log.py:pcen_compression]` and `[log.py:compress_log]` Per-Channel Energy Normalization (`tf.scan` along axis `0`)
  and general `'log'` / `'log10'` / `'db'` / `'pcen'` spectral compression in `tf.float32`.
- `[log.py:logmel]` and `[pcen.py:pcen_mel]` Registered `'logmel'` and `'pcen_mel'` frontends combining
  `mel_spectrogram`, spectral compression, and `apply_feature_normalization`.
- `[mfcc.py:mfcc]` Registered `'mfcc'` frontend applying log-mel extraction,
  `tf.signal.mfccs_from_log_mel_spectrograms`, slicing `[..., :config.n_mfcc]` into `[T, n_mfcc, C]`, and feature
  normalization.
- `[kaldi.py:kaldi_fbank]` Registered `'kaldi_fbank'` frontend overriding `mel` to `filterbank_impl='kaldi_compatible'`,
  `mel_scale='htk'`, and `mel_norm='none'` before log compression and feature normalization.
- `[ast.py:ast_kaldi_fbank]` Registered `'ast_kaldi_fbank'` frontend matching AST's `torchaudio.compliance.kaldi.fbank`
  subset and returning `tf.float32` `[T, n_mels, 1]`.

## Invariants

- `[stft.py:ensure_waveform_tc]` Casts inputs to `tf.float32`, expands rank-1 `[T]` to `[T, 1]`, and raises
  `ValueError('Audio frontend input must be a waveform with shape [T] or [T, C]')` if resulting rank != `2`.
- `[stft.py:_window_fn]` Accepts `config.window in {'rectangular', 'hann', 'hamming', 'povey'}` (`hann_window ** 0.85`),
  center-pads `win_length` to `frame_length`, and raises `ValueError` on unknown windows.
- `[stft.py:reflect_or_constant_pad]` Returns `audio` unchanged when `pad <= 0`; when `pad > 0`, raises `ValueError`
  unless `pad_mode in {'constant', 'reflect'}` and falls back via `tf.cond` to `'CONSTANT'` padding when
  `tf.shape(audio)[0] <= pad`.
- `[stft.py:stft_power_spectrogram]` Floors magnitude at `config.eps` before `tf.pow` when `config.power not in {1, 2}`
  and reconstructs two-sided bins via `[stft.py:_full_spectrum_from_onesided]` when `not config.onesided`;
  `[stft.py:stft_magnitude]` raises `ValueError` if `config.stft is None`.
- `[mel.py:_f_max]`, `[mel.py:mel_weight_matrix]`, `[mel.py:mel_spectrogram]`, and `[kaldi.py:kaldi_fbank]` Enforce
  `f_max <= sample_rate / 2.0`, `config.stft is not None`, and `config.mel is not None` (plus
  `config.stft.onesided is True` in `mel_spectrogram`, raising `ValueError` otherwise); `'tf'` and `'torchaudio'`
  implementations require `mel_scale == 'htk'` and `mel_norm == 'none'`; `'librosa'` and
  `'torchaudio_or_kaldi_compatible'` raise `ImportError` when optional `librosa` is absent.
- `[log.py:compress_log]` and `[log.py:pcen_compression]` Floor inputs at `config.amin` in `tf.float32`, floor `'max'`
  dB reference peaks at `1e-30` (`[log.py:_ref_value]`), clamp `'pcen'` outputs below at `0.0`, and raise `ValueError`
  on unknown `config.kind`; `[pcen.py:pcen_mel]` forces `compression.kind = 'pcen'`.
- `[ast.py:ast_kaldi_fbank]` Requires `config.stft` and `config.mel` to be non-`None` and
  `stft.n_fft == _next_power_of_two(stft.win_length)` (raising `ValueError` otherwise); extracts channel `0` only,
  removes global and per-frame DC offset, applies `0.97` preemphasis with replicated first sample and non-periodic Hann
  window, floors mel energies at `[ast.py:_KALDI_EPSILON]` (`1.1920928955078125e-7`), and returns `[T, n_mels, 1]`.

## Side Effects

- `[__init__.py:__all__]` Importing `justdata.acoustic.frontends` imports `ast`, `kaldi`, `log`, `mel`, `mfcc`, `pcen`,
  and `stft`, registering 8 frontends (`ast_kaldi_fbank`, `kaldi_fbank`, `logmel`, `mel_power`, `mfcc`, `pcen_mel`,
  `raw_waveform`, `stft_magnitude`) into `justdata.acoustic.registry._AUDIO_FRONTENDS` under `_REGISTRY_LOCK`.
- `[mel.py:mel_weight_matrix]` Lazily imports optional `librosa` into `sys.modules` at call time when
  `mel.filterbank_impl in {'librosa', 'torchaudio_or_kaldi_compatible'}`.

## Verification

- Run: `uv run pytest tests/acoustic`

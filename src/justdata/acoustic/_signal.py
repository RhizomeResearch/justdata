from __future__ import annotations

import tensorflow as tf


_EPS = tf.constant(1e-8, dtype=tf.float32)


def _fit_length(audio: tf.Tensor, target_length: tf.Tensor) -> tf.Tensor:
    target_length = tf.cast(tf.maximum(target_length, 1), tf.int32)
    audio = audio[:target_length]
    pad = tf.maximum(target_length - tf.shape(audio)[0], 0)
    return tf.pad(audio, [[0, pad], [0, 0]])


def _convolve_channels(
    audio: tf.Tensor, ir: tf.Tensor, compensate_delay: bool
) -> tf.Tensor:
    time = tf.shape(audio)[0]
    kernel = tf.reverse(ir, axis=[0])[:, tf.newaxis, tf.newaxis]
    kernel_length = tf.shape(kernel)[0]
    start = (
        tf.argmax(tf.abs(ir), output_type=tf.int32)
        if compensate_delay
        else (kernel_length - 1) // 2
    )

    def convolve_channel(channel: tf.Tensor) -> tf.Tensor:
        signal = channel[tf.newaxis, :, tf.newaxis]
        padded = tf.pad(
            signal, [[0, 0], [kernel_length - 1, kernel_length - 1], [0, 0]]
        )
        full = tf.nn.conv1d(padded, kernel, stride=1, padding="VALID")[0, :, 0]
        return _fit_length(full[start:, tf.newaxis], time)[:, 0]

    channels_first = tf.transpose(audio, [1, 0])
    convolved = tf.map_fn(
        convolve_channel,
        channels_first,
        fn_output_signature=tf.float32,
    )
    return tf.transpose(convolved, [1, 0])


def _normalize_noise(noise: tf.Tensor) -> tf.Tensor:
    noise = tf.cast(noise, tf.float32)
    axes = tf.range(tf.rank(noise))
    noise = noise - tf.reduce_mean(noise, axis=axes, keepdims=True)
    rms = tf.sqrt(tf.reduce_mean(tf.square(noise), axis=axes, keepdims=True))
    return tf.math.divide_no_nan(noise, tf.maximum(rms, _EPS))


def _add_at_snr(
    audio_or_features: tf.Tensor, noise: tf.Tensor, snr_db: tf.Tensor
) -> tf.Tensor:
    x = tf.cast(audio_or_features, tf.float32)
    noise = tf.cast(noise, tf.float32)
    signal_power = tf.reduce_mean(tf.square(x))
    noise_power = tf.reduce_mean(tf.square(noise))
    target_ratio = tf.pow(tf.constant(10.0, dtype=tf.float32), snr_db / 10.0)
    scale = tf.sqrt(tf.math.divide_no_nan(signal_power, noise_power * target_ratio))
    return x + noise * scale


def _pink_noise(audio: tf.Tensor) -> tf.Tensor:
    """Shape white noise in [T, C] layout into unit-RMS pink noise."""
    time = tf.shape(audio)[0]
    channels_first = tf.transpose(audio, [1, 0])
    spectrum = tf.signal.rfft(channels_first, fft_length=[time])
    num_bins = tf.shape(spectrum)[-1]
    freqs = tf.cast(tf.range(num_bins), tf.float32)
    weights = tf.where(freqs > 0.0, tf.math.rsqrt(freqs), tf.zeros_like(freqs))
    spectrum = spectrum * tf.cast(weights[tf.newaxis, :], spectrum.dtype)
    pink = tf.signal.irfft(spectrum, fft_length=[time])
    pink = tf.transpose(pink, [1, 0])
    return _normalize_noise(pink)

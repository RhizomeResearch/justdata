import numpy as np
import pytest
import tensorflow as tf

from justdata.acoustic._signal import _convolve_channels, _fit_length


def _mapped_convolution(audio, ir, compensate_delay):
    time = tf.shape(audio)[0]
    kernel = tf.reverse(ir, [0])[:, None, None]
    length = tf.shape(kernel)[0]
    start = (
        tf.argmax(tf.abs(ir), output_type=tf.int32)
        if compensate_delay
        else (length - 1) // 2
    )

    def channel_convolution(channel):
        padded = tf.pad(
            channel[None, :, None], [[0, 0], [length - 1, length - 1], [0, 0]]
        )
        full = tf.nn.conv1d(padded, kernel, stride=1, padding="VALID")[0, :, 0]
        return _fit_length(full[start:, None], time)[:, 0]

    return tf.transpose(
        tf.map_fn(
            channel_convolution, tf.transpose(audio), fn_output_signature=tf.float32
        )
    )


@pytest.mark.parametrize("compensate", [False, True])
@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("time,kernel", [(1, 1), (2, 5), (128, 4), (128, 33)])
def test_convolution_exactly_matches_channel_map(compensate, channels, time, kernel):
    rng = np.random.default_rng(12)
    audio = tf.constant(rng.normal(size=(time, channels)), tf.float32)
    ir = tf.constant(rng.normal(size=kernel), tf.float32)

    def reference(audio, ir):
        return _mapped_convolution(audio, ir, compensate)

    def actual(audio, ir):
        return _convolve_channels(audio, ir, compensate)

    for static_channels in (channels, None):
        signature = [
            tf.TensorSpec([None, static_channels], tf.float32),
            tf.TensorSpec([None], tf.float32),
        ]
        compiled_actual = tf.function(actual, input_signature=signature)
        compiled_reference = tf.function(reference, input_signature=signature)
        assert (
            compiled_actual.get_concrete_function().output_shapes
            == compiled_reference.get_concrete_function().output_shapes
        )
        for left, right in [
            (actual(audio, ir), reference(audio, ir)),
            (compiled_actual(audio, ir), compiled_reference(audio, ir)),
        ]:
            assert left.shape == right.shape
            assert left.dtype == right.dtype
            assert left.numpy().tobytes() == right.numpy().tobytes()

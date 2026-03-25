import numpy as np
import tensorflow as tf

from justdata.acoustic.decoding import decode_wav_file


def test_decode_wav_float32_range(tmp_path, write_wav_file):
    path = write_wav_file(
        tmp_path / "mono.wav",
        np.array([-32768, 0, 32767], dtype=np.int16),
    )

    waveform, sample_rate = decode_wav_file(path)

    values = waveform.numpy()
    assert waveform.dtype == tf.float32
    assert sample_rate.numpy() == 16000
    assert values.min() >= -1.0
    assert values.max() <= 1.0
    assert np.isclose(values[0, 0], -1.0)


def test_decode_wav_shape_t_c_for_mono(tmp_path, write_wav_file):
    path = write_wav_file(tmp_path / "mono.wav", np.arange(4, dtype=np.int16))

    waveform, _ = decode_wav_file(path)

    assert waveform.shape == (4, 1)


def test_decode_wav_shape_t_c_for_stereo(tmp_path, write_wav_file):
    data = np.stack(
        [
            np.arange(4, dtype=np.int16),
            np.arange(10, 14, dtype=np.int16),
        ],
        axis=1,
    )
    path = write_wav_file(tmp_path / "stereo.wav", data)

    waveform, _ = decode_wav_file(path)

    assert waveform.shape == (4, 2)

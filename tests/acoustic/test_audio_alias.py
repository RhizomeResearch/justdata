import justdata.acoustic as acoustic
import justdata.audio as audio
from justdata.audio.configs import AudioPreset as AliasAudioPreset


def test_audio_alias_imports_acoustic_symbols():
    assert audio.AudioPreset is acoustic.AudioPreset
    assert audio.get_audio_frontend is acoustic.get_audio_frontend
    assert AliasAudioPreset is acoustic.AudioPreset

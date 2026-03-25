from uuid import uuid4

import pytest

from justdata.acoustic.registry import (
    get_audio_frontend,
    has_audio_frontend,
    list_audio_frontends,
    register_audio_frontend,
)
from justdata.core.registry import get_pipeline, has_pipeline


def test_registry_duplicate_name_raises():
    name = f"unit_frontend_{uuid4().hex}"

    @register_audio_frontend(name)
    def first(sample):
        return sample

    with pytest.raises(ValueError, match="already registered"):

        @register_audio_frontend(name)
        def second(sample):
            return sample


def test_registry_missing_name_raises_clear_error():
    with pytest.raises(ValueError, match="Audio frontend 'missing_frontend' not found"):
        get_audio_frontend("missing_frontend")


def test_list_registries_returns_sorted_names():
    names = list_audio_frontends()

    assert names == tuple(sorted(names))
    assert "raw_waveform" in names
    assert has_audio_frontend("raw_waveform")


def test_acoustic_default_pipeline_registered():
    assert has_pipeline("acoustic/default")

    pipeline = get_pipeline("acoustic/default", apply_presets=False)
    funcs = pipeline.build(is_training=False)

    assert pipeline.pipeline_name == "acoustic/default"
    assert pipeline.modality == "acoustic"
    assert len(funcs) == 4
    assert all(callable(fn) for fn in funcs)

from pathlib import Path

import justdata.acoustic  # noqa: F401
from justdata.core.presets import get_dataset_presets
from justdata.core.registry import get_pipeline
from justdata.vision.presets import get_dataset_presets as get_vision_presets


def test_core_has_no_vision_imports():
    core_dir = Path(__file__).parents[1] / "src" / "justdata" / "core"
    for path in core_dir.glob("*.py"):
        assert "justdata.vision" not in path.read_text()


def test_acoustic_identity_pipeline_registered():
    pipeline = get_pipeline("acoustic/identity", apply_presets=False)

    assert pipeline.pipeline_name == "acoustic/identity"
    assert pipeline.modality == "acoustic"
    assert len(pipeline.build(is_training=False)) == 4


def test_acoustic_presets_do_not_use_vision_default():
    assert get_dataset_presets("unknown_audio", modality="acoustic") == {}
    assert get_vision_presets("unknown_vision")["postproc_kwargs"]["image_size"] == 224

import importlib
import subprocess
import sys
import textwrap
from types import ModuleType

import pytest

import justdata.acoustic as acoustic
import justdata.audio as audio
from justdata.audio.configs import AudioPreset as AliasAudioPreset


PUBLIC_ACOUSTIC_MODULES = tuple(
    name
    for name in acoustic.__all__
    if isinstance(getattr(acoustic, name), ModuleType)
)

NESTED_MODULES = (
    "augment.batch",
    "augment.spectrogram",
    "augment.waveform",
    "corruptions.noise",
    "frontends.stft",
)


def _run_identity_check(module_suffix: str, *, check_registries: bool = False):
    canonical_name = f"justdata.acoustic.{module_suffix}"
    alias_name = f"justdata.audio.{module_suffix}"
    code_parts = ["import importlib\n"]
    if check_registries:
        code_parts.append(
            textwrap.dedent(
                """
                from justdata.acoustic import registry

                def registry_state():
                    return {
                        name: tuple(value())
                        for name, value in vars(registry).items()
                        if name.startswith("list_audio_")
                        and callable(value)
                    }

                before = registry_state()
                """
            )
        )
    code_parts.append(
        f"canonical = importlib.import_module({canonical_name!r})\n"
        f"alias = importlib.import_module({alias_name!r})\n"
        f"assert alias is canonical, ({canonical_name!r}, {alias_name!r})\n"
    )
    if check_registries:
        code_parts.append(
            "after = registry_state()\nassert after == before, (before, after)\n"
        )

    code = "".join(code_parts)
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, (
        f"Import identity failed for {canonical_name!r} and {alias_name!r}.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


def test_audio_alias_imports_acoustic_symbols():
    assert audio.AudioPreset is acoustic.AudioPreset
    assert audio.get_audio_frontend is acoustic.get_audio_frontend
    assert AliasAudioPreset is acoustic.AudioPreset


@pytest.mark.parametrize("module_suffix", PUBLIC_ACOUSTIC_MODULES)
def test_public_audio_modules_preserve_identity_in_fresh_process(module_suffix):
    _run_identity_check(module_suffix)


@pytest.mark.parametrize("module_suffix", NESTED_MODULES)
def test_nested_audio_modules_preserve_identity_and_registries(module_suffix):
    _run_identity_check(module_suffix, check_registries=True)


def test_every_loaded_acoustic_descendant_has_an_audio_alias():
    canonical_prefix = "justdata.acoustic."
    for canonical_name, module in tuple(sys.modules.items()):
        if canonical_name.startswith(canonical_prefix):
            alias_name = f"justdata.audio.{canonical_name.removeprefix(canonical_prefix)}"
            assert importlib.import_module(alias_name) is module, (
                canonical_name,
                alias_name,
            )

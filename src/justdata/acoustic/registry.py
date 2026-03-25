import threading
from collections.abc import Callable
from typing import TypeVar


RegistryFn = TypeVar("RegistryFn", bound=Callable)


_REGISTRY_LOCK = threading.Lock()

_AUDIO_DECODERS: dict[str, Callable] = {}
_AUDIO_RESAMPLERS: dict[str, Callable] = {}
_AUDIO_CHANNEL_STRATEGIES: dict[str, Callable] = {}
_AUDIO_SEGMENT_STRATEGIES: dict[str, Callable] = {}
_AUDIO_FRONTENDS: dict[str, Callable] = {}
_AUDIO_WAVEFORM_AUGMENTS: dict[str, Callable] = {}
_AUDIO_SPECTROGRAM_AUGMENTS: dict[str, Callable] = {}
_AUDIO_BATCH_AUGMENTS: dict[str, Callable] = {}
_AUDIO_NORMALIZATIONS: dict[str, Callable] = {}
_AUDIO_EVAL_VIEW_STRATEGIES: dict[str, Callable] = {}
_AUDIO_POSTPROCESSORS: dict[str, Callable] = {}
_AUDIO_CORRUPTIONS: dict[str, Callable] = {}


def _register(registry: dict[str, Callable], label: str, name: str):
    def decorator(fn: RegistryFn) -> RegistryFn:
        with _REGISTRY_LOCK:
            if name in registry:
                existing = registry[name]
                raise ValueError(
                    f"{label} '{name}' already registered by "
                    f"{existing.__module__}.{existing.__qualname__}"
                )
            registry[name] = fn
        return fn

    return decorator


def _get(registry: dict[str, Callable], label: str, name: str) -> Callable:
    if name not in registry:
        available = ", ".join(sorted(registry)) or "<none>"
        raise ValueError(f"{label} '{name}' not found. Available: {available}")
    return registry[name]


def _list(registry: dict[str, Callable]) -> tuple[str, ...]:
    return tuple(sorted(registry))


def _has(registry: dict[str, Callable], name: str) -> bool:
    return name in registry


def register_audio_decoder(name: str):
    return _register(_AUDIO_DECODERS, "Audio decoder", name)


def get_audio_decoder(name: str) -> Callable:
    return _get(_AUDIO_DECODERS, "Audio decoder", name)


def list_audio_decoders() -> tuple[str, ...]:
    return _list(_AUDIO_DECODERS)


def has_audio_decoder(name: str) -> bool:
    return _has(_AUDIO_DECODERS, name)


def register_audio_resampler(name: str):
    return _register(_AUDIO_RESAMPLERS, "Audio resampler", name)


def get_audio_resampler(name: str) -> Callable:
    return _get(_AUDIO_RESAMPLERS, "Audio resampler", name)


def list_audio_resamplers() -> tuple[str, ...]:
    return _list(_AUDIO_RESAMPLERS)


def has_audio_resampler(name: str) -> bool:
    return _has(_AUDIO_RESAMPLERS, name)


def register_audio_channel_strategy(name: str):
    return _register(_AUDIO_CHANNEL_STRATEGIES, "Audio channel strategy", name)


def get_audio_channel_strategy(name: str) -> Callable:
    return _get(_AUDIO_CHANNEL_STRATEGIES, "Audio channel strategy", name)


def list_audio_channel_strategies() -> tuple[str, ...]:
    return _list(_AUDIO_CHANNEL_STRATEGIES)


def has_audio_channel_strategy(name: str) -> bool:
    return _has(_AUDIO_CHANNEL_STRATEGIES, name)


def register_audio_segment_strategy(name: str):
    return _register(_AUDIO_SEGMENT_STRATEGIES, "Audio segment strategy", name)


def get_audio_segment_strategy(name: str) -> Callable:
    return _get(_AUDIO_SEGMENT_STRATEGIES, "Audio segment strategy", name)


def list_audio_segment_strategies() -> tuple[str, ...]:
    return _list(_AUDIO_SEGMENT_STRATEGIES)


def has_audio_segment_strategy(name: str) -> bool:
    return _has(_AUDIO_SEGMENT_STRATEGIES, name)


def register_audio_frontend(name: str):
    return _register(_AUDIO_FRONTENDS, "Audio frontend", name)


def get_audio_frontend(name: str) -> Callable:
    return _get(_AUDIO_FRONTENDS, "Audio frontend", name)


def list_audio_frontends() -> tuple[str, ...]:
    return _list(_AUDIO_FRONTENDS)


def has_audio_frontend(name: str) -> bool:
    return _has(_AUDIO_FRONTENDS, name)


def register_audio_waveform_augment(name: str):
    return _register(_AUDIO_WAVEFORM_AUGMENTS, "Audio waveform augment", name)


def get_audio_waveform_augment(name: str) -> Callable:
    return _get(_AUDIO_WAVEFORM_AUGMENTS, "Audio waveform augment", name)


def list_audio_waveform_augments() -> tuple[str, ...]:
    return _list(_AUDIO_WAVEFORM_AUGMENTS)


def has_audio_waveform_augment(name: str) -> bool:
    return _has(_AUDIO_WAVEFORM_AUGMENTS, name)


def register_audio_spectrogram_augment(name: str):
    return _register(_AUDIO_SPECTROGRAM_AUGMENTS, "Audio spectrogram augment", name)


def get_audio_spectrogram_augment(name: str) -> Callable:
    return _get(_AUDIO_SPECTROGRAM_AUGMENTS, "Audio spectrogram augment", name)


def list_audio_spectrogram_augments() -> tuple[str, ...]:
    return _list(_AUDIO_SPECTROGRAM_AUGMENTS)


def has_audio_spectrogram_augment(name: str) -> bool:
    return _has(_AUDIO_SPECTROGRAM_AUGMENTS, name)


def register_audio_batch_augment(name: str):
    return _register(_AUDIO_BATCH_AUGMENTS, "Audio batch augment", name)


def get_audio_batch_augment(name: str) -> Callable:
    return _get(_AUDIO_BATCH_AUGMENTS, "Audio batch augment", name)


def list_audio_batch_augments() -> tuple[str, ...]:
    return _list(_AUDIO_BATCH_AUGMENTS)


def has_audio_batch_augment(name: str) -> bool:
    return _has(_AUDIO_BATCH_AUGMENTS, name)


def register_audio_normalization(name: str):
    return _register(_AUDIO_NORMALIZATIONS, "Audio normalization", name)


def get_audio_normalization(name: str) -> Callable:
    return _get(_AUDIO_NORMALIZATIONS, "Audio normalization", name)


def list_audio_normalizations() -> tuple[str, ...]:
    return _list(_AUDIO_NORMALIZATIONS)


def has_audio_normalization(name: str) -> bool:
    return _has(_AUDIO_NORMALIZATIONS, name)


def register_audio_eval_view_strategy(name: str):
    return _register(_AUDIO_EVAL_VIEW_STRATEGIES, "Audio eval view strategy", name)


def get_audio_eval_view_strategy(name: str) -> Callable:
    return _get(_AUDIO_EVAL_VIEW_STRATEGIES, "Audio eval view strategy", name)


def list_audio_eval_view_strategies() -> tuple[str, ...]:
    return _list(_AUDIO_EVAL_VIEW_STRATEGIES)


def has_audio_eval_view_strategy(name: str) -> bool:
    return _has(_AUDIO_EVAL_VIEW_STRATEGIES, name)


def register_audio_postprocessor(name: str):
    return _register(_AUDIO_POSTPROCESSORS, "Audio postprocessor", name)


def get_audio_postprocessor(name: str) -> Callable:
    return _get(_AUDIO_POSTPROCESSORS, "Audio postprocessor", name)


def list_audio_postprocessors() -> tuple[str, ...]:
    return _list(_AUDIO_POSTPROCESSORS)


def has_audio_postprocessor(name: str) -> bool:
    return _has(_AUDIO_POSTPROCESSORS, name)


def register_audio_corruption(name: str):
    return _register(_AUDIO_CORRUPTIONS, "Audio corruption", name)


def get_audio_corruption(name: str) -> Callable:
    return _get(_AUDIO_CORRUPTIONS, "Audio corruption", name)


def list_audio_corruptions() -> tuple[str, ...]:
    return _list(_AUDIO_CORRUPTIONS)


def has_audio_corruption(name: str) -> bool:
    return _has(_AUDIO_CORRUPTIONS, name)

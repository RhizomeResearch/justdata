import threading
from typing import Any, Dict, Protocol


class DatasetAdapter(Protocol):
    """Maps raw dataset samples to a canonical dict schema."""

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]: ...


_ADAPTERS: Dict[str, DatasetAdapter] = {}
_ADAPTER_LOCK = threading.Lock()


def register_adapter(dataset_name: str):
    def decorator(fn: DatasetAdapter):
        with _ADAPTER_LOCK:
            if dataset_name in _ADAPTERS:
                raise ValueError(
                    f"Adapter for '{dataset_name}' already registered by "
                    f"{_ADAPTERS[dataset_name].__module__}.{_ADAPTERS[dataset_name].__qualname__}"
                )
            _ADAPTERS[dataset_name] = fn
        return fn

    return decorator


def get_adapter(dataset_name: str) -> DatasetAdapter:
    return _ADAPTERS.get(dataset_name, _default_adapter)


def _default_adapter(sample: dict) -> dict:
    """Identity — assumes sample already has the expected keys."""
    return sample

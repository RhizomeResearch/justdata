from __future__ import annotations

from justdata.core.finalization import _pad_nested


def pad_nested(value, pad_size, *, metadata_mode: str = "full"):
    """Pad the first dimension of tensors in nested dictionaries."""
    return _pad_nested(value, pad_size, metadata_mode=metadata_mode)


__all__ = ["pad_nested"]

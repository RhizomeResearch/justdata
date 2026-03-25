from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


AugmentationDomain = Literal["waveform", "spectrogram", "image", "batch"]


@dataclass(frozen=True)
class AugmentationMetadata:
    name: str
    domain: AugmentationDomain
    is_training_only: bool
    requires_labels: bool


def attach_augmentation_metadata(fn, metadata: AugmentationMetadata):
    registrations = tuple(getattr(fn, "augmentation_metadata", ()))
    if metadata not in registrations:
        registrations = registrations + (metadata,)
    fn.augmentation_metadata = registrations
    fn.name = metadata.name
    fn.domain = metadata.domain
    fn.is_training_only = metadata.is_training_only
    fn.requires_labels = metadata.requires_labels
    return fn


__all__ = [
    "AugmentationDomain",
    "AugmentationMetadata",
    "attach_augmentation_metadata",
]

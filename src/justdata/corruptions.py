from __future__ import annotations

from typing import Literal


def create_corruption_datasets(
    modality: Literal["vision", "acoustic"],
    corruption_types: list[str],
    severity: int,
    base_dataset: str,
    preset: str | None,
    *,
    split: str = "validation",
    seed: int = 0,
    **load_kwargs,
):
    if modality == "vision":
        import justdata.vision  # noqa: F401
        from justdata.core.registry import get_pipeline
        from justdata.vision.minic import create_minic_datasets

        kwargs = dict(load_kwargs)
        kwargs.setdefault("dataset_names_arg", base_dataset)
        kwargs.setdefault("splits_arg", split)
        kwargs.setdefault("dataset_type", "train" if split == "train" else "validation")
        kwargs.setdefault("seed", seed)
        if "pipeline" not in kwargs and "postprocess_fn" not in kwargs:
            kwargs["pipeline"] = get_pipeline(dataset=base_dataset, preset=preset)
        return create_minic_datasets(
            corruption_types=corruption_types,
            severity=severity,
            **kwargs,
        )

    if modality == "acoustic":
        import justdata.acoustic  # noqa: F401
        from justdata.acoustic.corruptions import create_audio_corruption_datasets

        return create_audio_corruption_datasets(
            corruption_types=corruption_types,
            severity=severity,
            base_dataset=base_dataset,
            preset=preset,
            split=split,
            seed=seed,
            **load_kwargs,
        )

    raise ValueError("modality must be 'vision' or 'acoustic'.")


__all__ = ["create_corruption_datasets"]

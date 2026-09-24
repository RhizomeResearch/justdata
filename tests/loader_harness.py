from unittest.mock import patch

from justdata.core.loader import load_ds


def identity(sample, *args, **kwargs):
    return sample


def load_mocked(source, *, dataset_type="validation", batch_size=2, seed=0, **options):
    """Run ``load_ds`` over an in-memory source dataset.

    Identity stage callbacks are supplied unless a ``pipeline`` or an explicit
    callback is given; ``options`` are forwarded to ``load_ds`` unchanged.
    """
    callbacks = (
        {}
        if "pipeline" in options
        else {
            "preprocess_fn": identity,
            "augment_fn": identity,
            "late_augment_fn": identity,
            "postprocess_fn": identity,
        }
    )
    with patch("justdata.core.loader.fetch_ds", return_value=source):
        return load_ds(
            dataset_names_arg="mock",
            splits_arg=dataset_type,
            dataset_type=dataset_type,
            batch_size=batch_size,
            seed=seed,
            **(callbacks | options),
        )

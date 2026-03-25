from unittest.mock import patch

import numpy as np
import tensorflow as tf

from justdata.core.loader import load_ds


def _identity(sample, *args, **kwargs):
    return sample


def test_as_numpy_returns_numpy_arrays():
    raw = tf.data.Dataset.from_tensor_slices({"x": np.array([1, 2, 3], dtype=np.int32)})
    raw = raw.apply(tf.data.experimental.assert_cardinality(3))

    with patch("justdata.core.loader.fetch_ds", return_value=raw):
        ds, _n = load_ds(
            dataset_names_arg="mock",
            splits_arg="validation",
            dataset_type="validation",
            batch_size=2,
            seed=0,
            preprocess_fn=_identity,
            augment_fn=_identity,
            late_augment_fn=_identity,
            postprocess_fn=_identity,
            cache_dataset=False,
            as_numpy=True,
        )

    batch = next(iter(ds))

    assert isinstance(batch["x"], np.ndarray)
    np.testing.assert_array_equal(batch["x"], [1, 2])

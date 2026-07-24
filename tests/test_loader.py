from unittest.mock import patch

import numpy as np
import pytest
import tensorflow as tf

from justdata.core.loader import fetch_ds, load_ds
from justdata.core.registry import get_pipeline_for_dataset
from justdata.vision.minic import create_minic_datasets


def test_load_ds_classification(synthetic_classification_ds):
    preproc, aug, laug, postproc = get_pipeline_for_dataset(
        "cifar10",
        task_type="classification",
        apply_presets=False,
        is_training=True,
        aug_kwargs={"image_size": 32},
        postproc_kwargs={"image_size": 32, "num_classes": 10},
    )

    with patch(
        "justdata.core.loader.fetch_ds", return_value=synthetic_classification_ds
    ):
        ds, N = load_ds(
            dataset_names_arg="mock",
            splits_arg="train",
            dataset_type="train",
            batch_size=4,
            seed=42,
            preprocess_fn=preproc,
            augment_fn=aug,
            late_augment_fn=laug,
            postprocess_fn=postproc,
            num_classes=10,
            shuffle_buffer=10,
            cache_dataset=False,
            drop_remainder=False,
        )

        assert N == 5
        batch = next(iter(ds))
        assert "image" in batch
        assert "label" in batch
        assert batch["image"].shape == (4, 3, 32, 32)
        assert batch["image"].dtype == tf.float32


def test_load_ds_segmentation(synthetic_segmentation_ds):
    preproc, aug, laug, postproc = get_pipeline_for_dataset(
        "kitti_road",
        task_type="segmentation",
        apply_presets=False,
        is_training=True,
        aug_kwargs={"image_size": 32},
        postproc_kwargs={"image_size": 32},
    )

    with patch("justdata.core.loader.fetch_ds", return_value=synthetic_segmentation_ds):
        ds, N = load_ds(
            dataset_names_arg="mock",
            splits_arg="train",
            dataset_type="train",
            batch_size=4,
            seed=42,
            preprocess_fn=preproc,
            augment_fn=aug,
            late_augment_fn=laug,
            postprocess_fn=postproc,
            num_classes=None,
            shuffle_buffer=10,
            cache_dataset=False,
            drop_remainder=False,
        )

        assert N > 0
        batch = next(iter(ds))
        assert "image" in batch
        assert "mask" in batch
        assert batch["image"].dtype == tf.float32


def test_create_minic_datasets(synthetic_classification_ds):
    preproc, aug, laug, postproc = get_pipeline_for_dataset(
        "cifar10",
        task_type="classification",
        apply_presets=False,
        is_training=False,
        aug_kwargs={"image_size": 32},
        postproc_kwargs={"image_size": 32, "num_classes": 10},
    )

    with patch(
        "justdata.core.loader.fetch_ds", return_value=synthetic_classification_ds
    ):
        ds_list, N = create_minic_datasets(
            corruption_types=["noise", "gaussian_noise"],
            severity=1,
            corruption_versions={"gaussian_noise": "1.0.0"},
            dataset_names_arg="mock",
            splits_arg="val",
            dataset_type="validation",
            batch_size=4,
            seed=42,
            preprocess_fn=preproc,
            augment_fn=aug,
            late_augment_fn=laug,
            postprocess_fn=postproc,
            num_classes=10,
            cache_dataset=False,
        )

        assert isinstance(ds_list, list)
        assert len(ds_list) == 2
        assert N == 5

        for name, ds in zip(
            ("noise", "gaussian_noise"),
            ds_list,
            strict=True,
        ):
            batch = next(iter(ds))
            assert "image" in batch
            assert "metadata" in batch
            assert "corruption" in batch["metadata"]
            assert "corruption_version" in batch["metadata"]
            assert "corruption_identity" in batch["metadata"]
            assert "corruption_identity_hash" in batch["metadata"]
            assert "severity" in batch["metadata"]
            assert "corruption_domain" in batch["metadata"]
            assert batch["metadata"]["corruption"].numpy()[0].decode() == name
            assert batch["metadata"]["corruption_version"].numpy()[0] == b"1.0.0"
            assert (
                batch["metadata"]["corruption_identity"]
                .numpy()[0]
                .decode()
                .startswith(f"{name}@1.0.0:sha256:")
            )
            assert batch["metadata"]["corruption_identity_hash"].dtype == tf.int64
            assert batch["metadata"]["corruption_domain"].numpy()[0] == b"image"
            assert batch["image"].shape == (4, 3, 32, 32)
            assert batch["image"].dtype == tf.float32


def _identity(sample, *args, **kwargs):
    return sample


def test_source_filter_runs_before_adapter_and_regular_filter(monkeypatch):
    counters = {"adapter": 0, "preprocess": 0}
    raw_ds = tf.data.Dataset.from_tensor_slices(
        {
            "x": np.arange(4, dtype=np.int64),
            "keep": [False, True, True, False],
        }
    )

    def source_loader(dataset_name, splits, data_dir):
        del dataset_name, splits, data_dir
        return [raw_ds]

    def adapter(sample):
        def bump(value):
            counters["adapter"] += 1
            return value

        value = tf.py_function(bump, [sample["x"]], Tout=tf.int64)
        value.set_shape([])
        return {"x": value + 10}

    def preprocess(sample):
        def bump(value):
            counters["preprocess"] += 1
            return value

        value = tf.py_function(bump, [sample["x"]], Tout=tf.int64)
        value.set_shape([])
        return {"x": value * 2}

    monkeypatch.setattr(
        "justdata.core.loader.get_source_loader", lambda _name: source_loader
    )
    monkeypatch.setattr("justdata.core.loader.get_adapter", lambda _name: adapter)

    ds, _tools = load_ds(
        dataset_names_arg="mock",
        splits_arg="validation",
        dataset_type="validation",
        batch_size=2,
        seed=0,
        preprocess_fn=preprocess,
        augment_fn=_identity,
        late_augment_fn=_identity,
        postprocess_fn=_identity,
        cache_dataset=True,
        return_raw_ds=True,
        source_filter_fn=lambda sample: sample["keep"],
        filter_fn=lambda sample: tf.equal(sample["x"], 22),
    )

    assert _iterate_x_twice(ds) == [[22], [22]]
    assert counters == {"adapter": 2, "preprocess": 2}


def test_fetch_ds_source_filter_is_optional(monkeypatch):
    raw_ds = tf.data.Dataset.from_tensor_slices({"x": [1, 2]})

    monkeypatch.setattr(
        "justdata.core.loader.get_source_loader",
        lambda _name: lambda _dataset, _splits, _data_dir: [raw_ds],
    )
    monkeypatch.setattr("justdata.core.loader.get_adapter", lambda _name: _identity)

    ds = fetch_ds(["mock"], ["validation"])

    assert [int(sample["x"].numpy()) for sample in ds] == [1, 2]


def _counting_postprocess(counter):
    def postprocess(sample, num_classes=None):
        del num_classes

        def bump(value):
            counter["calls"] += 1
            return value

        value = tf.py_function(bump, [sample["x"]], Tout=sample["x"].dtype)
        value.set_shape(sample["x"].shape)
        return sample | {"x": value * 2}

    return postprocess


def _range_dataset(size):
    return tf.data.Dataset.from_tensor_slices(
        {"x": np.arange(size, dtype=np.int64)}
    ).apply(tf.data.experimental.assert_cardinality(size))


def _counted_dataset(size, counter):
    def generate():
        for value in range(size):
            counter["source"] += 1
            yield {"x": np.int64(value)}

    return tf.data.Dataset.from_generator(
        generate,
        output_signature={"x": tf.TensorSpec([], tf.int64)},
    )


def _counting_preprocess(counter):
    def preprocess(sample):
        def bump(value):
            counter["preprocess"] += 1
            return value

        value = tf.py_function(bump, [sample["x"]], Tout=sample["x"].dtype)
        value.set_shape(sample["x"].shape)
        return sample | {"x": value * 2}

    return preprocess


def _load_counted_raw_dataset(counter, **kwargs):
    with patch(
        "justdata.core.loader.fetch_ds",
        return_value=_counted_dataset(3, counter),
    ) as fetch:
        ds, _tools = load_ds(
            dataset_names_arg="mock",
            splits_arg="validation",
            dataset_type="validation",
            batch_size=2,
            seed=0,
            preprocess_fn=_counting_preprocess(counter),
            augment_fn=_identity,
            late_augment_fn=_identity,
            postprocess_fn=_identity,
            return_raw_ds=True,
            **kwargs,
        )
    return ds, fetch


def _iterate_x_twice(ds):
    return [[sample["x"].numpy().item() for sample in ds] for _ in range(2)]


def test_loader_cache_is_disabled_by_default():
    counter = {"source": 0, "preprocess": 0}
    ds, _fetch = _load_counted_raw_dataset(counter)

    assert _iterate_x_twice(ds) == [[0, 2, 4], [0, 2, 4]]
    assert counter == {"source": 6, "preprocess": 6}


def test_loader_memory_cache_requires_explicit_opt_in():
    counter = {"source": 0, "preprocess": 0}
    ds, _fetch = _load_counted_raw_dataset(
        counter,
        cache_dataset=True,
        cache_path="",
    )

    assert _iterate_x_twice(ds) == [[0, 2, 4], [0, 2, 4]]
    assert counter == {"source": 3, "preprocess": 3}


def test_loader_disk_cache_is_separate_from_source_cache(tmp_path):
    counter = {"source": 0, "preprocess": 0}
    source_cache_root = tmp_path / "sources"
    decoded_cache_path = tmp_path / "decoded" / "validation"
    decoded_cache_path.parent.mkdir()
    ds, fetch = _load_counted_raw_dataset(
        counter,
        data_dir=source_cache_root,
        cache_dataset=True,
        cache_path=str(decoded_cache_path),
    )

    assert _iterate_x_twice(ds) == [[0, 2, 4], [0, 2, 4]]
    assert counter == {"source": 3, "preprocess": 3}
    fetch.assert_called_once_with(["mock"], ["validation"], source_cache_root)
    assert list(decoded_cache_path.parent.glob("validation*"))
    assert not source_cache_root.exists()


def test_late_augmentation_sees_unpadded_final_batch():
    batch_size = 4

    def record_batch_size(batch, num_classes=None, seed=None):
        del num_classes, seed
        observed_size = tf.shape(batch["x"])[0]
        return batch | {
            "late_batch_size": tf.fill([observed_size], observed_size),
        }

    with patch(
        "justdata.core.loader.fetch_ds",
        return_value=_range_dataset(batch_size + 1),
    ):
        ds, _n = load_ds(
            dataset_names_arg="mock",
            splits_arg="train",
            dataset_type="train",
            batch_size=batch_size,
            seed=0,
            preprocess_fn=_identity,
            augment_fn=_identity,
            late_augment_fn=record_batch_size,
            postprocess_fn=_identity,
            shuffle_buffer=1,
            cache_dataset=False,
            drop_remainder=False,
        )

    final_batch = list(ds)[-1]

    np.testing.assert_array_equal(
        final_batch["late_batch_size"].numpy(),
        [1, 0, 0, 0],
    )
    np.testing.assert_array_equal(
        final_batch["padding_mask"].numpy(),
        [True, False, False, False],
    )


def test_pipeline_augment_eval_runs_sample_and_batch_augmentation():
    class EvalAugmentPipeline:
        kwargs = {"augment_eval": True}

        @staticmethod
        def build(is_training):
            assert is_training is False

            def augment(sample, seed=None):
                del seed
                return sample | {"x": sample["x"] + 1}

            def late_augment(batch, num_classes=None, seed=None):
                del num_classes, seed
                return batch | {"x": batch["x"] * 10}

            return _identity, augment, late_augment, _identity

    with patch(
        "justdata.core.loader.fetch_ds",
        return_value=_range_dataset(3),
    ):
        ds, n_batches = load_ds(
            dataset_names_arg="mock",
            splits_arg="validation",
            dataset_type="validation",
            batch_size=2,
            seed=0,
            pipeline=EvalAugmentPipeline(),
            deterministic=True,
        )

    batches = list(ds)
    assert n_batches == 2
    np.testing.assert_array_equal(batches[0]["x"].numpy(), [10, 20])
    np.testing.assert_array_equal(batches[1]["x"].numpy(), [30, 0])


def test_validation_skips_augmentation_without_opt_in():
    def augment(sample, seed=None):
        del seed
        return sample | {"x": sample["x"] + 1}

    def late_augment(batch, num_classes=None, seed=None):
        del num_classes, seed
        return batch | {"x": batch["x"] * 10}

    with patch(
        "justdata.core.loader.fetch_ds",
        return_value=_range_dataset(2),
    ):
        ds, _n = load_ds(
            dataset_names_arg="mock",
            splits_arg="validation",
            dataset_type="validation",
            batch_size=2,
            seed=0,
            preprocess_fn=_identity,
            augment_fn=augment,
            late_augment_fn=late_augment,
            postprocess_fn=_identity,
            deterministic=True,
        )

    np.testing.assert_array_equal(next(iter(ds))["x"].numpy(), [0, 1])


def test_validation_model_input_cache_reuses_postprocessed_samples(tmp_path):
    counter = {"calls": 0}
    with patch("justdata.core.loader.fetch_ds", return_value=_range_dataset(4)):
        ds, _n = load_ds(
            dataset_names_arg="mock",
            splits_arg="validation",
            dataset_type="validation",
            batch_size=2,
            seed=0,
            preprocess_fn=_identity,
            augment_fn=_identity,
            late_augment_fn=_identity,
            postprocess_fn=_counting_postprocess(counter),
            cache_dataset=False,
            cache_model_inputs=True,
            model_input_cache_path=str(tmp_path / "model-inputs"),
        )

    first = [batch["x"].numpy().tolist() for batch in ds]
    second = [batch["x"].numpy().tolist() for batch in ds]

    assert first == second == [[0, 2], [4, 6]]
    assert counter["calls"] == 4


def test_train_model_input_cache_reuses_postprocess_and_reshuffles(tmp_path):
    counter = {"calls": 0}
    with patch("justdata.core.loader.fetch_ds", return_value=_range_dataset(20)):
        ds, _n = load_ds(
            dataset_names_arg="mock",
            splits_arg="train",
            dataset_type="train",
            batch_size=20,
            seed=7,
            preprocess_fn=_identity,
            augment_fn=_identity,
            late_augment_fn=_identity,
            postprocess_fn=_counting_postprocess(counter),
            shuffle_buffer=20,
            cache_dataset=False,
            cache_model_inputs=True,
            model_input_cache_path=str(tmp_path / "train-model-inputs"),
            allow_train_model_input_cache=True,
        )

    first = next(iter(ds))["x"].numpy()
    second = next(iter(ds))["x"].numpy()

    assert counter["calls"] == 20
    assert sorted(first.tolist()) == list(range(0, 40, 2))
    assert sorted(second.tolist()) == list(range(0, 40, 2))
    assert not np.array_equal(first, second)


def test_train_model_input_cache_requires_explicit_opt_in(tmp_path):
    with pytest.raises(ValueError, match="allow_train_model_input_cache"):
        load_ds(
            dataset_names_arg="mock",
            splits_arg="train",
            dataset_type="train",
            batch_size=2,
            seed=0,
            preprocess_fn=_identity,
            augment_fn=_identity,
            late_augment_fn=_identity,
            postprocess_fn=_identity,
            cache_dataset=False,
            cache_model_inputs=True,
            model_input_cache_path=str(tmp_path / "train-model-inputs"),
        )


def test_raw_dataset_finalizer_supports_model_input_cache():
    with patch(
        "justdata.core.loader.fetch_ds",
        return_value=_range_dataset(3),
    ):
        raw_ds, tools = load_ds(
            dataset_names_arg="mock",
            splits_arg="train",
            dataset_type="validation",
            batch_size=2,
            seed=0,
            preprocess_fn=_identity,
            augment_fn=_identity,
            late_augment_fn=_identity,
            postprocess_fn=_identity,
            cache_model_inputs=True,
            return_raw_ds=True,
        )

    ds, n_batches = tools["finalize_fn"](raw_ds)

    assert n_batches == 2
    np.testing.assert_array_equal(
        list(ds)[-1]["padding_mask"].numpy(),
        [True, False],
    )


def test_model_input_cache_rejects_preprocess_cache_path_collision(tmp_path):
    cache_path = str(tmp_path / "same-cache")
    with pytest.raises(ValueError, match="must be different"):
        load_ds(
            dataset_names_arg="mock",
            splits_arg="validation",
            dataset_type="validation",
            batch_size=2,
            seed=0,
            preprocess_fn=_identity,
            augment_fn=_identity,
            late_augment_fn=_identity,
            postprocess_fn=_identity,
            cache_dataset=True,
            cache_path=cache_path,
            cache_model_inputs=True,
            model_input_cache_path=cache_path,
        )

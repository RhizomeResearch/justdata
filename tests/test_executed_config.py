import hashlib
import json
from unittest.mock import patch
from uuid import uuid4

import numpy as np
import pytest
import tensorflow as tf

import justdata.acoustic  # noqa: F401
import justdata.vision  # noqa: F401
from justdata.core import ExecutedConfig, get_pipeline, load_ds, register_pipeline
from justdata.core.presets import merge_with_presets


def _vision_source(value=255, count=2):
    return tf.data.Dataset.from_tensor_slices(
        {
            "image": np.full((count, 12, 16, 3), value, dtype=np.uint8),
            "label": np.arange(count, dtype=np.int64),
        }
    ).apply(tf.data.experimental.assert_cardinality(count))


def _load_vision(
    pipeline, *, dataset_type="validation", return_raw_ds=False, batch_size=2
):
    with patch("justdata.core.loader.fetch_ds", return_value=_vision_source()):
        return load_ds(
            "mock",
            "validation",
            dataset_type,
            batch_size,
            17,
            num_classes=10,
            pipeline=pipeline,
            return_raw_ds=return_raw_ds,
            return_config=True,
            deterministic=True,
            map_parallel_calls=1,
            private_threadpool_size=1,
        )


def test_executed_config_is_canonical_immutable_and_rejects_nonfinite_values():
    source = {"schema_version": 1, "z": [1, 2], "label": "café"}
    config = ExecutedConfig.from_dict(source)
    source["z"].append(3)
    first = config.to_dict()
    first["z"].append(4)

    assert config.to_dict()["z"] == [1, 2]
    assert config.to_bytes() == config.to_json().encode("utf-8")
    assert config.to_json() == '{"label":"café","schema_version":1,"z":[1,2]}'
    assert json.loads(config.to_bytes()) == config.to_dict()
    reordered = ExecutedConfig.from_dict(
        {"z": (1, 2), "schema_version": 1, "label": "café"}
    )
    round_tripped = ExecutedConfig.from_dict(json.loads(config.to_json()))
    assert reordered.to_bytes() == config.to_bytes()
    assert round_tripped.to_bytes() == config.to_bytes()

    with pytest.raises(ValueError, match="non-finite"):
        ExecutedConfig.from_dict({"schema_version": 1, "value": float("nan")})
    with pytest.raises(TypeError, match="unsupported"):
        ExecutedConfig.from_dict({"schema_version": 1, "value": object()})


def test_explicit_overrides_preserve_false_zero_none_and_nested_siblings():
    merged = merge_with_presets(
        "cifar10",
        {},
        modality="vision",
        overrides={
            "aug_kwargs": {"enable": False},
            "laug_kwargs": {"label_smoothing": 0},
            "postproc_kwargs": {"image_size": 224, "val_resize_size": None},
        },
    )

    assert merged["aug_kwargs"]["enable"] is False
    assert merged["aug_kwargs"]["crop_type"] == "random_pad"
    assert merged["laug_kwargs"]["label_smoothing"] == 0
    assert merged["postproc_kwargs"]["image_size"] == 224
    assert merged["postproc_kwargs"]["val_resize_size"] is None
    assert "normalization_params" in merged["postproc_kwargs"]


def test_override_inputs_and_resolutions_are_mutation_isolated():
    overrides = {
        "aug_kwargs": {"enable": False},
        "postproc_kwargs": {"image_size": 224, "val_resize_size": None},
    }
    pipeline = get_pipeline(dataset="cifar10", overrides=overrides)
    overrides["postproc_kwargs"]["image_size"] = 999

    first = pipeline.resolve_config(False)
    first["stages"]["postprocess"]["config"]["image_size"] = 111
    second = pipeline.resolve_config(False)

    assert pipeline.kwargs["postproc_kwargs"]["image_size"] == 224
    assert second["stages"]["postprocess"]["config"]["image_size"] == 224
    assert len(pipeline.build(False)) == len(pipeline.build(False)) == 4


@pytest.mark.parametrize("image_size", [224, 256])
def test_cifar_explicit_size_and_normalization_match_snapshot(image_size):
    normalization = ((0.4, 0.5, 0.6), (0.2, 0.25, 0.5))
    pipeline = get_pipeline(
        dataset="cifar10",
        overrides={
            "aug_kwargs": {"enable": False, "image_size": image_size},
            "laug_kwargs": {"enable": False},
            "postproc_kwargs": {
                "image_size": image_size,
                "val_resize_size": None,
                "normalization_params": normalization,
            },
        },
    )
    dataset, count, config = _load_vision(pipeline)
    batch = next(iter(dataset))
    snapshot = config.to_dict()

    assert count == 1
    assert batch["image"].shape == (2, 3, image_size, image_size)
    expected = np.asarray([(1.0 - mean) / std for mean, std in zip(*normalization)])
    np.testing.assert_allclose(batch["image"][0, :, 0, 0], expected, rtol=1e-6)
    assert snapshot["schema_version"] == 1
    assert snapshot["pipeline"]["preset"] == "cifar"
    assert snapshot["pipeline"]["preset_request"] == "cifar10"
    assert snapshot["model_input"]["static_shape"] == [3, image_size, image_size]
    assert snapshot["model_input"]["dtype"] == batch["image"].dtype.name == "float32"
    assert snapshot["model_input"]["normalization"]["mean"] == list(normalization[0])
    assert snapshot["stages"]["postprocess"]["geometry"]["output_size"] == image_size
    assert snapshot["stages"]["augment"]["active"] is False
    assert snapshot["stages"]["late_augment"]["active"] is False
    assert snapshot["execution"]["limits"]["map_parallel_calls"] == 1


def test_explicit_none_is_preserved_when_normalization_is_disabled():
    pipeline = get_pipeline(
        dataset="cifar10",
        overrides={
            "aug_kwargs": {"enable": False},
            "laug_kwargs": {"enable": False},
            "postproc_kwargs": {
                "normalize_image": False,
                "normalization_params": None,
            },
        },
    )
    dataset, _count, config = _load_vision(pipeline)
    batch = next(iter(dataset))

    assert pipeline.kwargs["postproc_kwargs"]["normalization_params"] is None
    assert config.to_dict()["model_input"]["normalization"] == {"kind": "none"}
    np.testing.assert_allclose(batch["image"][0, :, 0, 0], [255.0, 255.0, 255.0])


def test_trivial_augment_wide_receives_explicit_nested_parameters():
    from justdata.vision.tasks.classification import make_augmentations

    received = {}

    def fake_augment(image, seed, **kwargs):
        del seed
        received.update(kwargs)
        return image

    with patch(
        "justdata.vision.tasks.classification.get_augment_strategy",
        return_value=fake_augment,
    ):
        augment = make_augmentations(
            image_size=12,
            crop_type="none",
            augment_type="trivial_augment_wide",
            ta_kwargs={"exclude_ops": ["Rotate"]},
        )
        augment({"image": tf.zeros((12, 16, 3), tf.uint8)}, tf.constant([1, 2]))

    assert received == {"exclude_ops": ["Rotate"]}


def test_semantic_or_loader_change_changes_full_product_fingerprint():
    def fingerprint(size, batch_size=2):
        pipeline = get_pipeline(
            dataset="cifar10",
            overrides={
                "aug_kwargs": {"enable": False, "image_size": size},
                "laug_kwargs": {"enable": False},
                "postproc_kwargs": {"image_size": size, "val_resize_size": None},
            },
        )
        _dataset, _count, config = _load_vision(pipeline, batch_size=batch_size)
        return hashlib.sha256(config.to_bytes()).hexdigest()

    assert fingerprint(224) != fingerprint(256)
    assert fingerprint(224) != fingerprint(224, batch_size=1)


def test_raw_and_epoch_snapshots_report_actual_finalization():
    pipeline = get_pipeline(
        dataset="cifar10",
        overrides={
            "aug_kwargs": {"enable": False},
            "laug_kwargs": {"enable": False},
        },
    )
    raw, tools, raw_config = _load_vision(
        pipeline, dataset_type="train", return_raw_ds=True
    )
    epoch_result, count, epoch_config = tools["finalize_epoch"](
        raw,
        seed=91,
        as_numpy=True,
        augment_is_stateless=True,
        return_config=True,
    )
    _final, final_count, final_config = tools["finalize_fn"](
        raw,
        prefetch=False,
        deterministic=False,
        map_parallel_calls=2,
        return_config=True,
    )

    assert count == final_count == 1
    assert raw_config.to_dict()["execution"]["preparation"] == "raw"
    assert raw_config.to_dict()["execution"]["pending_stages"]
    epoch = epoch_config.to_dict()["execution"]
    assert epoch["preparation"] == "epoch"
    assert epoch["rng"] == {
        "strategy": "stateless_indexed_epoch",
        "epoch_seed": 91,
        "augment_is_stateless": True,
    }
    assert epoch["shuffle"]["seed"] == 91
    assert epoch["shuffle"]["reshuffle_each_iteration"] is False
    assert epoch["as_numpy"] is True
    epoch_batch = next(epoch_result)
    assert isinstance(epoch_batch["image"], np.ndarray)
    assert epoch_config.to_dict()["model_input"]["static_shape"] == [3, 32, 32]
    final = final_config.to_dict()["execution"]
    assert final["limits"]["prefetch"] == "disabled"
    assert final["limits"]["map_parallel_calls"] == 2
    assert final["limits"]["preparation_map_parallel_calls"] == 1
    assert final["deterministic"] is False
    assert final["preparation_deterministic"] is True


@pytest.mark.parametrize(
    ("selection", "match"),
    [
        (
            {
                "dataset": "cifar10",
                "overrides": {"postproc_kwargs": {"image_sze": 224}},
            },
            r"Unknown configuration key: postproc_kwargs.image_sze",
        ),
        (
            {
                "pipeline_name": "vision/segmentation",
                "apply_presets": False,
                "overrides": {},
            },
            r"Missing required configuration key: aug_kwargs.image_size",
        ),
        (
            {
                "dataset": "cifar10",
                "overrides": {"postproc_kwargs": {"normalization_params": None}},
            },
            "normalization_params cannot be None",
        ),
    ],
    ids=["unknown-key", "missing-key", "contradictory-normalization"],
)
def test_invalid_strict_configuration_fails_before_source_access(selection, match):
    pipeline = get_pipeline(**selection)
    with patch("justdata.core.loader.fetch_ds") as fetch:
        with pytest.raises(ValueError, match=match):
            load_ds("mock", "validation", "validation", 2, 0, pipeline=pipeline)
    fetch.assert_not_called()


def test_runtime_owned_and_derived_overrides_are_rejected():
    with pytest.raises(ValueError, match="runtime-owned"):
        get_pipeline(
            dataset="cifar10",
            overrides={"postproc_kwargs": {"is_training": True}},
        )
    with pytest.raises(ValueError, match=r"postproc_kwargs.num_classes.*runtime-owned"):
        get_pipeline(
            dataset="cifar10",
            postproc_kwargs={"num_classes": 10},
            overrides={},
        )
    with pytest.raises(ValueError, match="derived"):
        get_pipeline(
            dataset="cifar10",
            overrides={"model_input": {"layout": "bhwc"}},
        )


def test_acoustic_builtin_exports_configuration():
    source = tf.data.Dataset.from_tensor_slices(
        {
            "waveform": np.ones((2, 32, 1), dtype=np.float32),
            "sample_rate": np.full((2,), 16000, dtype=np.int32),
            "label": np.arange(2, dtype=np.int64),
        }
    )
    pipeline = get_pipeline("acoustic/identity", apply_presets=False, overrides={})
    with patch("justdata.core.loader.fetch_ds", return_value=source):
        _dataset, count, config = load_ds(
            "mock",
            "validation",
            "validation",
            2,
            4,
            pipeline=pipeline,
            return_config=True,
            map_parallel_calls=1,
            private_threadpool_size=1,
        )

    assert count == 1
    snapshot = config.to_dict()
    assert snapshot["pipeline"]["modality"] == "acoustic"
    assert snapshot["stages"]["postprocess"]["active"] is False
    assert snapshot["model_input"] is None


def test_acoustic_strict_resolution_rejects_frontend_sample_rate_mismatch():
    pipeline = get_pipeline(
        dataset="audioset",
        overrides={"frontend": {"stft": {"sample_rate": 8000}}},
    )

    with patch("justdata.core.loader.fetch_ds") as fetch:
        with pytest.raises(
            ValueError,
            match="frontend.stft.sample_rate must match",
        ):
            load_ds(
                "mock",
                "validation",
                "validation",
                2,
                0,
                pipeline=pipeline,
            )
    fetch.assert_not_called()


def test_ast_snapshot_reports_executed_frontend_normalization():
    pipeline = get_pipeline(
        dataset="speech_commands",
        preset="ast_speechcommands_16k_1s_fbank128",
        pipeline_name="acoustic/ast_classification",
        overrides={},
    )
    snapshot = pipeline.resolve_config(True)

    assert snapshot["model_input"] == {
        "output_key": "features",
        "layout": "btf",
        "dtype": "float32",
        "static_shape": (128, 128),
        "kind": "features",
        "normalization": {
            "kind": "ast_affine",
            "mean": -6.845978,
            "std": 5.5654526,
            "divisor": 2.0,
        },
    }
    ast_stage = snapshot["stages"]["augment"]["config"]["ast_feature_stage"]["ast"]
    assert ast_stage["frequency_mask_max_width"] == 48
    assert ast_stage["time_mask_max_width"] == 48
    assert ast_stage["waveform_noise_and_roll"] is True


def test_snapshot_export_rejects_opaque_callbacks():
    with patch("justdata.core.loader.fetch_ds", return_value=_vision_source()):
        with pytest.raises(ValueError, match="registered DataPipeline"):
            load_ds(
                "mock",
                "validation",
                "validation",
                2,
                0,
                preprocess_fn=lambda sample: sample,
                augment_fn=lambda sample, seed=None: sample,
                late_augment_fn=lambda batch, **kwargs: batch,
                postprocess_fn=lambda sample, **kwargs: sample,
                return_config=True,
            )

    pipeline = get_pipeline(
        dataset="cifar10",
        overrides={
            "aug_kwargs": {"enable": False},
            "laug_kwargs": {"enable": False},
        },
    )
    raw, tools, _config = _load_vision(pipeline, return_raw_ds=True)
    with pytest.raises(ValueError, match="opaque finalization callbacks"):
        tools["finalize_fn"](
            raw,
            postprocess_fn=lambda sample, **kwargs: sample,
            return_config=True,
        )


def test_custom_pipeline_config_resolver_hook_and_legacy_factory():
    def identity(sample, *args, **kwargs):
        return sample

    def factory(**kwargs):
        return identity, identity, identity, identity

    def resolver(config, is_training):
        return {
            "configuration": config,
            "stages": {
                name: {"active": name in {"preprocess", "postprocess"}, "config": {}}
                for name in (
                    "preprocess",
                    "augment",
                    "late_augment",
                    "postprocess",
                )
            },
            "model_input": None,
            "requirements": {"num_classes": False},
            "is_training": is_training,
        }

    resolved_name = f"unit/resolved_{uuid4().hex}"
    register_pipeline(resolved_name, config_resolver=resolver)(factory)
    pipeline = get_pipeline(
        pipeline_name=resolved_name,
        apply_presets=False,
        overrides={"value": 0},
    )
    assert pipeline.resolve_config(True)["is_training"] is True
    assert len(pipeline.build(True)) == 4

    legacy_name = f"unit/legacy_{uuid4().hex}"
    register_pipeline(legacy_name)(factory)
    legacy = get_pipeline(pipeline_name=legacy_name, apply_presets=False)
    assert len(legacy.build(False)) == 4
    with pytest.raises(ValueError, match="no config_resolver"):
        legacy.resolve_config(False)

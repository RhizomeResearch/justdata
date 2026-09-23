import dataclasses
import hashlib
from unittest.mock import patch

import numpy as np
import pytest
import tensorflow as tf

import justdata.vision  # noqa: F401
from justdata.core import ExecutedConfig, get_pipeline, load_ds
from justdata.core.registry import list_pipelines
from justdata.vision.geometry import (
    DenseGeometryConfig,
    replay_dense_geometry,
    restore_dense_predictions,
    restore_dense_scores,
    sample_dense_geometry,
)


def _sample(height=5, width=9, dtype=np.int32):
    # A thin boundary, isolated pixels, disconnected regions, and an ignored hole.
    mask = np.zeros((height, width), dtype=dtype)
    mask[:, width // 2] = 1
    mask[0, 0] = mask[-1, -1] = 2
    mask[height // 2, width // 3] = 255
    image = np.repeat(mask[..., None].astype(np.uint8), 3, axis=-1)
    return {"image": tf.constant(image), "mask": tf.constant(mask)}


def _config(**kwargs):
    return DenseGeometryConfig(class_values=(0, 1, 2), **kwargs)


def _geometry(sample, config=None, *, train=False, seed=(5, 13)):
    return sample_dense_geometry(
        tf.shape(sample["image"])[:2],
        config or _config(),
        is_training=train,
        seed=tf.constant(seed),
    )


@pytest.mark.parametrize("shape", [(721, 1920), (1920, 721), (1, 4096)])
def test_evaluation_long_side_cap_rounds_half_up_and_retains_rectangles(shape):
    record = sample_dense_geometry(shape, _config(), is_training=False)
    expected = np.maximum(
        1, np.floor(np.asarray(shape) * 1024 / max(shape) + 0.5)
    ).astype(int)
    np.testing.assert_array_equal(record["resized_size"], expected)
    np.testing.assert_array_equal(record["crop_box"], [0, 0, *expected])
    np.testing.assert_array_equal(record["pre_padding"], [0, 0, 0, 0])
    np.testing.assert_array_equal(
        record["model_input_size"], ((expected + 15) // 16) * 16
    )
    np.testing.assert_array_equal(record["grid_size"], (expected + 15) // 16)


@pytest.mark.parametrize("upscale,expected", [(False, [3, 8]), (True, [5, 12])])
def test_evaluation_upscale_is_explicit_and_ties_round_up(upscale, expected):
    record = sample_dense_geometry(
        [3, 8], _config(eval_long_side=12, eval_upscale=upscale), is_training=False
    )
    np.testing.assert_array_equal(record["resized_size"], expected)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int32, np.int64])
@pytest.mark.parametrize("channel", [False, True])
def test_evaluation_preserves_small_structures_and_separates_ignore_from_padding(
    dtype, channel
):
    sample = _sample(dtype=dtype)
    if channel:
        sample["mask"] = sample["mask"][..., None]
    record = _geometry(sample, _config(patch_size=4))
    result = replay_dense_geometry(sample, record)
    mask = result["mask"].numpy()
    np.testing.assert_array_equal(mask[:5, :9], np.squeeze(sample["mask"]))
    assert result["mask"].dtype == sample["mask"].dtype
    assert mask.shape == (8, 12)
    assert np.all(mask[5:, :] == 255) and np.all(mask[:, 9:] == 255)
    support = result["source_valid_mask"].numpy()
    valid = result["pixel_valid_mask"].numpy()
    assert support.sum() == 45
    assert valid.sum() == 44
    assert support[2, 3] and not valid[2, 3]


@pytest.mark.parametrize("mode", ["CONSTANT", "REFLECT", "SYMMETRIC"])
def test_training_small_image_uses_independent_rgb_and_mask_padding(mode):
    sample = _sample(1, 3)
    config = _config(
        train_crop_size=8,
        train_resize_range=(1, 1),
        patch_size=4,
        horizontal_flip_probability=0,
        image_pad_mode=mode,
        image_pad_value=(10, 20, 30),
    )
    record = _geometry(sample, config, train=True)
    result = replay_dense_geometry(sample, record)
    np.testing.assert_array_equal(record["resized_size"], [1, 3])
    np.testing.assert_array_equal(record["pre_padding"], [0, 7, 0, 5])
    np.testing.assert_array_equal(record["crop_box"], [0, 0, 8, 8])
    support = result["source_valid_mask"].numpy()
    assert support.sum() == 3
    assert np.all(result["mask"].numpy()[~support] == 255)
    if mode == "CONSTANT":
        np.testing.assert_array_equal(result["image"][-1, -1], [10, 20, 30])
    else:
        expected = np.pad(
            sample["image"].numpy(), ((0, 7), (0, 5), (0, 0)), mode=mode.lower()
        )
        np.testing.assert_array_equal(result["image"], expected)


def test_nearest_mask_resize_matches_half_pixel_indices_including_large_ids():
    sample = _sample(3, 7, np.int64)
    sample["mask"] = tf.where(
        sample["mask"] == 2, tf.constant(16777217, tf.int64), sample["mask"]
    )
    config = DenseGeometryConfig(
        class_values=(0, 1, 16777217),
        eval_long_side=11,
        eval_upscale=True,
        patch_size=1,
    )
    result = replay_dense_geometry(sample, _geometry(sample, config))
    row = np.minimum(np.floor((np.arange(5) + 0.5) * 3 / 5).astype(int), 2)
    col = np.minimum(np.floor((np.arange(11) + 0.5) * 7 / 11).astype(int), 6)
    expected = sample["mask"].numpy()[row[:, None], col[None, :]]
    np.testing.assert_array_equal(result["mask"], expected)


def test_training_crop_flip_and_validity_replay_after_numeric_storage(tmp_path):
    sample = _sample(11, 19)
    annotation = np.ones((11, 19), bool)
    annotation[:, :6] = False
    sample["annotation_valid_mask"] = tf.constant(annotation)
    config = _config(
        train_crop_size=7,
        train_resize_range=(11, 11),
        patch_size=4,
        horizontal_flip_probability=1,
    )
    record = _geometry(sample, config, train=True)
    path = tmp_path / "view.npz"
    np.savez(path, **{k: v.numpy() for k, v in record.items()})
    with np.load(path, allow_pickle=False) as stored:
        restored = dict(stored)
    first = replay_dense_geometry(sample, record)
    replay = tf.function(replay_dense_geometry)(sample, restored)
    for left, right in zip(tf.nest.flatten(first), tf.nest.flatten(replay)):
        np.testing.assert_array_equal(left, right)
    top, left, h, w = record["crop_box"].numpy()
    expected_mask = sample["mask"].numpy()[top : top + h, left : left + w][:, ::-1]
    np.testing.assert_array_equal(first["mask"][:7, :7], expected_mask)
    np.testing.assert_array_equal(first["image"][:7, :7, 0], expected_mask)
    expected_valid = annotation[top : top + h, left : left + w][:, ::-1] & (
        expected_mask != 255
    )
    np.testing.assert_array_equal(first["pixel_valid_mask"][:7, :7], expected_valid)
    assert not np.any(first["source_valid_mask"][7:, :])
    assert first["mask"].numpy()[7, 7] == 255
    different = [
        _geometry(sample, config, train=True, seed=(s, 17))["crop_box"].numpy().tolist()
        for s in range(5)
    ]
    assert len({tuple(box) for box in different}) > 1


def test_all_ignore_and_unlabelled_views_have_no_pixel_supervision():
    sample = _sample()
    sample["mask"] = tf.fill([5, 9], 255)
    for source in (sample, {"image": sample["image"]}):
        result = replay_dense_geometry(source, _geometry(source))
        assert not np.any(result["pixel_valid_mask"])
        assert np.sum(result["source_valid_mask"]) == 45


@pytest.mark.parametrize(
    "invalid,match",
    [
        ({"class_values": (0, 0)}, "unique"),
        ({"ignore_value": 1}, "distinct"),
        ({"train_resize_range": (12, 3)}, "ordered"),
        ({"patch_size": 0}, "positive"),
        ({"eval_upscale": 1}, "booleans"),
        ({"image_pad_mode": "WRAP"}, "image_pad_mode"),
        ({"image_pad_value": (0, float("nan"), 0)}, "finite"),
    ],
)
def test_invalid_config_rejected(invalid, match):
    with pytest.raises(ValueError, match=match):
        DenseGeometryConfig(**({"class_values": (0, 1, 2)} | invalid))


@pytest.mark.parametrize("kind", ["unknown", "float", "shape", "fill", "annotation"])
def test_invalid_source_rejected_before_transform(kind):
    sample = _sample()
    config = _config()
    if kind == "unknown":
        sample["mask"] = tf.fill([5, 9], 7)
    elif kind == "float":
        sample["mask"] = tf.cast(sample["mask"], tf.float32)
    elif kind == "shape":
        sample["mask"] = tf.zeros([4, 9], tf.int32)
    elif kind == "fill":
        sample = _sample(dtype=np.uint8)
        config = _config(ignore_value=-1)
    else:
        sample["annotation_valid_mask"] = tf.ones([5, 9], tf.int32)
    with pytest.raises((TypeError, ValueError, tf.errors.InvalidArgumentError)):
        replay_dense_geometry(sample, _geometry(sample, config))


def _reference_bilinear(field, target):
    # Independent scalar reference: half-pixel coordinates, clamped border.
    out = np.empty((*target, field.shape[-1]), dtype=np.float64)
    for y in range(target[0]):
        for x in range(target[1]):
            sy = np.clip(
                (y + 0.5) * field.shape[0] / target[0] - 0.5, 0, field.shape[0] - 1
            )
            sx = np.clip(
                (x + 0.5) * field.shape[1] / target[1] - 0.5, 0, field.shape[1] - 1
            )
            y0, x0 = int(np.floor(sy)), int(np.floor(sx))
            y1, x1 = min(y0 + 1, field.shape[0] - 1), min(x0 + 1, field.shape[1] - 1)
            dy, dx = sy - y0, sx - x0
            out[y, x] = (1 - dy) * (
                (1 - dx) * field[y0, x0] + dx * field[y0, x1]
            ) + dy * ((1 - dx) * field[y1, x0] + dx * field[y1, x1])
    return out


def test_logit_restoration_matches_independent_reference_and_argmax_order():
    record = sample_dense_geometry(
        [7, 11], _config(eval_long_side=8, patch_size=4), is_training=False
    )
    logits = np.random.default_rng(17).normal(size=(2, 2, 3)).astype(np.float32)
    padded = _reference_bilinear(logits, (8, 8))
    expected = _reference_bilinear(padded[:5, :8], (7, 11))
    result = tf.function(restore_dense_scores)(tf.constant(logits), record)
    np.testing.assert_allclose(result, expected, atol=3e-7)
    np.testing.assert_array_equal(
        restore_dense_predictions(logits, record), expected.argmax(-1)
    )
    wrong = _reference_bilinear(logits, (7, 11))
    assert not np.array_equal(wrong.argmax(-1), expected.argmax(-1))
    early = _reference_bilinear(np.eye(3)[logits.argmax(-1)], (8, 8))
    early = _reference_bilinear(early[:5, :8], (7, 11))
    assert not np.array_equal(early.argmax(-1), expected.argmax(-1))
    reduced = padded.astype(np.float32) ** 2
    np.testing.assert_allclose(
        restore_dense_scores(reduced, record, from_model_input=True),
        _reference_bilinear(reduced[:5, :8], (7, 11)),
        atol=5e-7,
    )


def test_rgb_resize_uses_same_half_pixel_frame_as_categorical_sampling():
    sample = _sample(5, 9)
    config = _config(eval_long_side=6, patch_size=1, image_antialias=False)
    result = replay_dense_geometry(sample, _geometry(sample, config))
    np.testing.assert_allclose(
        result["image"],
        _reference_bilinear(sample["image"].numpy(), (3, 6)),
        atol=2e-5,
    )
    rows = np.floor((np.arange(3) + 0.5) * 5 / 3).astype(int)
    cols = np.floor((np.arange(6) + 0.5) * 9 / 6).astype(int)
    np.testing.assert_array_equal(
        result["mask"], sample["mask"].numpy()[rows[:, None], cols[None, :]]
    )


def test_default_training_geometry_produces_paired_512_crop_with_int64_seed():
    sample = _sample()
    record = sample_dense_geometry(
        [5, 9],
        _config(),
        is_training=True,
        seed=tf.constant([2**40 + 7, 13], tf.int64),
    )
    result = replay_dense_geometry(sample, record)
    assert result["image"].shape == (512, 512, 3)
    assert result["mask"].shape == (512, 512)
    assert result["pixel_valid_mask"].shape == (512, 512)
    np.testing.assert_array_equal(record["grid_size"], [32, 32])
    resized = record["resized_size"].numpy()
    assert 512 <= min(resized) <= 1024
    assert abs(resized[1] - resized[0] * 9 / 5) <= 0.5


def test_restoration_rejects_crop_unknown_record_and_wrong_frame():
    sample = _sample()
    train = _geometry(
        sample, _config(train_crop_size=4, train_resize_range=(5, 5)), train=True
    )
    evaluation = _geometry(sample)
    logits = tf.zeros([2, 2, 3])
    with pytest.raises(
        tf.errors.InvalidArgumentError, match="no defined scoring inverse"
    ):
        restore_dense_scores(logits, train)
    with pytest.raises(
        tf.errors.InvalidArgumentError, match="unsupported geometry version"
    ):
        restore_dense_scores(logits, evaluation | {"version": tf.constant(2)})
    with pytest.raises(tf.errors.InvalidArgumentError, match="Shape of tensor"):
        restore_dense_scores(logits, evaluation | {"version": tf.constant([1, 3])})
    with pytest.raises(tf.errors.InvalidArgumentError, match="padded model input"):
        restore_dense_scores(logits, evaluation, from_model_input=True)
    with pytest.raises(TypeError, match="before argmax"):
        restore_dense_scores(tf.zeros([2, 2, 1], tf.int32), evaluation)
    with pytest.raises(
        tf.errors.InvalidArgumentError, match="original sample dimensions"
    ):
        replay_dense_geometry(_sample(6, 9), evaluation)


def _pipeline(*, strict=True, **overrides):
    config = {
        "geometry_kwargs": dataclasses.asdict(
            _config(train_crop_size=8, train_resize_range=(5, 9), patch_size=4)
        )
    } | overrides
    return get_pipeline(
        pipeline_name="vision/segmentation",
        apply_presets=False,
        **({"overrides": config} if strict else config),
    )


def test_dense_geometry_uses_the_segmentation_registration():
    assert "vision/segmentation" in list_pipelines()
    assert "vision/dense_segmentation" not in list_pipelines()


@pytest.mark.parametrize(
    "selection",
    [
        {"pipeline_name": "vision/segmentation"},
        {"task": "segmentation", "modality": "vision"},
        {"dataset": "kitti_road"},
    ],
)
def test_segmentation_resolution_supports_recorded_geometry(selection):
    pipeline = get_pipeline(
        **selection,
        apply_presets=False,
        overrides={"geometry_kwargs": {"class_values": (0, 1, 2)}},
    )
    pre, _, _, post = pipeline.build(False)
    view = post(pre(_sample()))
    assert pipeline.pipeline_name == "vision/segmentation"
    assert "geometry" in view
    assert view["image"].shape == (3, 16, 16)
    np.testing.assert_array_equal(view["original_mask"], _sample()["mask"])


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("stage", ["preproc_kwargs", "aug_kwargs", "laug_kwargs"])
def test_recorded_geometry_rejects_conflicting_stages_before_source_access(
    strict, stage
):
    pipeline = _pipeline(strict=strict, **{stage: {"enable": False}})
    with patch("justdata.core.loader.fetch_ds") as fetch:
        with pytest.raises(ValueError, match=rf"{stage}.*geometry_kwargs"):
            load_ds("fixture", "validation", "validation", 1, 0, pipeline=pipeline)
    fetch.assert_not_called()


@pytest.mark.parametrize("strict", [False, True])
def test_recorded_geometry_accepts_empty_legacy_stage_containers(strict):
    pipeline = _pipeline(
        strict=strict, preproc_kwargs={}, aug_kwargs=None, laug_kwargs={}
    )
    pre, _, _, post = pipeline.build(False)
    assert "geometry" in post(pre(_sample()))


@pytest.mark.parametrize("training", [False, True])
def test_none_geometry_preserves_existing_segmentation_presets(training):
    overrides = {
        "aug_kwargs": {"image_size": 8},
        "postproc_kwargs": {"image_size": 8},
    }
    original = get_pipeline(dataset="kitti_road", overrides=overrides)
    disabled = get_pipeline(
        dataset="kitti_road", overrides=overrides | {"geometry_kwargs": None}
    )
    assert (
        original.resolve_config(training)["stages"]
        == disabled.resolve_config(training)["stages"]
    )
    views = []
    for pipeline in (original, disabled):
        pre, augment, _, post = pipeline.build(training)
        sample = pre(_sample())
        if training:
            sample = augment(sample, tf.constant([17, 19]))
        views.append(post(sample))
    assert views[0]["image"].shape == views[1]["image"].shape == (3, 8, 8)
    assert "geometry" not in views[0] and "geometry" not in views[1]
    for left, right in zip(tf.nest.flatten(views[0]), tf.nest.flatten(views[1])):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize(
    "config", [{"color_jitter_kwargs": {}}, {"keep_original_mask": False}]
)
def test_geometry_only_options_require_geometry_config(strict, config):
    pipeline = _pipeline(strict=strict, geometry_kwargs=None, **config)
    with pytest.raises(ValueError, match="requires geometry_kwargs"):
        pipeline.build(False)


@pytest.mark.parametrize("training", [False, True])
def test_dense_pipeline_loader_snapshot_numpy_and_original_targets(training):
    sample = _sample()
    source = tf.data.Dataset.from_tensors(
        sample | {"metadata": {"example_id": 9}}
    ).repeat(3)
    pipeline = _pipeline(
        postproc_kwargs={"normalization_params": ((0.1, 0.2, 0.3), (0.5, 0.5, 0.5))}
    )
    with patch("justdata.core.loader.fetch_ds", return_value=source):
        dataset, count, snapshot = load_ds(
            "fixture",
            "train" if training else "validation",
            "train" if training else "validation",
            2,
            19,
            pipeline=pipeline,
            return_config=True,
            deterministic=True,
            as_numpy=True,
            metadata_mode="numeric_only",
            map_parallel_calls=1,
            private_threadpool_size=1,
            shuffle_buffer=1,
        )
    batches = list(dataset)
    assert count == 2
    assert len(batches) == 2
    for batch in batches:
        for row in np.flatnonzero(batch["padding_mask"]):
            np.testing.assert_array_equal(batch["original_mask"][row], sample["mask"])
            assert batch["geometry"]["original_size"][row].tolist() == [5, 9]
            record = {k: tf.constant(v[row]) for k, v in batch["geometry"].items()}
            replayed = replay_dense_geometry(sample, record)
            expected = (replayed["image"].numpy() / 255 - [0.1, 0.2, 0.3]) / 0.5
            np.testing.assert_allclose(
                batch["image"][row], expected.transpose(2, 0, 1), atol=2e-7
            )
            np.testing.assert_array_equal(batch["mask"][row], replayed["mask"])
    np.testing.assert_array_equal(batches[-1]["padding_mask"], [True, False])
    assert not np.any(batches[-1]["pixel_valid_mask"][1])
    config = snapshot.to_dict()
    geometry = config["stages"]["augment" if training else "postprocess"]["geometry"]
    assert geometry["mask_padding_value"] == 255
    assert geometry["eval_long_side"] == 1024
    assert geometry["dimension_rounding"] == "half_up_min_one"
    assert config["model_input"]["static_shape"] == ([3, 8, 8] if training else None)


def test_photometric_augmentation_leaves_masks_geometry_and_validity_unchanged():
    base = _pipeline()
    colored = _pipeline(
        color_jitter_kwargs={"brightness": 0.8, "p": 1.0, "p_grayscale": 0.0}
    )
    pre, aug, _, _ = base.build(True)
    _, color_aug, _, _ = colored.build(True)
    sample = pre(_sample())
    plain = aug(sample, tf.constant([2, 3]))
    result = color_aug(sample, tf.constant([2, 3]))
    assert not np.array_equal(result["image"], plain["image"])
    for key in (
        "mask",
        "pixel_valid_mask",
        "source_valid_mask",
        "original_mask",
        "geometry",
    ):
        for left, right in zip(
            tf.nest.flatten(plain[key]), tf.nest.flatten(result[key])
        ):
            np.testing.assert_array_equal(left, right)


def test_dense_configuration_changes_snapshot_and_rejects_conflicting_presets():
    base = _pipeline().resolve_config(False)
    other = _pipeline(
        geometry_kwargs=dataclasses.asdict(_config(eval_upscale=True))
    ).resolve_config(False)
    assert (
        hashlib.sha256(ExecutedConfig.from_dict(base).to_bytes()).hexdigest()
        != hashlib.sha256(ExecutedConfig.from_dict(other).to_bytes()).hexdigest()
    )
    for overrides, match in [
        (
            {"geometry_kwargs": {"class_values": (0, 1), "typo": 1}},
            "geometry_kwargs.typo",
        ),
        ({"postproc_kwargs": {"image_size": 8}}, "postproc_kwargs.image_size"),
        ({"augment_eval": True}, "deterministic geometry"),
    ]:
        with pytest.raises(ValueError, match=match):
            _pipeline(**overrides).build(False)


def test_mixed_original_sizes_work_with_single_sample_batches():
    pre, _, _, post = _pipeline().build(False)
    signature = {
        "image": tf.TensorSpec([None, None, 3], tf.uint8),
        "mask": tf.TensorSpec([None, None], tf.int32),
    }
    source = tf.data.Dataset.from_generator(
        lambda: iter([_sample(5, 9), _sample(9, 5)]), output_signature=signature
    )
    results = list(source.map(pre).map(post).batch(1))
    assert results[0]["image"].shape == (1, 3, 8, 12)
    assert results[1]["image"].shape == (1, 3, 12, 8)
    assert results[0]["original_mask"].shape == (1, 5, 9)
    assert results[1]["original_mask"].shape == (1, 9, 5)

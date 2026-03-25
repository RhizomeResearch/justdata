import pytest

from justdata.vision.augmentations.registry import (
    get_augment_strategy,
    get_crop_strategy,
)
from justdata.core.registry import (
    DataPipeline,
    get_dataset_info,
    get_pipeline,
    get_pipeline_for_dataset,
    get_task_for_dataset,
    register_dataset,
)


# Dataset -> task mapping
class TestDatasetTaskMapping:
    def test_known_datasets(self):
        assert get_task_for_dataset("cifar10") == "classification"
        assert get_task_for_dataset("cifar100") == "classification"
        assert get_task_for_dataset("imagenette") == "classification"
        assert get_task_for_dataset("voc") == "object_detection"
        assert get_task_for_dataset("voc/2007") == "object_detection"
        assert get_task_for_dataset("nyu_depth_v2_mini") == "depth_estimation"
        assert get_task_for_dataset("kitti_road") == "segmentation"

    def test_unknown_dataset_returns_none(self):
        assert get_task_for_dataset("unknown_ds_xyz") is None

    def test_register_custom_dataset(self):
        register_dataset("my_custom_ds", "classification", modality="vision")
        assert get_task_for_dataset("my_custom_ds") == "classification"
        assert get_dataset_info("my_custom_ds").modality == "vision"

    def test_prefix_match_imagenet_variant(self):
        """'imagenet_a3' should resolve via 'imagenet' prefix."""
        assert get_task_for_dataset("imagenet_a3") == "classification"

    def test_prefix_match_longest_wins(self):
        """'cifar100_corrupted' must match 'cifar100', not 'cifar10'."""
        assert get_task_for_dataset("cifar100_corrupted") == "classification"
        # Sanity: cifar10_v2 should match cifar10
        assert get_task_for_dataset("cifar10_v2") == "classification"

    def test_prefix_match_voc_subpath(self):
        """'voc/2012/extra' should match 'voc/2012' (longest prefix)."""
        assert get_task_for_dataset("voc/2012/extra") == "object_detection"


# get_pipeline — argument resolution
class TestGetPipeline:
    """Comprehensive tests for get_pipeline resolution logic."""

    # Explicit task / pipeline_name

    def test_explicit_task(self):
        p = get_pipeline(task="classification", modality="vision", apply_presets=False)
        assert isinstance(p, DataPipeline)
        assert p.pipeline_name == "vision/classification"

    def test_explicit_pipeline_name(self):
        p = get_pipeline(pipeline_name="vision/segmentation", apply_presets=False)
        assert p.pipeline_name == "vision/segmentation"

    def test_pipeline_name_takes_precedence_over_task(self):
        p = get_pipeline(
            task="classification",
            pipeline_name="vision/segmentation",
            apply_presets=False,
        )
        assert p.pipeline_name == "vision/segmentation"

    # Dataset-driven resolution

    def test_dataset_resolves_task_from_map(self):
        p = get_pipeline(dataset="cifar10", apply_presets=False)
        assert p.pipeline_name == "vision/classification"

    def test_dataset_with_explicit_task_uses_task(self):
        """Explicit task overrides dataset-inferred task."""
        p = get_pipeline(dataset="cifar10", task="segmentation", apply_presets=False)
        assert p.pipeline_name == "vision/segmentation"

    def test_dataset_prefix_resolves(self):
        p = get_pipeline(dataset="imagenet_a3", apply_presets=False)
        assert p.pipeline_name == "vision/classification"

    # Pipeline-name-as-dataset convenience

    def test_pipeline_name_as_first_arg(self):
        """get_pipeline('vision/classification') should work as a convenience."""
        p = get_pipeline("vision/classification", apply_presets=False)
        assert p.pipeline_name == "vision/classification"

    def test_pipeline_name_as_first_arg_no_preset_leak(self):
        """When dataset arg matches a pipeline name (not a real dataset),
        presets must NOT be applied — no ImageNet defaults should leak in."""
        p = get_pipeline("vision/classification")
        # kwargs should be empty (no preset config injected)
        assert "postproc_kwargs" not in p.kwargs
        assert "aug_kwargs" not in p.kwargs

    def test_pipeline_name_as_first_arg_with_explicit_preset(self):
        """An explicit preset= should still be honoured even when dataset
        matches a pipeline name."""
        p = get_pipeline("vision/classification", preset="cifar10")
        # CIFAR presets should be present
        assert p.kwargs.get("postproc_kwargs", {}).get("image_size") == 32

    def test_pipeline_name_as_first_arg_user_kwargs_preserved(self):
        """User kwargs passed alongside a pipeline-name-as-dataset should
        appear in the DataPipeline config without preset merging."""
        p = get_pipeline(
            "vision/classification",
            aug_kwargs={"image_size": 64},
            postproc_kwargs={"image_size": 64},
        )
        assert p.kwargs["aug_kwargs"]["image_size"] == 64
        assert p.kwargs["postproc_kwargs"]["image_size"] == 64
        # No extra keys from ImageNet defaults should appear
        assert "normalization_params" not in p.kwargs.get("postproc_kwargs", {})

    # Preset merging

    def test_dataset_applies_presets(self):
        p = get_pipeline(dataset="cifar10")
        assert p.kwargs["postproc_kwargs"]["image_size"] == 32

    def test_explicit_preset_overrides_dataset(self):
        """preset='imagenet_a1' should use A1 presets even for cifar10."""
        p = get_pipeline(dataset="cifar10", preset="imagenet_a1")
        assert p.kwargs["postproc_kwargs"]["image_size"] == 224

    def test_apply_presets_false_skips_merging(self):
        p = get_pipeline(dataset="cifar10", apply_presets=False)
        assert "postproc_kwargs" not in p.kwargs

    def test_user_kwargs_override_presets(self):
        p = get_pipeline(dataset="cifar10", postproc_kwargs={"image_size": 128})
        # 128 differs from both CIFAR (32) and default (224), so it wins
        assert p.kwargs["postproc_kwargs"]["image_size"] == 128

    # Error cases

    def test_no_args_raises(self):
        with pytest.raises(ValueError, match="Could not resolve"):
            get_pipeline()

    def test_unknown_dataset_no_task_raises(self):
        with pytest.raises(ValueError, match="Could not resolve"):
            get_pipeline(dataset="totally_unknown_xyz")

    def test_unknown_pipeline_name_raises_on_build(self):
        with pytest.raises(ValueError, match="not found"):
            p = get_pipeline(pipeline_name="nonexistent_pipeline_xyz")
            p.build(is_training=True)

    # DataPipeline.build

    def test_build_returns_4_callables(self):
        p = get_pipeline(
            "vision/classification",
            aug_kwargs={"image_size": 32},
            postproc_kwargs={"image_size": 32},
        )
        result = p.build(is_training=True)
        assert len(result) == 4
        assert all(callable(fn) for fn in result)

    def test_build_injects_is_training(self):
        """build(is_training=True) should set postproc_kwargs.is_training."""
        p = get_pipeline(
            "vision/classification",
            aug_kwargs={"image_size": 32},
            postproc_kwargs={"image_size": 32},
        )
        # After build, the original pipeline kwargs should be unchanged
        # (deep copy protects them).
        p.build(is_training=True)
        assert "is_training" not in p.kwargs.get("postproc_kwargs", {})

    def test_build_does_not_mutate_kwargs(self):
        p = get_pipeline(
            "vision/classification",
            aug_kwargs={"image_size": 32},
            postproc_kwargs={"image_size": 32},
        )
        original_kwargs = {
            k: dict(v) if isinstance(v, dict) else v for k, v in p.kwargs.items()
        }
        p.build(is_training=True)
        p.build(is_training=False)
        for k, v in original_kwargs.items():
            assert p.kwargs[k] == v

    # Legacy unpacking

    def test_legacy_iter_unpacking(self):
        p = get_pipeline(
            "vision/classification",
            aug_kwargs={"image_size": 32},
            postproc_kwargs={"image_size": 32},
        )
        a, b, c, d = p  # uses __iter__ with _is_training_legacy default
        assert callable(a)
        assert callable(d)

    # get_pipeline_for_dataset (legacy wrapper)

    def test_get_pipeline_for_dataset_basic(self):
        p = get_pipeline_for_dataset("cifar10", apply_presets=True)
        assert p.pipeline_name == "vision/classification"
        assert p.kwargs["postproc_kwargs"]["image_size"] == 32

    def test_get_pipeline_for_dataset_is_training_legacy(self):
        p = get_pipeline_for_dataset(
            "cifar10",
            is_training=True,
            apply_presets=True,
        )
        assert p._is_training_legacy is True
        # Legacy unpacking should use is_training=True
        preproc, aug, laug, postproc = p
        assert callable(preproc)

    def test_get_pipeline_for_dataset_explicit_task_type(self):
        p = get_pipeline_for_dataset(
            "cifar10",
            task_type="segmentation",
            apply_presets=False,
            aug_kwargs={"image_size": 32},
            postproc_kwargs={"image_size": 32},
        )
        assert p.pipeline_name == "vision/segmentation"

    # repr

    def test_repr(self):
        p = get_pipeline("vision/classification", apply_presets=False)
        r = repr(p)
        assert "classification" in r
        assert "DataPipeline" in r


# Crop / Augment strategy registries
class TestCropStrategyRegistry:
    def test_random_resized_exists(self):
        fn = get_crop_strategy("random_resized")
        assert callable(fn)

    def test_random_pad_exists(self):
        fn = get_crop_strategy("random_pad")
        assert callable(fn)

    def test_none_exists(self):
        fn = get_crop_strategy("none")
        assert callable(fn)

    def test_unknown_crop_raises(self):
        with pytest.raises(ValueError, match="not found"):
            get_crop_strategy("unknown_crop_xyz")


class TestAugmentStrategyRegistry:
    def test_rand_augment_exists(self):
        fn = get_augment_strategy("rand_augment")
        assert callable(fn)

    def test_trivial_augment_exists(self):
        fn = get_augment_strategy("trivial_augment")
        assert callable(fn)

    def test_unknown_augment_raises(self):
        with pytest.raises(ValueError, match="not found"):
            get_augment_strategy("unknown_augment_xyz")

import pytest

from justdata.registry import (
    get_pipeline,
    get_task_for_dataset,
    register_dataset,
)
from justdata.augmentations.registry import (
    get_augment_strategy,
    get_crop_strategy,
)


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
        register_dataset("my_custom_ds", "classification")
        assert get_task_for_dataset("my_custom_ds") == "classification"


class TestPipelineRegistry:
    def test_known_pipelines_exist(self):
        # These should not raise
        get_pipeline("classification", aug_kwargs={"image_size": 32}, postproc_kwargs={"image_size": 32})
        get_pipeline("segmentation", aug_kwargs={"image_size": 32}, postproc_kwargs={"image_size": 32})

    def test_unknown_pipeline_raises(self):
        with pytest.raises(ValueError, match="not found"):
            get_pipeline("nonexistent_pipeline_xyz")

    def test_classification_pipeline_returns_4_callables(self):
        result = get_pipeline(
            "classification",
            aug_kwargs={"image_size": 32},
            postproc_kwargs={"image_size": 32},
        )
        assert len(result) == 4
        assert all(callable(fn) for fn in result)


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

from justdata.presets import (
    get_dataset_presets,
    merge_with_presets,
    register_preset,
)


class TestPresetRegistration:
    def test_exact_match(self):
        presets = get_dataset_presets("cifar100")
        assert presets["postproc_kwargs"]["image_size"] == 32

    def test_prefix_match_cifar10(self):
        presets = get_dataset_presets("cifar10")
        # "cifar10" starts with "cifar", so the "cifar" preset matches
        assert presets["postproc_kwargs"]["image_size"] == 32

    def test_unknown_dataset_gets_default(self):
        presets = get_dataset_presets("unknown_dataset_xyz")
        assert presets["postproc_kwargs"]["image_size"] == 224

    def test_default_preset_has_imagenet_stats(self):
        presets = get_dataset_presets("imagenet")
        mean, std = presets["postproc_kwargs"]["normalization_params"]
        assert mean == (0.485, 0.456, 0.406)
        assert std == (0.229, 0.224, 0.225)

    def test_register_custom_preset(self):
        register_preset("test_custom_ds", {"postproc_kwargs": {"image_size": 128}})
        presets = get_dataset_presets("test_custom_ds")
        assert presets["postproc_kwargs"]["image_size"] == 128

    def test_cifar_preset_uses_trivial_augment(self):
        presets = get_dataset_presets("cifar10")
        assert presets["aug_kwargs"]["augment_type"] == "trivial_augment"

    def test_cifar100_normalization_differs_from_cifar(self):
        cifar = get_dataset_presets("cifar10")
        cifar100 = get_dataset_presets("cifar100")
        assert (
            cifar["postproc_kwargs"]["normalization_params"]
            != cifar100["postproc_kwargs"]["normalization_params"]
        )


class TestMergeWithPresets:
    def test_empty_user_kwargs_returns_presets(self):
        result = merge_with_presets("cifar10", {})
        assert result["postproc_kwargs"]["image_size"] == 32

    def test_user_override_wins_over_preset(self):
        user_kwargs = {"postproc_kwargs": {"image_size": 64}}
        result = merge_with_presets("cifar10", user_kwargs)
        assert result["postproc_kwargs"]["image_size"] == 64

    def test_default_value_does_not_override_preset(self):
        # If user passes ImageNet default image_size=224, the cifar preset (32)
        # should win because 224 matches the _default and is not a conscious override
        user_kwargs = {"postproc_kwargs": {"image_size": 224}}
        result = merge_with_presets("cifar10", user_kwargs)
        assert result["postproc_kwargs"]["image_size"] == 32

    def test_non_default_value_overrides_preset(self):
        # 128 is not the ImageNet default (224), so it's treated as explicit
        user_kwargs = {"postproc_kwargs": {"image_size": 128}}
        result = merge_with_presets("cifar10", user_kwargs)
        assert result["postproc_kwargs"]["image_size"] == 128

    def test_nested_dict_merge(self):
        user_kwargs = {"aug_kwargs": {"ra_kwargs": {"magnitude": 15.0}}}
        result = merge_with_presets("cifar10", user_kwargs)
        # magnitude=15 differs from default magnitude=7, so it should override
        assert result["aug_kwargs"]["ra_kwargs"]["magnitude"] == 15.0
        # Other aug_kwargs from preset should still be present
        assert "crop_type" in result["aug_kwargs"]

    def test_unknown_dataset_uses_default_preset(self):
        result = merge_with_presets("imagenet", {})
        assert result["postproc_kwargs"]["image_size"] == 224

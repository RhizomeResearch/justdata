from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from uuid import uuid4

import pytest

from justdata.core.presets import (
    get_dataset_presets as get_core_dataset_presets,
    get_resolved_preset as get_core_resolved_preset,
    register_preset as register_core_preset,
)
from justdata.vision.presets import (
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
        assert presets["aug_kwargs"]["augment_type"] == "trivial_augment_wide"

    def test_cifar100_normalization_differs_from_cifar(self):
        cifar = get_dataset_presets("cifar10")
        cifar100 = get_dataset_presets("cifar100")
        assert (
            cifar["postproc_kwargs"]["normalization_params"]
            != cifar100["postproc_kwargs"]["normalization_params"]
        )


class TestPresetRegistryIsolation:
    def test_registration_and_lookup_values_are_snapshots(self):
        name = f"unit_preset_{uuid4().hex}"
        config = {"nested": {"values": [1, 2]}}
        register_core_preset(name, config, modality="unit")
        original_hash = get_core_resolved_preset(name, modality="unit").hash()

        config["nested"]["values"].append(3)
        fetched = get_core_dataset_presets(name, modality="unit")
        fetched["nested"]["values"].append(4)
        resolved = get_core_resolved_preset(name, modality="unit")
        resolved.config["nested"]["values"].append(5)

        fresh = get_core_resolved_preset(name, modality="unit")
        assert fresh.config == {"nested": {"values": [1, 2]}}
        assert fresh.hash() == original_hash

    def test_duplicate_normalized_key_is_rejected_per_modality(self):
        name = f"unit_preset_{uuid4().hex}"
        register_core_preset(name.upper(), {"value": 1}, modality="unit")

        with pytest.raises(
            ValueError,
            match=rf"Preset '{name}' already registered for modality 'unit'",
        ):
            register_core_preset(name, {"value": 2}, modality="unit")

        register_core_preset(name, {"value": 3}, modality="other")
        assert get_core_dataset_presets(name, modality="other") == {"value": 3}

    def test_concurrent_duplicate_registration_has_one_winner(self):
        name = f"unit_preset_{uuid4().hex}"
        workers = 8
        barrier = Barrier(workers)

        def register_once(index):
            barrier.wait()
            try:
                register_core_preset(name, {"winner": index}, modality="threaded")
            except ValueError as exc:
                return str(exc)
            return None

        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(register_once, range(workers)))

        errors = [result for result in results if result is not None]
        assert len(errors) == workers - 1
        assert all(name in error and "threaded" in error for error in errors)
        assert get_core_dataset_presets(name, modality="threaded")["winner"] in range(
            workers
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

import tensorflow as tf

from justdata.adapters import (
    _default_adapter,
    get_adapter,
    register_adapter,
)


class TestDefaultAdapter:
    def test_identity(self, classification_sample):
        result = _default_adapter(classification_sample)
        assert result is classification_sample

    def test_all_keys_preserved(self):
        sample = {"image": tf.zeros([2, 2, 3]), "label": 1, "extra": "value"}
        result = _default_adapter(sample)
        assert set(result.keys()) == {"image", "label", "extra"}


class TestAdapterRegistry:
    def test_unregistered_dataset_returns_default(self):
        adapter = get_adapter("nonexistent_dataset_xyz")
        assert adapter is _default_adapter

    def test_register_and_retrieve(self):
        from justdata.adapters import _ADAPTERS

        # Clean up in case of re-runs
        _ADAPTERS.pop("test_ds_adapter", None)

        @register_adapter("test_ds_adapter")
        def my_adapter(sample):
            return {"image": sample["photo"], "label": sample["class"]}

        adapter = get_adapter("test_ds_adapter")
        sample = {"photo": tf.zeros([2, 2, 3]), "class": 5}
        result = adapter(sample)
        assert "image" in result
        assert result["label"] == 5

from __future__ import annotations

from typing import Any, Iterable, Mapping

import tensorflow as tf

from justdata.core.metadata import (
    MetadataSidecar,
    python_value as _python_value,
    stable_int64_hash,
)


class MetadataEncoder:
    """Encode common acoustic metadata into JAX/TF-friendly numeric tensors."""

    _FIELD_KEYS = {
        "device_id": ("device_id", "device"),
        "scene_id": ("scene_id", "scene"),
        "city_id": ("city_id", "city"),
        "split_id": ("split_id", "split"),
        "dataset_id": ("dataset_id", "dataset"),
    }

    def __init__(self) -> None:
        self.vocab: dict[str, dict[str, int]] = {
            field: {} for field in self._FIELD_KEYS
        }

    def fit(self, examples_or_vocab: Iterable[Mapping[str, Any]] | Mapping[str, Any]):
        if isinstance(examples_or_vocab, Mapping) and self._looks_like_vocab(
            examples_or_vocab
        ):
            self.vocab = {
                field: {str(value): int(i) for i, value in enumerate(values)}
                for field, values in examples_or_vocab.items()
                if field in self._FIELD_KEYS
            }
            for field in self._FIELD_KEYS:
                self.vocab.setdefault(field, {})
            return self

        examples = (
            [examples_or_vocab]
            if isinstance(examples_or_vocab, Mapping)
            else list(examples_or_vocab)
        )
        for field, keys in self._FIELD_KEYS.items():
            values: list[str] = []
            for example in examples:
                value = self._first_present(example, keys)
                if value is None or isinstance(
                    _python_value(value), (int, float, bool)
                ):
                    continue
                text = str(_python_value(value))
                if text not in values:
                    values.append(text)
            self.vocab[field] = {value: i for i, value in enumerate(values)}
        return self

    def encode(self, metadata: dict) -> dict[str, tf.Tensor]:
        encoded: dict[str, tf.Tensor] = {}

        for field, keys in self._FIELD_KEYS.items():
            value = self._first_present(metadata, keys)
            encoded[field] = tf.constant(
                self._encode_categorical(field, value),
                dtype=tf.int32,
            )

        device_value = self._first_present(metadata, self._FIELD_KEYS["device_id"])
        encoded["is_known_device"] = tf.constant(
            self._is_known("device_id", device_value),
            dtype=tf.bool,
        )

        source_id = metadata.get("source_id")
        encoded["source_id_hash"] = tf.constant(
            stable_int64_hash(source_id),
            dtype=tf.int64,
        )
        encoded["example_id"] = tf.constant(
            self._example_id(metadata),
            dtype=tf.int64,
        )
        return encoded

    def decode(self, encoded: dict) -> dict:
        decoded = {}
        for field in self._FIELD_KEYS:
            value = int(_python_value(encoded[field]))
            reverse = {idx: key for key, idx in self.vocab.get(field, {}).items()}
            decoded[field] = reverse.get(value, value)

        decoded["is_known_device"] = bool(_python_value(encoded["is_known_device"]))
        decoded["source_id_hash"] = int(_python_value(encoded["source_id_hash"]))
        decoded["example_id"] = int(_python_value(encoded["example_id"]))
        return decoded

    @staticmethod
    def _looks_like_vocab(value: Mapping[str, Any]) -> bool:
        return all(isinstance(v, (list, tuple, set)) for v in value.values())

    @staticmethod
    def _first_present(
        metadata: Mapping[str, Any], keys: tuple[str, ...]
    ) -> Any | None:
        for key in keys:
            if key in metadata:
                return metadata[key]
        return None

    def _encode_categorical(self, field: str, value: Any) -> int:
        value = _python_value(value)
        if value is None:
            return -1
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        return self.vocab.get(field, {}).get(str(value), -1)

    def _is_known(self, field: str, value: Any) -> bool:
        value = _python_value(value)
        if value is None:
            return False
        if isinstance(value, int):
            return True
        return str(value) in self.vocab.get(field, {})

    @staticmethod
    def _example_id(metadata: Mapping[str, Any]) -> int:
        if all(key in metadata for key in ("dataset", "split", "clip_id")):
            value = (
                f"{_python_value(metadata['dataset'])}::"
                f"{_python_value(metadata['split'])}::"
                f"{_python_value(metadata['clip_id'])}"
            )
            return stable_int64_hash(value)

        value = metadata.get("example_id")
        if isinstance(_python_value(value), int):
            return int(_python_value(value))
        return stable_int64_hash(value)


__all__ = [
    "MetadataEncoder",
    "MetadataSidecar",
    "stable_int64_hash",
]

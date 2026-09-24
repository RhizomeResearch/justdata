from justdata.acoustic.metadata import (
    MetadataEncoder,
    MetadataSidecar,
    stable_int64_hash,
)
from justdata.core import metadata as core_metadata


def test_acoustic_metadata_reexports_core_sidecar_and_hash():
    # Sidecar transport and its hash pins are tested in tests/test_loader_*.py.
    assert MetadataSidecar is core_metadata.MetadataSidecar
    assert stable_int64_hash is core_metadata.stable_int64_hash


def test_metadata_encoder_vocab_roundtrip():
    encoder = MetadataEncoder().fit(
        [
            {
                "device": "dev-a",
                "scene": "park",
                "city": "Paris",
                "split": "train",
                "dataset": "unit",
            }
        ]
    )

    encoded = encoder.encode(
        {
            "device": "dev-a",
            "scene": "park",
            "city": "Paris",
            "split": "train",
            "dataset": "unit",
            "clip_id": "clip-1",
            "source_id": "source-a",
        }
    )
    decoded = encoder.decode(encoded)

    assert set(encoded) == {
        "device_id",
        "scene_id",
        "city_id",
        "is_known_device",
        "split_id",
        "dataset_id",
        "example_id",
        "source_id_hash",
    }
    assert decoded["device_id"] == "dev-a"
    assert decoded["scene_id"] == "park"
    assert decoded["city_id"] == "Paris"
    assert decoded["split_id"] == "train"
    assert decoded["dataset_id"] == "unit"
    assert decoded["is_known_device"] is True
    assert decoded["example_id"] == stable_int64_hash("unit::train::clip-1")

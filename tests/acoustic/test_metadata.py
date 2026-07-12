from justdata.acoustic.metadata import (
    MetadataEncoder,
    MetadataSidecar,
    stable_int64_hash,
)
from justdata.core.metadata import MetadataSidecar as CoreMetadataSidecar
from justdata.core.metadata import stable_int64_hash as core_stable_int64_hash


def test_metadata_encoder_stable_hash():
    assert stable_int64_hash is core_stable_int64_hash
    assert stable_int64_hash("example-a") == 5862446126654077062
    assert stable_int64_hash("dataset::train::clip-a") == 1147561614405543418
    assert stable_int64_hash("dataset::train::clip-a") == stable_int64_hash(
        "dataset::train::clip-a"
    )
    assert stable_int64_hash("dataset::train::clip-a") != stable_int64_hash(
        "dataset::train::clip-b"
    )


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


def test_sidecar_writes_strings_by_example_id(tmp_path):
    assert MetadataSidecar is CoreMetadataSidecar
    path = tmp_path / "metadata.jsonl"
    sidecar = MetadataSidecar()
    sidecar.add(7, {"clip_id": "clip-a", "nested": {"city": "Paris"}})
    sidecar.write_jsonl(str(path))

    loaded = MetadataSidecar.read_jsonl(str(path))

    assert loaded.records[7]["clip_id"] == "clip-a"
    assert loaded.records[7]["nested"]["city"] == "Paris"

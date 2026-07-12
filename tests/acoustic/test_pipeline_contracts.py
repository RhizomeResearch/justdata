from dataclasses import dataclass
from unittest.mock import patch

import numpy as np
import pytest
import tensorflow as tf

import justdata.acoustic  # noqa: F401
from justdata.acoustic.adapters import acoustic_source_adapter, speech_commands_adapter
from justdata.acoustic.corruptions import apply_audio_corruption
from justdata.acoustic.registry import (
    get_audio_batch_augment,
    get_audio_spectrogram_augment,
    get_audio_waveform_augment,
    list_audio_batch_augments,
    list_audio_corruptions,
    list_audio_spectrogram_augments,
    list_audio_waveform_augments,
)
from justdata.acoustic.sources import load_huggingface_audio_splits
from justdata.core import get_pipeline, load_ds
from justdata.core.registry import list_pipelines


@dataclass(frozen=True)
class AcousticPipelineContract:
    name: str
    pipeline_name: str
    preset: str
    sample_rate: int
    output_key: str
    eval_shape: tuple[int, ...]
    label_shape: tuple[int, ...]
    train_shape: tuple[int, ...] | None = None


ACOUSTIC_PIPELINE_CONTRACTS = (
    AcousticPipelineContract("waveform", "acoustic/classification", "audio_default_16k_waveform", 16000, "waveform", (16000,), ()),
    AcousticPipelineContract("logmel", "acoustic/classification", "audio_default_32k_logmel64", 32000, "features", (101, 64), ()),
    AcousticPipelineContract("passt", "acoustic/classification", "dcase2025_task1_passt_32k_1s", 32000, "features", (1, 128, 101), (), (1, 124, 61)),
    AcousticPipelineContract("ast", "acoustic/ast_classification", "ast_speechcommands_16k_1s_fbank128", 16000, "features", (128, 128), (35,)),
    AcousticPipelineContract("dcase-efficientat", "acoustic/classification", "dcase2025_task1_efficientat_32k_1s", 32000, "features", (1, 128, 101), ()),
)


def _load_contract(contract, raw_ds, *, dataset_type, seed=29, as_numpy=False):
    pipeline = get_pipeline(
        pipeline_name=contract.pipeline_name,
        preset=contract.preset,
        modality="acoustic",
    )
    with patch("justdata.core.loader.fetch_ds", return_value=raw_ds):
        return load_ds(
            dataset_names_arg="synthetic",
            splits_arg=dataset_type,
            dataset_type=dataset_type,
            batch_size=2,
            seed=seed,
            num_classes=35,
            pipeline=pipeline,
            cache_dataset=False,
            deterministic=True,
            shuffle_buffer=1,
            metadata_mode="numeric_only",
            as_numpy=as_numpy,
        )


@pytest.mark.parametrize(
    "contract",
    ACOUSTIC_PIPELINE_CONTRACTS,
    ids=lambda case: f"acoustic-{case.name}",
)
@pytest.mark.parametrize("dataset_type", ["validation", "train"])
def test_acoustic_pipeline_contract_end_to_end(
    contract, dataset_type, make_synthetic_acoustic_ds
):
    raw_ds = make_synthetic_acoustic_ds(sample_rate=contract.sample_rate)
    ds, cardinality = _load_contract(contract, raw_ds, dataset_type=dataset_type)
    batches = list(ds)
    expected_shape = (
        contract.train_shape
        if dataset_type == "train" and contract.train_shape is not None
        else contract.eval_shape
    )

    assert int(cardinality.numpy()) == 2
    assert len(batches) == 2
    assert batches[0][contract.output_key].shape == (2, *expected_shape)
    assert batches[0][contract.output_key].dtype == tf.float32
    assert batches[0]["label"].shape == (2, *contract.label_shape)
    assert set(batches[0]["metadata"]) >= {"device_id"}
    assert "device_name" not in batches[0]["metadata"]
    np.testing.assert_array_equal(batches[-1]["padding_mask"], [True, False])


def test_acoustic_eval_contract_is_deterministic_and_supports_numpy(
    make_synthetic_acoustic_ds,
):
    contract = ACOUSTIC_PIPELINE_CONTRACTS[1]

    def snapshot():
        raw_ds = make_synthetic_acoustic_ds(sample_rate=contract.sample_rate)
        iterator, _ = _load_contract(
            contract, raw_ds, dataset_type="validation", as_numpy=True
        )
        return list(iterator)

    first = snapshot()
    second = snapshot()
    assert isinstance(first[0]["features"], np.ndarray)
    for left, right in zip(first, second):
        np.testing.assert_allclose(left["features"], right["features"])


@pytest.mark.parametrize(
    "pipeline_name",
    [name for name in list_pipelines() if name.startswith("acoustic/")],
)
@pytest.mark.parametrize("is_training", [False, True], ids=["eval", "train"])
def test_every_registered_acoustic_pipeline_builds(pipeline_name, is_training):
    preset = (
        "ast_speechcommands_16k_1s_fbank128"
        if pipeline_name == "acoustic/ast_classification"
        else "audio_default_16k_waveform"
    )
    functions = get_pipeline(
        pipeline_name=pipeline_name,
        preset=preset,
        modality="acoustic",
    ).build(is_training=is_training)
    assert len(functions) == 4
    assert all(callable(fn) for fn in functions)


_WAVEFORM_KWARGS = {
    "random_crop": {"target_length": 128},
    "rir_convolution": {"rir": [1.0, 0.25]},
    "speed_perturb": {"rates": [0.9]},
}


@pytest.mark.parametrize("name", list_audio_waveform_augments())
def test_registered_waveform_augment_contract(name):
    fn = get_audio_waveform_augment(name)
    waveform = tf.linspace(-0.8, 0.8, 256)[:, None]
    kwargs = dict(_WAVEFORM_KWARGS.get(name, {}))
    kwargs.update(seed=[7, 3], sample_rate=16000, prob=1.0, is_training=True)
    first = fn(waveform, **kwargs)
    second = fn(waveform, **kwargs)
    evaluation = fn(waveform, **{**kwargs, "is_training": False})

    assert first.dtype == tf.float32
    assert bool(tf.reduce_all(tf.math.is_finite(first)))
    np.testing.assert_allclose(first, second)
    np.testing.assert_allclose(evaluation, waveform)


@pytest.mark.parametrize("name", list_audio_spectrogram_augments())
def test_registered_spectrogram_augment_contract(name):
    fn = get_audio_spectrogram_augment(name)
    batched = name in {"mixstyle", "frequency_mixstyle"}
    spectrogram = tf.reshape(tf.linspace(-1.0, 1.0, 3 * 16 * 24), [3, 16, 24])
    value = spectrogram if batched else spectrogram[0]
    layout = "btf" if batched else "tf"
    kwargs = {"seed": [11, 5], "prob": 1.0, "layout": layout, "is_training": True}
    if name == "passt_patchout":
        kwargs.pop("prob")
        kwargs.update(n_freq_patches=1, n_time_patches=1)
    first = fn(value, **kwargs)
    second = fn(value, **kwargs)
    evaluation = fn(value, **{**kwargs, "is_training": False})

    assert first.dtype == tf.float32
    assert bool(tf.reduce_all(tf.math.is_finite(first)))
    np.testing.assert_allclose(first, second)
    np.testing.assert_allclose(evaluation, value)


@pytest.mark.parametrize("name", list_audio_batch_augments())
def test_registered_batch_augment_contract(name):
    fn = get_audio_batch_augment(name)
    inputs = tf.reshape(tf.linspace(-1.0, 1.0, 4 * 16 * 24), [4, 16, 24])
    labels = tf.one_hot(tf.range(4), 4)
    kwargs = {
        "seed": [13, 7],
        "prob": 1.0,
        "is_training": True,
        "num_classes": 4,
    }
    if name in {"cutmix_spec", "batch_mixstyle"}:
        kwargs["layout"] = "btf"
    if name == "batch_mixstyle":
        kwargs.pop("num_classes")
    first = fn(inputs, labels, **kwargs)
    second = fn(inputs, labels, **kwargs)
    evaluation = fn(inputs, labels, **{**kwargs, "is_training": False})
    first_values = first if isinstance(first, tuple) else (first,)
    second_values = second if isinstance(second, tuple) else (second,)
    eval_values = evaluation if isinstance(evaluation, tuple) else (evaluation,)

    for left, right in zip(first_values, second_values):
        assert bool(tf.reduce_all(tf.math.is_finite(left)))
        np.testing.assert_allclose(left, right)
    np.testing.assert_allclose(eval_values[0], inputs)


@pytest.mark.parametrize("name", list_audio_corruptions())
def test_registered_audio_corruption_contract(name):
    waveform = tf.linspace(-0.8, 0.8, 512)[:, None]
    first = apply_audio_corruption(
        waveform, name, severity=2, seed=[17, 9], config={"sample_rate": 16000}
    )
    second = apply_audio_corruption(
        waveform, name, severity=2, seed=[17, 9], config={"sample_rate": 16000}
    )

    assert first.shape == waveform.shape
    assert first.dtype == tf.float32
    assert bool(tf.reduce_all(tf.math.is_finite(first)))
    np.testing.assert_allclose(first, second)


def test_tfds_speech_commands_shape_is_adapted():
    sample = speech_commands_adapter(
        {
            "audio": tf.cast(tf.range(32), tf.int16),
            "label": tf.constant(2, tf.int64),
        }
    )
    assert sample["waveform"].shape == (32, 1)
    assert sample["waveform"].dtype == tf.float32
    assert int(sample["sample_rate"].numpy()) == 16000


def test_hf_audio_decoder_shape_reaches_canonical_adapter(monkeypatch):
    datasets = pytest.importorskip("datasets")

    class Decoder:
        def __getitem__(self, key):
            values = {
                "array": np.stack([np.arange(16), np.arange(16)]).astype(np.float32),
                "sampling_rate": 16000,
            }
            return values[key]

    class Dataset:
        column_names = ["audio", "label"]

        def __iter__(self):
            return iter([{"audio": Decoder(), "label": 1}])

        def __len__(self):
            return 1

    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: Dataset())
    raw = load_huggingface_audio_splits("hf_audio:synthetic", ["train"])[0]
    sample = acoustic_source_adapter(next(iter(raw)))

    assert sample["waveform"].shape == (16, 2)
    assert sample["waveform"].dtype == tf.float32
    assert int(sample["sample_rate"].numpy()) == 16000

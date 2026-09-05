import numpy as np
import pytest
import tensorflow as tf

from augmentation_harness import (
    assert_eval_disables_transform,
    assert_same_seed_same_output,
)
from justdata.acoustic.augment import (
    make_batch_augmentation_stage,
    make_spectrogram_augmentation_stage,
    frequency_mask,
    frequency_mixstyle,
    mel_bin_shift,
    mixstyle,
    random_eq,
    spectrogram_time_roll,
    time_frequency_erasing,
    time_mask,
)
from justdata.acoustic.registry import (
    get_audio_spectrogram_augment,
    get_audio_spectrogram_augment_metadata,
)
from justdata.acoustic.schema import FEATURES, LABEL, METADATA
from justdata.acoustic.tasks import make_augmentations, make_late_augmentations
from justdata.vision.augmentations.registry import get_augment_strategy


def _masked_output(transform, x):
    for offset in range(16):
        y = transform(x, seed=tf.constant([37 + offset, 11], dtype=tf.int32))
        if np.any(y.numpy() != x.numpy()):
            return y
    raise AssertionError("transform did not mask any element for the tested seeds")


def test_frequency_mask_masks_only_frequency_axis():
    x = tf.ones([6, 8], dtype=tf.float32)
    y = _masked_output(
        lambda value, seed: frequency_mask(
            value,
            seed=seed,
            max_width=3,
            fill_value="zero",
        ),
        x,
    )

    changed = y.numpy() != x.numpy()
    changed_freq = np.any(changed, axis=0)

    assert changed_freq.any()
    np.testing.assert_array_equal(changed, np.broadcast_to(changed_freq, changed.shape))


def test_time_mask_masks_only_time_axis():
    x = tf.ones([6, 8], dtype=tf.float32)
    y = _masked_output(
        lambda value, seed: time_mask(
            value,
            seed=seed,
            max_width=3,
            fill_value="zero",
        ),
        x,
    )

    changed = y.numpy() != x.numpy()
    changed_time = np.any(changed, axis=1)

    assert changed_time.any()
    np.testing.assert_array_equal(
        changed,
        np.broadcast_to(changed_time[:, np.newaxis], changed.shape),
    )


def test_mask_same_seed_same_output():
    def transform(x, seed, is_training):
        return frequency_mask(
            x,
            seed=seed,
            max_width=3,
            fill_value="zero",
            is_training=is_training,
        )

    assert_same_seed_same_output(transform, tf.ones([8, 10], dtype=tf.float32))


def test_mask_eval_disabled():
    def transform(x, seed, is_training):
        return time_mask(
            x,
            seed=seed,
            max_width=3,
            fill_value="zero",
            is_training=is_training,
        )

    assert_eval_disables_transform(transform, tf.ones([8, 10], dtype=tf.float32))


def test_time_roll_preserves_values():
    x = tf.reshape(tf.range(20, dtype=tf.float32), [4, 5])

    y = spectrogram_time_roll(x, seed=[1, 0], shift=2)

    np.testing.assert_array_equal(
        np.sort(y.numpy(), axis=None), np.sort(x.numpy(), axis=None)
    )


def test_mel_bin_shift_zero_pads_edges():
    x = tf.reshape(tf.range(15, dtype=tf.float32), [3, 5])

    y = mel_bin_shift(x, seed=[2, 0], shift=2, zero_pad=True)

    np.testing.assert_array_equal(y.numpy()[:, :2], np.zeros([3, 2], dtype=np.float32))
    np.testing.assert_array_equal(y.numpy()[:, 2:], x.numpy()[:, :3])


def test_time_frequency_erasing_shape():
    x = tf.ones([7, 9, 2], dtype=tf.float32)

    y = time_frequency_erasing(
        x,
        seed=[3, 0],
        max_time_width=3,
        max_freq_width=4,
        fill_value="zero",
    )

    assert y.shape == x.shape


def test_random_eq_changes_frequency_bands():
    x = tf.ones([5, 12], dtype=tf.float32)

    y = random_eq(
        x,
        seed=[4, 0],
        num_bands=2,
        min_db=6.0,
        max_db=6.0,
        min_band_width=2,
        max_band_width=2,
    )

    ratio = y.numpy() / x.numpy()
    assert np.any(np.abs(ratio - 1.0) > 1e-5)
    np.testing.assert_allclose(
        ratio,
        np.broadcast_to(ratio[:1, :], ratio.shape),
        rtol=1e-6,
    )


def test_mixstyle_same_shape():
    x = tf.reshape(tf.range(2 * 4 * 3, dtype=tf.float32), [2, 4, 3])

    y = mixstyle(x, seed=[5, 0], alpha=0.4, layout="btf")

    assert y.shape == x.shape


def test_mixstyle_preserves_batch_size():
    x = tf.ones([4, 6, 5, 2], dtype=tf.float32)

    y = mixstyle(x, seed=[6, 0], alpha=0.4, layout="btfc")

    assert y.shape[0] == x.shape[0]


def test_frequency_mixstyle_stats_axes():
    x = np.zeros([2, 4, 3, 2], dtype=np.float32)
    for batch in range(2):
        for freq in range(3):
            x[batch, :, freq, :] = batch * 10.0 + freq

    y = frequency_mixstyle(tf.constant(x), seed=[7, 0], alpha=0.4, layout="btfc")
    per_frequency_mean = np.mean(y.numpy(), axis=(1, 3))

    assert per_frequency_mean.shape == (2, 3)
    assert not np.allclose(per_frequency_mean[:, 0], per_frequency_mean[:, 1])


def test_spectrogram_augments_have_audio_metadata_only():
    fn = get_audio_spectrogram_augment("frequency_mask")
    metadata = get_audio_spectrogram_augment_metadata("frequency_mask")

    assert fn.domain == "spectrogram"
    assert metadata.name == "frequency_mask"
    assert metadata.domain == "spectrogram"
    assert metadata.is_training_only is True
    assert metadata.requires_labels is False
    with pytest.raises(ValueError):
        get_augment_strategy("frequency_mask")


def test_pre_frontend_augmentations_reject_spectrogram_configuration():
    with pytest.raises(ValueError, match="post-frontend features"):
        make_augmentations(spectrogram_augmentations={"frequency_mask": {}})


@pytest.mark.parametrize("batch_size", [1, 3])
def test_late_spectrogram_augmentation_maps_real_rows_deterministically(batch_size):
    features = tf.ones([batch_size, 12, 8], dtype=tf.float32)
    batch = {
        FEATURES: features,
        LABEL: tf.range(batch_size),
        METADATA: {"example_id": tf.strings.as_string(tf.range(batch_size))},
    }
    stage = make_late_augmentations(
        spectrogram_augmentations={
            "frequency_mask": {
                "max_width": 6,
                "fill_value": "zero",
            }
        },
        model_layout="btf",
    )

    first = stage(batch, seed=[41, 9])
    second = stage(batch, seed=[41, 9])

    np.testing.assert_array_equal(first[FEATURES], second[FEATURES])
    np.testing.assert_array_equal(first[LABEL], batch[LABEL])
    np.testing.assert_array_equal(
        first[METADATA]["example_id"], batch[METADATA]["example_id"]
    )
    assert np.any(first[FEATURES].numpy() != features.numpy())
    if batch_size == 3:
        assert not np.array_equal(first[FEATURES][0], first[FEATURES][1])


@pytest.mark.parametrize(
    ("model_layout", "shape"),
    [
        ("btf", (2, 12, 8)),
        ("btfc", (2, 12, 8, 1)),
        ("bcft", (2, 1, 8, 12)),
    ],
)
def test_late_spectrogram_augmentation_infers_sample_layout(model_layout, shape):
    batch = {FEATURES: tf.ones(shape, dtype=tf.float32)}
    stage = make_late_augmentations(
        spectrogram_augmentations={"time_mask": {"max_width": 4, "fill_value": "zero"}},
        model_layout=model_layout,
    )

    result = stage(batch, seed=[17, 3])

    assert result[FEATURES].shape == shape


def test_late_spectrogram_augmentation_rejects_incompatible_layout():
    with pytest.raises(ValueError, match="Spectrogram augmentation layout"):
        make_late_augmentations(
            spectrogram_augmentations={"time_mask": {}},
            model_layout="bctf",
        )


def test_late_patchout_preserves_metadata_and_records_per_row_debug_data():
    batch = {
        FEATURES: tf.ones([3, 1, 8, 12], dtype=tf.float32),
        METADATA: {"example_id": tf.constant(["a", "b", "c"])},
    }
    stage = make_late_augmentations(
        spectrogram_augmentations={
            "patchout": {
                "structured_frequency": 2,
                "structured_time": 3,
                "debug": True,
            }
        },
        model_layout="bcft",
    )

    result = stage(batch)

    assert result[FEATURES].shape == (3, 1, 6, 9)
    np.testing.assert_array_equal(
        result[METADATA]["example_id"], batch[METADATA]["example_id"]
    )
    assert result[METADATA]["patchout"]["structured_frequency"].shape == (3, 2)
    assert result[METADATA]["patchout"]["structured_time"].shape == (3, 3)


def test_late_spectrogram_augmentation_is_eval_opt_in():
    batch = {FEATURES: tf.ones([1, 12, 8], dtype=tf.float32)}
    config = {"frequency_mask": {"max_width": 8, "fill_value": "zero"}}
    disabled = make_late_augmentations(
        spectrogram_augmentations=config,
        model_layout="btf",
        is_training=False,
    )
    enabled = make_late_augmentations(
        spectrogram_augmentations=config,
        model_layout="btf",
        is_training=False,
        augment_eval=True,
    )

    np.testing.assert_array_equal(
        disabled(batch, seed=[29, 4])[FEATURES], batch[FEATURES]
    )
    assert np.any(
        enabled(batch, seed=[29, 4])[FEATURES].numpy() != batch[FEATURES].numpy()
    )


@pytest.mark.parametrize(
    "layout,shape", [("tf", (3, 12, 8)), ("tfc", (3, 12, 8, 1)), ("cft", (3, 1, 8, 12))]
)
@pytest.mark.parametrize("patchout", [False, True])
def test_late_spectrogram_matches_full_batch_mapping(layout, shape, patchout):
    feature_key = "custom_features"
    specs = {"time_mask": {"max_width": 5, "fill_value": "zero"}}
    if patchout:
        specs["passt_patchout"] = {"structured_time": 2, "structured_frequency": 1}
    batch = {
        feature_key: tf.reshape(tf.range(np.prod(shape), dtype=tf.float32), shape),
        "waveform": tf.reshape(tf.range(3 * 256, dtype=tf.float32), [3, 256, 1]),
        LABEL: tf.constant([0, 2, 1]),
        METADATA: {
            "id": tf.constant(["a", "b", "c"]),
            "nested": {"weight": tf.constant([0.1, 0.2, 0.3])},
        },
    }
    seed = tf.constant([19, 31], tf.int64)
    per_sample = make_spectrogram_augmentation_stage(
        specs, feature_key=feature_key, layout=layout
    )
    signature = tf.nest.map_structure(
        lambda value: tf.TensorSpec(value.shape[1:], value.dtype), batch
    )
    signature[feature_key] = tf.TensorSpec([None] * (len(shape) - 1), tf.float32)
    spec_seed, batch_seed = tf.unstack(tf.random.split(tf.cast(seed, tf.int32), 2))
    expected = tf.map_fn(
        lambda values: per_sample(values[0], seed=values[1]),
        (batch, tf.random.split(spec_seed, 3)),
        fn_output_signature=signature,
    )
    # Batch mixing must still receive the same labels, waveform, and second seed.
    batch_specs = {"mixup": {"alpha": 0.4}}
    batch_stage = make_batch_augmentation_stage(
        batch_specs, input_key=feature_key, label_mode="single_label"
    )
    expected = batch_stage(expected, num_classes=3, seed=batch_seed)
    stage = make_late_augmentations(
        spectrogram_augmentations=specs,
        spectrogram_key=feature_key,
        spectrogram_layout=layout,
        batch_augmentations=batch_specs,
        input_key=feature_key,
        label_mode="single_label",
    )
    actual = tf.function(stage)(batch, num_classes=3, seed=seed)
    tf.nest.assert_same_structure(actual, expected)
    for left, right in zip(tf.nest.flatten(actual), tf.nest.flatten(expected)):
        np.testing.assert_array_equal(left, right)


def test_late_spectrogram_preserves_tensor_metadata():
    batch = {FEATURES: tf.ones([2, 12, 8]), METADATA: tf.constant(["a", "b"])}
    stage = make_late_augmentations(
        spectrogram_augmentations={"time_mask": {}}, model_layout="btf"
    )
    result = stage(batch, seed=[5, 1])
    np.testing.assert_array_equal(result[METADATA], batch[METADATA])


@pytest.mark.parametrize("invalid", [tf.ones([2, 16]), tf.constant(1.0)])
def test_late_spectrogram_rejects_invalid_passthrough_batch_shapes(invalid):
    stage = make_late_augmentations(
        spectrogram_augmentations={"time_mask": {}}, model_layout="btf"
    )
    with pytest.raises(ValueError):
        stage({FEATURES: tf.ones([3, 12, 8]), "waveform": invalid}, seed=[5, 1])


def test_late_spectrogram_checks_dynamic_passthrough_batch_size():
    stage = make_late_augmentations(
        spectrogram_augmentations={"time_mask": {}}, model_layout="btf"
    )

    @tf.function(
        input_signature=[
            tf.TensorSpec([None, 12, 8], tf.float32),
            tf.TensorSpec([None, 16], tf.float32),
        ]
    )
    def run(features, waveform):
        return stage({FEATURES: features, "waveform": waveform}, seed=[5, 1])

    run(tf.ones([3, 12, 8]), tf.ones([3, 16]))
    with pytest.raises(tf.errors.InvalidArgumentError):
        run(tf.ones([3, 12, 8]), tf.ones([2, 16]))


def test_late_patchout_retains_signature_validation_for_deferred_debug():
    stage = make_late_augmentations(
        spectrogram_augmentations={
            "patchout": {"config": {"debug": True}, "debug": None}
        },
        model_layout="btf",
    )
    with pytest.raises(ValueError):
        stage({FEATURES: tf.ones([2, 12, 8])}, seed=[5, 1])

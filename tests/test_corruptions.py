import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import tensorflow as tf

from justdata.vision.corruptions import blur as legacy_blur
from justdata.vision.corruptions import digital as legacy_digital
from justdata.vision.corruptions import noise as legacy_noise
from justdata.vision.corruptions import weather as legacy_weather
from justdata.vision.corruptions.registry import (
    CorruptionDescriptor,
    _CORRUPTION_DESCRIPTOR_REGISTRY,
    _CORRUPTION_LOCK,
    _CORRUPTION_REGISTRY,
    apply_corruption,
    apply_minic_corruption,
    get_corruption_descriptor,
    list_corruption_descriptors,
    list_corruption_versions,
    list_corruptions,
    register_corruption,
)

NEW_CORRUPTIONS = (
    "brightness_reduction",
    "contrast_reduction",
    "gaussian_blur",
    "gaussian_noise",
    "jpeg_compression",
)
LEGACY_CORRUPTIONS = ("blur", "digital", "noise", "weather")
SEED = tf.constant([17, 29], dtype=tf.int32)


def _image(height=17, width=19):
    return tf.reshape(
        tf.cast(tf.range(height * width * 3) % 256, tf.uint8),
        [height, width, 3],
    )


def _severity_parameter(descriptor, name):
    return tuple(
        dict(parameters)[name] for _level, parameters in descriptor.severity_parameters
    )


def _test_descriptor(name):
    return CorruptionDescriptor(
        name=name,
        version="1.0.0",
        input_domain="test_uint8_hwc",
        input_dtypes=("uint8",),
        output_domain="test_uint8_hwc",
        output_shape="same_as_input_hwc",
        output_dtype="uint8",
        severity_values=(1,),
        severity_parameters=((1, (("value", 1),)),),
        uses_randomness=False,
        seed_contract="caller_supplied_int32_shape_2; ignored",
        clipping_policy="clip_float32_to_closed_interval_0_255",
        rounding_policy="clip_then_round_half_to_even",
        implementation="test_only",
    )


def test_corruptions_registered():
    expected = set(LEGACY_CORRUPTIONS + NEW_CORRUPTIONS)
    assert expected.issubset(set(list_corruptions()))
    assert expected.issubset(
        {descriptor.name for descriptor in list_corruption_descriptors()}
    )


def test_required_descriptors_are_plain_immutable_and_uniquely_identified():
    descriptors = [get_corruption_descriptor(name, "1.0.0") for name in NEW_CORRUPTIONS]
    assert len({descriptor.identity for descriptor in descriptors}) == len(descriptors)

    for descriptor in descriptors:
        assert json.loads(descriptor.to_json()) == descriptor.to_dict()
        assert descriptor.input_domain
        assert descriptor.output_domain
        assert descriptor.output_shape == "same_as_input_hwc"
        assert descriptor.output_dtype == "uint8"
        assert descriptor.severity_values == (1, 2, 3, 4, 5)
        assert descriptor.clipping_policy
        assert descriptor.rounding_policy
        assert descriptor.implementation
        with pytest.raises(FrozenInstanceError):
            descriptor.version = "2.0.0"


def test_required_descriptor_severity_tables_are_exact():
    blur = get_corruption_descriptor("gaussian_blur", "1.0.0")
    noise = get_corruption_descriptor("gaussian_noise", "1.0.0")
    jpeg = get_corruption_descriptor("jpeg_compression", "1.0.0")
    contrast = get_corruption_descriptor("contrast_reduction", "1.0.0")
    brightness = get_corruption_descriptor("brightness_reduction", "1.0.0")

    assert _severity_parameter(
        blur,
        "sigma_pixels",
    ) == (0.5, 1.0, 2.0, 3.0, 4.0)
    assert _severity_parameter(
        noise,
        "standard_deviation_unit_range",
    ) == (0.01, 0.02, 0.04, 0.08, 0.16)
    assert _severity_parameter(
        jpeg,
        "quality",
    ) == (90, 75, 55, 35, 15)
    assert _severity_parameter(
        contrast,
        "factor",
    ) == (0.9, 0.75, 0.6, 0.45, 0.3)
    assert _severity_parameter(
        contrast,
        "center_0_255",
    ) == (127.5, 127.5, 127.5, 127.5, 127.5)
    assert _severity_parameter(
        brightness,
        "factor",
    ) == (0.9, 0.8, 0.7, 0.6, 0.5)

    for descriptor in (blur, noise, contrast, brightness):
        assert descriptor.clipping_policy == ("clip_float32_to_closed_interval_0_255")
        assert descriptor.rounding_policy == "clip_then_round_half_to_even"
    assert dict(blur.implementation_parameters)["border_mode"] == (
        "reflect_101_without_repeating_edge"
    )
    assert dict(blur.implementation_parameters)["kernel_radius_formula"] == (
        "ceil_3_times_sigma"
    )
    assert dict(noise.implementation_parameters)["random_algorithm"] == "philox"


def test_jpeg_descriptor_declares_codec_settings():
    descriptor = get_corruption_descriptor("jpeg_compression", "1.0.0")
    settings = dict(descriptor.implementation_parameters)
    assert settings["encode_color_mode"] == "rgb"
    assert settings["chroma_downsampling"] is True
    assert settings["optimize_size"] is False
    assert settings["progressive"] is False
    assert settings["decode_channels"] == 3
    assert settings["decode_dct_method"] == "INTEGER_ACCURATE"
    assert "not_cross_version" in settings["byte_guarantee"]


def test_descriptor_lookup_and_version_enumeration_require_exact_version():
    assert list_corruption_versions("gaussian_noise") == ("1.0.0",)
    with pytest.raises(ValueError, match="gaussian_noise@2.0.0"):
        get_corruption_descriptor("gaussian_noise", "2.0.0")
    with pytest.raises(ValueError, match="gaussian_noise@2.0.0"):
        apply_corruption(
            _image(),
            name="gaussian_noise",
            version="2.0.0",
            severity=1,
            seed=SEED,
        )


def test_duplicate_name_version_registration_is_thread_safe():
    name = "_test_threadsafe_versioned_corruption"
    descriptor = _test_descriptor(name)

    def implementation(image, severity, seed):
        del severity, seed
        return image

    def attempt_registration():
        try:
            register_corruption(name, descriptor=descriptor)(implementation)
            return "registered"
        except ValueError:
            return "duplicate"

    try:
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(
                executor.map(lambda _index: attempt_registration(), range(8))
            )
        assert results.count("registered") == 1
        assert results.count("duplicate") == 7
    finally:
        with _CORRUPTION_LOCK:
            _CORRUPTION_REGISTRY.pop((name, descriptor.version), None)
            _CORRUPTION_DESCRIPTOR_REGISTRY.pop((name, descriptor.version), None)


@pytest.mark.parametrize("name", NEW_CORRUPTIONS)
@pytest.mark.parametrize("severity", [1, 5])
def test_apply_required_corruption_preserves_shape_and_uint8(name, severity):
    result = apply_corruption(
        _image(),
        name=name,
        version="1.0.0",
        severity=severity,
        seed=SEED,
    )
    assert result.shape == (17, 19, 3)
    assert result.dtype == tf.uint8


@pytest.mark.parametrize("name", NEW_CORRUPTIONS)
def test_required_corruption_same_seed_same_output(name):
    first = apply_corruption(
        _image(),
        name=name,
        version="1.0.0",
        severity=3,
        seed=SEED,
    )
    second = apply_corruption(
        _image(),
        name=name,
        version="1.0.0",
        severity=3,
        seed=SEED,
    )
    np.testing.assert_array_equal(first.numpy(), second.numpy())


def test_gaussian_noise_different_seed_changes_exact_output():
    first = apply_corruption(
        _image(),
        name="gaussian_noise",
        version="1.0.0",
        severity=3,
        seed=tf.constant([1, 2], dtype=tf.int32),
    )
    second = apply_corruption(
        _image(),
        name="gaussian_noise",
        version="1.0.0",
        severity=3,
        seed=tf.constant([1, 3], dtype=tf.int32),
    )
    assert not np.array_equal(first.numpy(), second.numpy())


def test_nonrandom_corruption_accepts_and_ignores_valid_seed():
    first = apply_corruption(
        _image(),
        name="brightness_reduction",
        version="1.0.0",
        severity=3,
        seed=tf.constant([1, 2], dtype=tf.int32),
    )
    second = apply_corruption(
        _image(),
        name="brightness_reduction",
        version="1.0.0",
        severity=3,
        seed=tf.constant([8, 9], dtype=tf.int32),
    )
    np.testing.assert_array_equal(first.numpy(), second.numpy())


def test_rounding_is_half_to_even_and_output_is_clipped():
    image = tf.constant([[[1, 3, 5], [253, 254, 255]]], dtype=tf.uint8)
    result = apply_corruption(
        image,
        name="brightness_reduction",
        version="1.0.0",
        severity=5,
        seed=SEED,
    )
    np.testing.assert_array_equal(
        result.numpy(),
        [[[0, 2, 2], [126, 127, 128]]],
    )

    noisy = apply_corruption(
        tf.constant([[[0, 0, 0], [255, 255, 255]]], dtype=tf.uint8),
        name="gaussian_noise",
        version="1.0.0",
        severity=5,
        seed=SEED,
    )
    assert noisy.dtype == tf.uint8
    assert np.min(noisy.numpy()) >= 0
    assert np.max(noisy.numpy()) <= 255


@pytest.mark.parametrize("severity", [0, 6])
def test_generic_corruption_rejects_invalid_severity_before_execution(severity):
    with pytest.raises(ValueError, match="severity"):
        apply_corruption(
            _image(),
            name="gaussian_noise",
            version="1.0.0",
            severity=severity,
            seed=SEED,
        )


@pytest.mark.parametrize(
    ("image", "error_type", "match"),
    [
        (tf.zeros([4, 4, 3], tf.float32), TypeError, "dtype"),
        (tf.zeros([4, 4], tf.uint8), ValueError, "rank-3"),
        (tf.zeros([4, 4, 1], tf.uint8), ValueError, "three RGB"),
    ],
)
def test_generic_corruption_validates_input_domain(image, error_type, match):
    with pytest.raises(error_type, match=match):
        apply_corruption(
            image,
            name="brightness_reduction",
            version="1.0.0",
            severity=1,
            seed=SEED,
        )


@pytest.mark.parametrize(
    "seed",
    [
        tf.constant(1, dtype=tf.int32),
        tf.constant([1], dtype=tf.int32),
        tf.constant([1, 2], dtype=tf.int64),
    ],
)
def test_generic_corruption_requires_complete_int32_stateless_seed(seed):
    with pytest.raises((TypeError, ValueError), match="seed|int32"):
        apply_corruption(
            _image(),
            name="gaussian_noise",
            version="1.0.0",
            severity=1,
            seed=seed,
        )


def test_generic_corruption_runs_in_tf_function_and_dataset_map():
    @tf.function
    def graph_apply(image):
        return apply_corruption(
            image,
            name="gaussian_noise",
            version="1.0.0",
            severity=tf.constant(2, dtype=tf.int32),
            seed=SEED,
        )

    expected = graph_apply(_image())
    dataset = tf.data.Dataset.from_tensors(_image()).map(graph_apply)
    actual = next(iter(dataset))
    np.testing.assert_array_equal(expected.numpy(), actual.numpy())


@pytest.mark.parametrize(
    ("name", "implementation"),
    [
        ("noise", legacy_noise.gaussian_noise),
        ("blur", legacy_blur.defocus_blur),
        ("weather", legacy_weather.snow),
        ("digital", legacy_digital.pixelate),
    ],
)
def test_minic_wrapper_preserves_legacy_float_input_and_finalization(
    name,
    implementation,
):
    image = tf.cast(_image(64, 64), tf.float32)
    raw = implementation(image, severity=3, seed=SEED)
    expected = tf.cast(tf.clip_by_value(raw, 0.0, 255.0), tf.uint8)
    actual = apply_minic_corruption(image, name, severity=3, seed=SEED)
    np.testing.assert_array_equal(expected.numpy(), actual.numpy())


def test_unversioned_custom_registration_remains_minic_compatible():
    name = "_test_unversioned_custom_corruption"

    def add_one(image, severity, seed):
        del severity, seed
        return image + 1.0

    try:
        register_corruption(name)(add_one)
        result = apply_minic_corruption(
            tf.zeros([2, 2, 3], dtype=tf.float32),
            name,
            severity=1,
            seed=SEED,
        )
        np.testing.assert_array_equal(result.numpy(), np.ones([2, 2, 3], np.uint8))
    finally:
        with _CORRUPTION_LOCK:
            _CORRUPTION_REGISTRY.pop((name, None), None)


def test_unknown_minic_corruption():
    with pytest.raises(ValueError, match="Unknown corruption"):
        apply_minic_corruption(_image(), "unknown_xyz", severity=1, seed=SEED)


def test_minic_rejects_invalid_severity():
    with pytest.raises(ValueError, match="severity"):
        apply_minic_corruption(_image(), "noise", severity=0, seed=SEED)

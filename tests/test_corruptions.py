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
from justdata.vision.corruptions.continuous import (
    apply_corruption_at,
    apply_corruption_stack,
    corruption_stack_identity,
    resolve_parameters,
)
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
    severity_table,
)
from justdata.vision.horizon import Horizon

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


@pytest.mark.parametrize(
    ("name", "digest"),
    [
        ("blur", "f3e017a21821daeabc9dc31f4fb3e1d785353ece9391ed5244616317b46b8c34"),
        (
            "brightness_reduction",
            "73f32f855970178d53eb036aee4a63c362cb79f4f8dc4045396fc42f7d99f2e7",
        ),
        (
            "contrast_reduction",
            "c8d27a81c2e8c7581da5bd17b92fde42228e75fca1ae7cf3b0d666bd2e328557",
        ),
        ("digital", "dd49088c4d173ccfc5d6b2670f86a55bc585ad3b339663767c27a2719c01def3"),
        (
            "gaussian_blur",
            "b2102a42e8eb72c28d0020f679d07a34e59747dd8371a72625cd09fac8adacc5",
        ),
        (
            "gaussian_noise",
            "bc2d1d4e35f4b32131b0b758b09f423965bde9f2a58282b237a0a523953df5bf",
        ),
        (
            "jpeg_compression",
            "a4563fb1d19783a792a7a8bad00750a3fc353a1ed78fd534cb2e0ff2da97b7b7",
        ),
        ("noise", "a15c8c949b66b5861c1febb15add0c395fed40c204e3be7c6e8c9a4d62b7fa68"),
        ("weather", "4646493008593649aa9884a5956074acfaba412bd35e0959afbffe6a7dd77922"),
    ],
)
def test_severity_only_descriptor_identities_are_unchanged(name, digest):
    descriptor = get_corruption_descriptor(name, "1.0.0")
    assert descriptor.identity == f"{name}@1.0.0:sha256:{digest}"
    assert not descriptor.supports_strength
    assert "identity_parameters" not in descriptor.to_dict()


def _strength_descriptor(**overrides):
    fields = dict(
        name="_test_strength",
        version="1.0.0",
        input_domain="test",
        input_dtypes=("uint8",),
        output_domain="test",
        output_shape="same_as_input_hwc",
        output_dtype="uint8",
        severity_values=(1, 2),
        severity_parameters=severity_table({"gain": (2.0, 8.0), "level": (1, 3)}),
        uses_randomness=False,
        seed_contract="ignored",
        clipping_policy="clip",
        rounding_policy="clip_then_round_half_to_even",
        implementation="test_only",
        identity_parameters=(("gain", 1.0), ("level", 0)),
        interpolation=(("gain", "log"), ("level", "nearest")),
        label_effect="none",
    )
    return CorruptionDescriptor(**(fields | overrides))


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"identity_parameters": (), "interpolation": ()}, "label_effect"),
        ({"identity_parameters": (("gain", 0.0), ("level", 0))}, "log"),
        ({"interpolation": (("gain", "cubic"), ("level", "nearest"))}, "interpolation"),
        ({"interpolation": (("level", "nearest"), ("gain", "log"))}, "same order"),
        (
            {
                "severity_values": (2, 3),
                "severity_parameters": (
                    (2, (("gain", 2.0), ("level", 1))),
                    (3, (("gain", 8.0), ("level", 3))),
                ),
            },
            "1..N",
        ),
        ({"rounding_policy": "clip_then_truncate_toward_zero"}, "rounding"),
        ({"spatial_parameters": ("gain",)}, "absolute_pixels"),
        (
            {"spatial_scale_policy": "relative_to_width", "reference_width": 0},
            "reference_width",
        ),
        ({"auxiliary_inputs": ("depth_m",)}, "modes"),
        (
            {"auxiliary_inputs": ("depth_m",), "modes": (("depth", ("depth_m",)),)},
            "fallback",
        ),
        ({"auxiliary_inputs": ("normals",), "modes": (("x", ()),)}, "auxiliary"),
    ],
)
def test_strength_contract_validation(overrides, match):
    with pytest.raises(ValueError, match=match):
        _strength_descriptor(**overrides)


def test_strength_contract_is_serialized_into_the_identity():
    descriptor = _strength_descriptor()
    data = descriptor.to_dict()
    assert data["identity_parameters"] == {"gain": 1.0, "level": 0}
    assert data["interpolation"] == {"gain": "log", "level": "nearest"}
    assert data["label_effect"] == "none"
    other = _strength_descriptor(identity_parameters=(("gain", 1.5), ("level", 0)))
    assert other.identity != descriptor.identity


def test_resolve_parameters_follows_declared_rules():
    descriptor = _strength_descriptor()
    register_corruption(descriptor.name, descriptor=descriptor)(lambda *a, **k: a[0])
    try:
        assert resolve_parameters(descriptor.name, "1.0.0", 0) == {
            "gain": 1.0,
            "level": 0.0,
        }
        assert resolve_parameters(descriptor.name, "1.0.0", 2) == {
            "gain": 8.0,
            "level": 3.0,
        }
        half = resolve_parameters(descriptor.name, "1.0.0", 1.5)
        assert half["gain"] == pytest.approx(4.0, rel=1e-6)
        assert half["level"] == 3.0
        below_half = resolve_parameters(descriptor.name, "1.0.0", 0.25)
        assert below_half["gain"] == pytest.approx(2.0**0.25, rel=1e-6)
        assert below_half["level"] == 0.0
    finally:
        with _CORRUPTION_LOCK:
            _CORRUPTION_REGISTRY.pop((descriptor.name, "1.0.0"), None)
            _CORRUPTION_DESCRIPTOR_REGISTRY.pop((descriptor.name, "1.0.0"), None)


def test_resolved_parameters_are_float32_table_values():
    resolved = resolve_parameters("low_light", "1.0.0", 3)
    row = dict(
        get_corruption_descriptor("low_light", "1.0.0").severity_parameters[2][1]
    )
    for name, value in resolved.items():
        assert value == float(np.float32(row[name]))


@pytest.mark.parametrize(
    ("strength", "error_type"),
    [
        (-0.1, ValueError),
        (5.01, ValueError),
        (float("nan"), ValueError),
        (True, TypeError),
        (tf.constant(1.0), TypeError),
    ],
)
def test_apply_corruption_at_rejects_invalid_strengths(strength, error_type):
    with pytest.raises(error_type, match="strength"):
        apply_corruption_at(
            _image(), name="haze", version="1.0.0", strength=strength, seed=SEED
        )


def test_continuous_entry_points_reject_severity_only_descriptors():
    with pytest.raises(ValueError, match="integer severities"):
        apply_corruption_at(
            _image(), name="gaussian_noise", version="1.0.0", strength=1, seed=SEED
        )
    with pytest.raises(ValueError, match="integer severities"):
        apply_corruption_stack(
            _image(), steps=[("jpeg_compression", "1.0.0", 1)], seed=SEED
        )


def test_apply_corruption_at_validates_auxiliary_inputs():
    region = tf.ones([17, 19], tf.bool)
    with pytest.raises(ValueError, match="not used"):
        apply_corruption_at(
            _image(),
            name="low_light",
            version="1.0.0",
            strength=1,
            seed=SEED,
            region=region,
        )
    with pytest.raises(TypeError, match="bool"):
        apply_corruption_at(
            _image(),
            name="sun_glitter",
            version="1.0.0",
            strength=1,
            seed=SEED,
            region=tf.ones([17, 19], tf.float32),
        )
    with pytest.raises(tf.errors.InvalidArgumentError, match="height and width"):
        apply_corruption_at(
            _image(),
            name="sun_glitter",
            version="1.0.0",
            strength=1,
            seed=SEED,
            region=tf.ones([17, 18], tf.bool),
        )
    with pytest.raises(tf.errors.InvalidArgumentError, match="non-negative"):
        apply_corruption_at(
            _image(),
            name="haze",
            version="1.0.0",
            strength=1,
            seed=SEED,
            depth_m=tf.fill([17, 19], float("nan")),
        )
    with pytest.raises(TypeError, match="Horizon"):
        apply_corruption_at(
            _image(),
            name="haze",
            version="1.0.0",
            strength=1,
            seed=SEED,
            horizon=(3.0, 4.0),
        )


@pytest.mark.parametrize("dtype", [tf.uint8, tf.float32])
def test_zero_strength_returns_the_rounded_input(dtype):
    image = tf.constant(
        [[[0.4, 1.5, 2.5], [254.6, 255.0, 300.0]]]
        if dtype == tf.float32
        else [[[0, 1, 2], [253, 254, 255]]],
        dtype=dtype,
    )
    expected = tf.cast(
        tf.math.rint(tf.clip_by_value(tf.cast(image, tf.float32), 0, 255)), tf.uint8
    )
    result = apply_corruption_at(
        image, name="low_light", version="1.0.0", strength=0, seed=SEED
    )
    np.testing.assert_array_equal(result.image.numpy(), expected.numpy())
    assert result.record.parameters == (
        ("exposure_gain", 1.0),
        ("shot_noise", 0.0),
        ("read_noise_variance", 0.0),
        ("compensation_cap", 1.0),
    )


def test_corruption_record_describes_the_call():
    record = apply_corruption_at(
        _image(),
        name="sun_glitter",
        version="1.0.0",
        strength=2.5,
        seed=SEED,
        episode_seed=tf.constant([5, 6], tf.int32),
        horizon=Horizon(4.0, 6.0),
    ).record
    data = record.to_dict()
    assert json.loads(json.dumps(data)) == data
    assert record.identity == get_corruption_descriptor("sun_glitter", "1.0.0").identity
    assert record.strength == 2.5
    assert record.mode == "horizon"
    assert record.auxiliary_inputs == ("horizon",)
    assert record.episode_seed == "provided"
    assert record.image_width == 19
    assert record.reference_width == 1024
    assert dict(record.parameters) == resolve_parameters("sun_glitter", "1.0.0", 2.5)


def test_stack_rounds_once_and_matches_float_composition():
    image = _image(32, 40)
    steps = [("overexposure", "1.0.0", 1.5), ("white_balance_warm", "1.0.0", 2.0)]
    stacked = apply_corruption_stack(image, steps=steps, seed=SEED)
    sequential = image
    for name, version, strength in steps:
        sequential = apply_corruption_at(
            sequential, name=name, version=version, strength=strength, seed=SEED
        ).image
    difference = np.abs(
        stacked.image.numpy().astype(int) - sequential.numpy().astype(int)
    )
    assert difference.max() <= 1
    assert [step.name for step in stacked.record.steps] == [
        "overexposure",
        "white_balance_warm",
    ]


def test_stack_identity_depends_on_order_and_strength():
    image = _image()
    steps = [("haze", "1.0.0", 1.0), ("low_light", "1.0.0", 2.0)]
    forward = apply_corruption_stack(image, steps=steps, seed=SEED).record
    reverse = apply_corruption_stack(image, steps=steps[::-1], seed=SEED).record
    stronger = apply_corruption_stack(
        image, steps=[steps[0], ("low_light", "1.0.0", 2.5)], seed=SEED
    ).record
    assert len({forward.identity, reverse.identity, stronger.identity}) == 3
    assert forward.identity == corruption_stack_identity(
        [
            (get_corruption_descriptor(name, version).identity, strength)
            for name, version, strength in steps
        ]
    )


def test_stack_routes_auxiliary_inputs_to_declaring_steps():
    image = _image()
    region = tf.ones([17, 19], tf.bool)
    result = apply_corruption_stack(
        image,
        steps=[("low_light", "1.0.0", 1.0), ("sun_glitter", "1.0.0", 1.0)],
        seed=SEED,
        region=region,
    )
    assert result.record.steps[0].auxiliary_inputs == ()
    assert result.record.steps[1].mode == "region"
    with pytest.raises(ValueError, match="not used"):
        apply_corruption_stack(
            image, steps=[("low_light", "1.0.0", 1.0)], seed=SEED, region=region
        )

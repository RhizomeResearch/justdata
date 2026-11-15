import numpy as np
import pytest
import tensorflow as tf

from justdata.vision.corruptions import (
    Horizon,
    apply_corruption,
    apply_corruption_at,
    get_corruption_descriptor,
    list_corruption_descriptors,
)
from justdata.vision.horizon import horizon_coordinates

CONDITIONS = (
    "haze",
    "lens_droplets",
    "low_light",
    "motion_blur",
    "overexposure",
    "rain",
    "shot_noise",
    "sun_glitter",
    "white_balance_cool",
    "white_balance_warm",
)
SEED = tf.constant([3, 9], tf.int32)
EPISODE = tf.constant([21, 4], tf.int32)


def _scene(height=48, width=64, seed=0):
    """Smooth RGB scene with texture: sky above row height/2, darker water below."""
    rng = np.random.default_rng(seed)
    coarse = rng.uniform(0, 1, size=(height // 8, width // 8, 3)).astype(np.float32)
    texture = tf.image.resize(coarse, [height, width], method="bilinear").numpy()
    rows = np.linspace(0, 1, height, dtype=np.float32)[:, None, None]
    base = np.where(rows < 0.5, [0.6, 0.75, 0.9], [0.1, 0.25, 0.35]).astype(np.float32)
    image = np.clip(base + 0.25 * (texture - 0.5), 0, 1) * 255
    return tf.constant(np.round(image).astype(np.uint8))


def _apply(image, name, strength, seed=SEED, **kwargs):
    return apply_corruption_at(
        image, name=name, version="1.0.0", strength=strength, seed=seed, **kwargs
    )


def test_conditions_are_registered_with_continuous_photometric_contracts():
    registered = {d.name for d in list_corruption_descriptors() if d.supports_strength}
    assert registered == set(CONDITIONS)
    for name in CONDITIONS:
        descriptor = get_corruption_descriptor(name, "1.0.0")
        assert descriptor.label_effect == "none"
        assert descriptor.severity_values == (1, 2, 3, 4, 5)
        assert descriptor.input_dtypes == ("uint8", "float32")
        assert descriptor.uses_randomness == bool(
            descriptor.episode_draws or descriptor.frame_draws
        )


@pytest.mark.parametrize("name", CONDITIONS)
def test_integer_strength_equals_severity_row_and_zero_is_identity(name):
    image = _scene()
    for severity in (1, 5):
        expected = apply_corruption(
            image, name=name, version="1.0.0", severity=severity, seed=SEED
        )
        actual = _apply(image, name, severity).image
        np.testing.assert_array_equal(actual.numpy(), expected.numpy())
    np.testing.assert_array_equal(_apply(image, name, 0).image.numpy(), image.numpy())


@pytest.mark.parametrize("name", CONDITIONS)
def test_condition_is_deterministic_within_an_execution_mode(name):
    image = _scene()

    @tf.function
    def graph_apply(value):
        return apply_corruption(
            value, name=name, version="1.0.0", severity=tf.constant(3), seed=SEED
        )

    eager = _apply(image, name, 3).image
    graph = graph_apply(image)
    assert eager.dtype == tf.uint8 and eager.shape == image.shape
    np.testing.assert_array_equal(eager.numpy(), _apply(image, name, 3).image.numpy())
    np.testing.assert_array_equal(graph.numpy(), graph_apply(image).numpy())
    # Grappler constant-folds the inlined implementation inside a caller's graph,
    # which can move a float32 value across a rounding boundary.
    mismatch = eager.numpy().astype(int) - graph.numpy().astype(int)
    assert np.abs(mismatch).max() <= 1
    assert np.mean(mismatch != 0) < 0.001


def _rms_contrast(image):
    return float(np.std(image.numpy().astype(np.float64)))


def _gradient_energy(image):
    values = image.numpy().astype(np.float64)
    return float(np.mean(np.abs(np.diff(values, axis=1))))


def _changed_share(image, original):
    return float(np.mean(np.any(image.numpy() != original.numpy(), axis=-1)))


def _saturated_share(image):
    return float(np.mean(np.all(image.numpy() >= 250, axis=-1)))


def _red_over_blue(image):
    values = image.numpy().astype(np.float64)
    return float(values[..., 0].mean() / values[..., 2].mean())


def _flat_std(image):
    return float(np.std(image.numpy().astype(np.float64)))


@pytest.mark.parametrize(
    ("name", "statistic", "direction"),
    [
        ("haze", _rms_contrast, -1),
        ("low_light", lambda image: float(np.mean(image.numpy())), -1),
        ("motion_blur", _gradient_energy, -1),
        ("overexposure", _saturated_share, 1),
        ("white_balance_warm", _red_over_blue, 1),
        ("white_balance_cool", _red_over_blue, -1),
    ],
)
def test_condition_statistic_moves_monotonically_with_strength(
    name, statistic, direction
):
    image = _scene(96, 128)
    values = [statistic(_apply(image, name, strength).image) for strength in range(6)]
    steps = np.diff(values) * direction
    assert np.all(steps >= 0), values
    assert values[-1] != values[0]


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        ("lens_droplets", {}),
        ("sun_glitter", {"horizon": Horizon(48.0, 48.0)}),
    ],
)
def test_affected_area_grows_with_strength(name, kwargs):
    image = _scene(96, 128)
    shares = [
        _changed_share(_apply(image, name, strength, **kwargs).image, image)
        for strength in (0, 1, 2, 3, 4, 5)
    ]
    assert np.all(np.diff(shares) >= 0), shares
    assert shares[0] == 0 and shares[-1] > shares[1]


def test_rain_streak_coverage_grows_with_strength():
    image = _scene(96, 128)
    dimmed = np.rint(image.numpy() * 0.7)
    coverage = []
    for strength in (1, 2, 3, 4, 5):
        output = _apply(image, "rain", strength).image.numpy()
        coverage.append(np.mean(np.any(output > dimmed + 1, axis=-1)))
    assert np.all(np.diff(coverage) > 0), coverage


def test_glitter_lights_more_of_the_water_with_strength():
    image = _scene(96, 128)
    water = np.zeros([96, 128], bool)
    water[48:] = True
    shares = []
    for strength in (1, 3, 5):
        output = _apply(
            image, "sun_glitter", strength, region=tf.constant(water)
        ).image.numpy()
        brightened = output.astype(int).sum(-1) - image.numpy().astype(int).sum(-1)
        shares.append(np.mean(brightened[water] > 3 * 64))
    assert shares[0] < shares[1] < shares[2], shares


@pytest.mark.parametrize("name", ["low_light", "shot_noise"])
def test_noise_grows_with_strength_on_a_flat_image(name):
    flat = tf.fill([64, 64, 3], tf.constant(128, tf.uint8))
    stds = [_flat_std(_apply(flat, name, strength).image) for strength in (1, 3, 5)]
    assert stds[0] < stds[1] < stds[2]


def test_low_light_never_brightens_without_noise():
    image = _scene()
    dimmed = _apply(image, "low_light", 2.5).image.numpy().astype(int)
    assert dimmed.mean() < image.numpy().mean()


@pytest.mark.parametrize("name", ["haze", "lens_droplets"])
def test_episode_only_conditions_ignore_the_frame_seed(name):
    image = _scene()
    first = _apply(image, name, 3, seed=SEED, episode_seed=EPISODE).image
    second = _apply(image, name, 3, seed=SEED + 1, episode_seed=EPISODE).image
    other_episode = _apply(image, name, 3, seed=SEED, episode_seed=EPISODE + 1).image
    np.testing.assert_array_equal(first.numpy(), second.numpy())
    assert not np.array_equal(first.numpy(), other_episode.numpy())


def test_droplet_layout_persists_across_frames_and_strengths():
    image = _scene(96, 128)

    def covered(strength, seed):
        output = _apply(
            image, "lens_droplets", strength, seed=seed, episode_seed=EPISODE
        )
        return np.any(output.image.numpy() != image.numpy(), axis=-1)

    weak = covered(2, SEED)
    strong = covered(3, SEED + 7)
    assert weak.any()
    assert np.mean(strong[weak]) > 0.95


def _column_profile(output, image):
    difference = np.abs(output.numpy().astype(int) - image.numpy().astype(int))
    return difference.sum(axis=(0, 2)).astype(float)


def test_glitter_column_persists_while_highlights_flicker():
    image = _scene(96, 256)
    horizon = Horizon(40.0, 40.0)
    first = _apply(
        image, "sun_glitter", 4, seed=SEED, episode_seed=EPISODE, horizon=horizon
    )
    second = _apply(
        image, "sun_glitter", 4, seed=SEED + 1, episode_seed=EPISODE, horizon=horizon
    )
    assert not np.array_equal(first.image.numpy(), second.image.numpy())
    columns = np.arange(256)
    centres = [
        np.average(columns, weights=_column_profile(result.image, image))
        for result in (first, second)
    ]
    assert abs(centres[0] - centres[1]) < 8


def test_motion_blur_angle_persists_with_small_frame_jitter():
    image = _scene(96, 128)
    first = _apply(image, "motion_blur", 2, seed=SEED, episode_seed=EPISODE).image
    jittered = _apply(
        image, "motion_blur", 2, seed=SEED + 1, episode_seed=EPISODE
    ).image
    other = _apply(image, "motion_blur", 2, seed=SEED, episode_seed=EPISODE + 5).image

    def difference(a, b):
        return np.mean(np.abs(a.numpy().astype(int) - b.numpy().astype(int)))

    assert 0 < difference(first, jittered) < difference(first, other)


def _streak_angle(output, image):
    """Dominant orientation in degrees of what rain added, from the structure tensor."""
    added = output.numpy().astype(float).sum(-1) - np.rint(image.numpy() * 0.7).sum(-1)
    gradient_y, gradient_x = np.gradient(added)
    return 0.5 * np.degrees(
        np.arctan2(
            2 * np.sum(gradient_x * gradient_y),
            np.sum(gradient_x**2) - np.sum(gradient_y**2),
        )
    )


def test_rain_slant_persists_while_drops_move():
    flat = tf.fill([192, 256, 3], tf.constant(60, tf.uint8))
    first = _apply(flat, "rain", 5, seed=SEED, episode_seed=EPISODE).image
    moved = _apply(flat, "rain", 5, seed=SEED + 1, episode_seed=EPISODE).image
    assert not np.array_equal(first.numpy(), moved.numpy())

    episodes = [EPISODE + offset for offset in range(3)]
    angles = [
        [
            _streak_angle(
                _apply(flat, "rain", 5, seed=SEED + frame, episode_seed=episode).image,
                flat,
            )
            for frame in range(3)
        ]
        for episode in episodes
    ]
    within_episode = max(np.ptp(frames) for frames in angles)
    across_episodes = np.ptp([np.mean(frames) for frames in angles])
    assert within_episode < 0.6
    assert across_episodes > 2.0


def _two_resolutions():
    large = _scene(192, 256, seed=3)
    small = tf.cast(
        tf.round(tf.image.resize(large, [96, 128], method="area")), tf.uint8
    )
    return small, large


@pytest.mark.parametrize(
    "name",
    [
        "haze",
        "lens_droplets",
        "motion_blur",
        "overexposure",
        "white_balance_cool",
        "white_balance_warm",
    ],
)
def test_renders_agree_pixelwise_across_resolutions(name):
    small, large = _two_resolutions()
    rendered_small = _apply(small, name, 3).image
    rendered_large = _apply(large, name, 3).image
    downsampled = tf.image.resize(rendered_large, [96, 128], method="area")
    difference = np.abs(downsampled.numpy() - rendered_small.numpy().astype(np.float32))
    assert difference.mean() < 2.0


@pytest.mark.parametrize("name", ["low_light", "rain", "shot_noise", "sun_glitter"])
def test_per_frame_noise_renders_agree_statistically_across_resolutions(name):
    small, large = _two_resolutions()
    rendered_small = _apply(small, name, 3).image.numpy().astype(np.float64)
    rendered_large = _apply(large, name, 3).image.numpy().astype(np.float64)
    assert abs(rendered_small.mean() - rendered_large.mean()) < 1.0
    assert abs(rendered_small.std() - rendered_large.std()) < 1.0


def test_haze_modes_follow_the_auxiliary_inputs():
    image = _scene()
    depth = tf.fill([48, 64], 300.0)
    camera = Horizon(20.0, 20.0, camera_height_m=2.0, focal_length_px=60.0)
    assert _apply(image, "haze", 2).record.mode == "homogeneous"
    assert _apply(image, "haze", 2, horizon=camera).record.mode == "horizon"
    by_depth = _apply(image, "haze", 2, depth_m=depth, horizon=camera).record
    assert by_depth.mode == "depth"
    assert by_depth.auxiliary_inputs == ("depth_m", "horizon")
    with pytest.raises(ValueError, match="camera_height_m"):
        _apply(image, "haze", 2, horizon=Horizon(20.0, 20.0))


def test_haze_hides_far_pixels_first():
    image = _scene()
    depth = np.full([48, 64], 10.0, np.float32)
    depth[:, 32:] = 1000.0
    hazed = _apply(image, "haze", 3, depth_m=tf.constant(depth)).image.numpy()
    change = np.abs(hazed.astype(int) - image.numpy().astype(int)).mean(axis=(0, 2))
    assert change[32:].mean() > 5 * change[:32].mean()

    infinite = np.full([48, 64], np.inf, np.float32)
    fogged = _apply(image, "haze", 3, depth_m=tf.constant(infinite)).image.numpy()
    assert np.ptp(fogged.reshape(-1, 3), axis=0).max() <= 1


def test_glitter_stays_inside_the_region():
    image = _scene(64, 96)
    region = np.zeros([64, 96], bool)
    region[40:, 10:60] = True
    result = _apply(image, "sun_glitter", 1, region=tf.constant(region))
    assert result.record.mode == "region"
    changed = np.any(result.image.numpy() != image.numpy(), axis=-1)
    assert changed.any()
    assert not changed[~region].any()


def test_horizon_coordinates_measure_distance_below_the_line():
    below, along = horizon_coordinates(
        tf.constant([10.0, 10.0]),
        100.0,
        tf.constant([50.0, 0.0]),
        tf.constant([15.0, 5.0]),
    )
    np.testing.assert_allclose(below.numpy(), [5.0, -5.0])
    np.testing.assert_allclose(along.numpy(), [0.0, -50.0])
    tilted, _ = horizon_coordinates(
        tf.constant([0.0, 100.0]), 100.0, tf.constant([0.0]), tf.constant([10.0])
    )
    np.testing.assert_allclose(tilted.numpy(), [10.0 / np.sqrt(2.0)], rtol=1e-6)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"left_row": float("nan"), "right_row": 1.0},
        {"left_row": 1.0, "right_row": 1.0, "camera_height_m": 2.0},
        {
            "left_row": 1.0,
            "right_row": 1.0,
            "camera_height_m": -2.0,
            "focal_length_px": 5.0,
        },
    ],
)
def test_horizon_rejects_invalid_geometry(kwargs):
    with pytest.raises(ValueError, match="Horizon"):
        Horizon(**kwargs)

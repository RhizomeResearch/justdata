import json
from dataclasses import replace

import numpy as np
import pytest
import tensorflow as tf

from justdata.vision.colorspace import (
    estimate_illuminant,
    linear_to_srgb,
    srgb_to_linear,
)
from justdata.vision.compositing import (
    BLENDING_MODES,
    BlendingRule,
    CompositeDescriptor,
    Instance,
    PlacementRule,
    composite_instances,
)
from justdata.vision.horizon import Horizon
from justdata.vision.poisson import seamless_clone, solve_poisson_dirichlet

HEIGHT, WIDTH = 96, 128
HORIZON_ROW = 40
SKY, WATER, BOAT = 1, 0, 3
SEED = tf.constant([5, 8], tf.int32)


def _scene():
    rng = np.random.default_rng(0)
    coarse = rng.uniform(0, 255, size=(HEIGHT // 8, WIDTH // 8, 3))
    image = tf.image.resize(coarse, [HEIGHT, WIDTH]).numpy()
    image[:HORIZON_ROW] = np.clip(image[:HORIZON_ROW] * 0.3 + 160, 0, 255)
    mask = np.full([HEIGHT, WIDTH], WATER, np.uint8)
    mask[:HORIZON_ROW] = SKY
    allowed = mask == WATER
    return np.round(image).astype(np.uint8), mask, allowed


def _boat(class_id=BOAT, size=(12, 16), source_id="boat", **kwargs):
    rng = np.random.default_rng(1)
    image = rng.integers(0, 256, size=(*size, 3), dtype=np.uint8)
    rows, columns = np.mgrid[: size[0], : size[1]]
    alpha = (
        ((rows - size[0] / 2 + 0.5) / (size[0] / 2)) ** 2
        + ((columns - size[1] / 2 + 0.5) / (size[1] / 2)) ** 2
    ) <= 1.0
    fields = dict(
        image=image,
        alpha=alpha,
        class_id=class_id,
        source_id=source_id,
        anchor_row_fraction=0.75,
    )
    return Instance(**(fields | kwargs))


def _composite(instances, *, placement=None, blending=None, seed=SEED, **kwargs):
    image, mask, allowed = _scene()
    arguments = dict(
        allowed=allowed,
        placement=placement or PlacementRule(min_visible_pixels=16),
        blending=blending or BlendingRule(),
        seed=seed,
    )
    return composite_instances(image, mask, instances, **(arguments | kwargs))


def test_hard_paste_copies_pixels_and_labels_only_its_footprint():
    image, mask, _allowed = _scene()
    boat = _boat()
    result = _composite([boat])
    (record,) = result.records
    assert record.status == "placed" and record.scale == 1.0
    top, left, bottom, right = record.box
    assert bottom - 1 == int(0.75 * HEIGHT)

    pasted = result.pasted.numpy()
    labels = result.mask.numpy()
    assert np.array_equal(pasted[top:bottom, left:right], boat.alpha)
    assert not pasted[:top].any() and not pasted[bottom:].any()
    assert np.all(labels[pasted] == BOAT)
    np.testing.assert_array_equal(labels[~pasted], mask[~pasted])
    output = result.image.numpy()[top:bottom, left:right]
    np.testing.assert_array_equal(output[boat.alpha], boat.image[boat.alpha])
    np.testing.assert_array_equal(result.image.numpy()[~pasted], image[~pasted])
    assert record.visible_pixels == int(boat.alpha.sum())


def test_every_request_is_placed_or_rejected_with_counted_reasons():
    sky_anchor = _boat(source_id="sky", anchor_row_fraction=0.1)
    too_wide = _boat(source_id="wide", size=(12, 200))
    instances = [_boat(source_id="a"), sky_anchor, too_wide, _boat(source_id="b")]
    placement = PlacementRule(min_visible_pixels=16, max_attempts=20)
    result = _composite(instances, placement=placement)
    records = {record.source_id: record for record in result.records}
    assert len(result.records) == len(instances)
    assert records["a"].status == records["b"].status == "placed"
    assert records["sky"].status == "rejected"
    assert records["sky"].reason == "no_allowed_anchor"
    assert records["wide"].reason == "outside_margin"
    assert records["wide"].attempts == 20
    assert dict(records["wide"].rejections) == {"outside_margin": 20}
    assert json.loads(json.dumps(records["wide"].to_dict()))["status"] == "rejected"


def test_pasted_instances_never_overlap_and_respect_constraints():
    placement = PlacementRule(
        scale="horizon",
        max_outside_allowed_share=0.0,
        min_visible_pixels=16,
        border_margin_px=4,
        max_attempts=200,
    )
    instances = [
        _boat(source_id=f"boat{i}", source_horizon_offset_px=30.0) for i in range(12)
    ]
    result = _composite(
        instances,
        placement=placement,
        horizon=Horizon(HORIZON_ROW, HORIZON_ROW),
        instance_map=np.zeros([HEIGHT, WIDTH], np.int32),
    )
    placed = [record for record in result.records if record.status == "placed"]
    assert len(placed) >= 3
    assert (
        sum(record.visible_pixels for record in placed) == result.pasted.numpy().sum()
    )
    _image, _mask, allowed = _scene()
    assert not np.any(result.pasted.numpy() & ~allowed)
    ids = result.instance_map.numpy()
    assert sorted(record.instance_id for record in placed) == list(
        range(1, len(placed) + 1)
    )
    for record in placed:
        top, left, bottom, right = record.box
        assert top >= 4 and left >= 4 and bottom <= HEIGHT - 4 and right <= WIDTH - 4
        assert np.sum(ids == record.instance_id) == record.visible_pixels
        below = record.anchor[0] + 0.5 - HORIZON_ROW
        assert record.scale == pytest.approx(below / 30.0, rel=1e-5)


def test_null_paste_moves_background_and_writes_no_label():
    image, mask, _allowed = _scene()
    result = _composite([_boat(class_id=None, source_id="null")])
    (record,) = result.records
    assert record.status == "placed" and record.null_source_anchor is not None
    np.testing.assert_array_equal(result.mask.numpy(), mask)
    assert not result.pasted.numpy().any()
    null = result.null_pasted.numpy()
    assert null.sum() == record.visible_pixels > 0
    shift = np.subtract(record.null_source_anchor, record.anchor)
    rows, columns = np.nonzero(null)
    np.testing.assert_array_equal(
        result.image.numpy()[rows, columns], image[rows + shift[0], columns + shift[1]]
    )


def test_null_paste_over_an_earlier_paste_restores_its_labels():
    _image, mask, _allowed = _scene()
    placement = PlacementRule(min_visible_pixels=16, allow_overlap=True)
    boat = _boat()
    result = _composite(
        [boat, _boat(class_id=None, source_id="null", size=(12, 30))],
        placement=placement,
    )
    labels = result.mask.numpy()
    pasted = result.pasted.numpy()
    null = result.null_pasted.numpy()
    assert not np.any(pasted & null)
    np.testing.assert_array_equal(labels[~pasted], mask[~pasted])
    assert result.records[0].visible_pixels == pasted.sum()


def _single(mode, **kwargs):
    return BlendingRule(mix=((mode, 1.0),), **kwargs)


def test_invalid_pixels_are_never_written():
    image, mask, _allowed = _scene()
    valid = np.ones([HEIGHT, WIDTH], bool)
    valid[:, WIDTH // 2 :] = False
    for mode in BLENDING_MODES:
        result = _composite(
            [_boat(source_id=str(i)) for i in range(4)],
            valid=valid,
            blending=_single(mode, edge_sigma_px=16.0),
        )
        np.testing.assert_array_equal(result.image.numpy()[~valid], image[~valid])
        np.testing.assert_array_equal(result.mask.numpy()[~valid], mask[~valid])


def test_compositing_is_deterministic_for_a_seed():
    instances = [_boat(source_id=str(i)) for i in range(3)]
    first = _composite(instances)
    second = _composite(instances)
    other = _composite(instances, seed=SEED + 1)
    np.testing.assert_array_equal(first.image.numpy(), second.image.numpy())
    assert [r.anchor for r in first.records] == [r.anchor for r in second.records]
    assert [r.anchor for r in first.records] != [r.anchor for r in other.records]


def _flat_scene(value=0):
    image = np.full([HEIGHT, WIDTH, 3], value, np.uint8)
    return image, np.zeros([HEIGHT, WIDTH], np.uint8), np.ones([HEIGHT, WIDTH], bool)


def _mast_boat(object_rgb, context_rgb, **kwargs):
    """A hull with a one-pixel mast: the mast is thinner than any soft edge."""
    alpha = np.zeros([20, 24], bool)
    alpha[10:, 4:20] = True
    alpha[:10, 11] = True
    crop = np.where(alpha[..., None], object_rgb, context_rgb).astype(np.uint8)
    return Instance(crop, alpha, BOAT, "mast_boat", 0.75, **kwargs)


@pytest.mark.parametrize("mode", BLENDING_MODES)
def test_labels_mark_exactly_the_pixels_rendered_at_least_half_object(mode):
    # White on black makes each output value the rendered object weight times 255.
    image, mask, allowed = _flat_scene(0)
    result = composite_instances(
        image,
        mask,
        [_mast_boat((255, 255, 255), (0, 0, 0), has_context=True)],
        allowed=allowed,
        placement=PlacementRule(min_visible_pixels=16),
        blending=_single(mode, edge_sigma_px=16.0),
        seed=SEED,
    )
    weight_at_least_half = result.image.numpy()[..., 0] >= 128
    np.testing.assert_array_equal(result.pasted.numpy(), weight_at_least_half)
    assert result.records[0].visible_pixels == weight_at_least_half.sum() > 100


@pytest.mark.parametrize(("has_context", "bleeds"), [(False, False), (True, True)])
def test_soft_edges_stay_near_the_object_and_use_context_only_when_declared(
    has_context, bleeds
):
    image, mask, allowed = _flat_scene(0)
    result = composite_instances(
        image,
        mask,
        [_mast_boat((128, 128, 128), (255, 0, 255), has_context=has_context)],
        allowed=allowed,
        placement=PlacementRule(min_visible_pixels=16),
        blending=_single("soft_edge", edge_sigma_px=16.0),
        seed=SEED,
    )
    output = result.image.numpy().astype(int)
    top, left, bottom, right = result.records[0].box
    reach = 6  # three edge sigmas: 16 px at width 1024 is 2 px here
    near = np.zeros([HEIGHT, WIDTH], bool)
    near[top - reach : bottom + reach, left - reach : right + reach] = True
    assert not np.any(output[~near])
    magenta = output[..., 0] - output[..., 1]
    assert (magenta.max() > 2) == bleeds


def _linear(srgb_uint8):
    return srgb_to_linear(np.asarray(srgb_uint8, np.float32) / 255.0).numpy()


def _textured_mast_boat(context_rgb, **kwargs):
    boat = _mast_boat((0, 0, 0), context_rgb, **kwargs)
    texture = np.random.default_rng(3).integers(120, 200, size=boat.image.shape)
    image = np.where(boat.alpha[..., None], texture, boat.image).astype(np.uint8)
    return replace(boat, image=image)


def _paste_on(image, boat, mode="harmonized", strength=1.0):
    mask = np.zeros([HEIGHT, WIDTH], np.uint8)
    return composite_instances(
        image,
        mask,
        [boat],
        allowed=np.ones([HEIGHT, WIDTH], bool),
        placement=PlacementRule(min_visible_pixels=16),
        blending=_single(mode, harmonization_strength=strength),
        seed=SEED,
    )


def _pasted_object(result, boat):
    top, left, bottom, right = result.records[0].box
    return result.image.numpy()[top:bottom, left:right][boat.alpha]


@pytest.mark.parametrize("strength", [1.0, 0.5])
def test_harmonized_applies_the_ratio_of_frame_illuminants(strength):
    image, _mask, _allowed = _scene()
    destination = estimate_illuminant(image / 255.0).numpy()
    warmer_and_brighter = destination * np.array([2.5, 2.0, 1.5], np.float32)
    boat = _textured_mast_boat(
        (0, 0, 0), source_illuminant=tuple(warmer_and_brighter.tolist())
    )
    result = _paste_on(image, boat, strength=strength)
    gains = (destination / warmer_and_brighter) ** strength
    expected = linear_to_srgb(_linear(boat.image[boat.alpha]) * gains).numpy() * 255
    np.testing.assert_allclose(_pasted_object(result, boat), expected, atol=1.0)
    recorded = dict(result.records[0].blending)["gains"]
    np.testing.assert_allclose(recorded, gains, rtol=1e-5)


def test_harmonized_ignores_the_surroundings_of_the_object():
    # Under the same light, the object keeps its colours whatever the water
    # around it looks like in either frame.
    image, _mask, _allowed = _scene()
    light = tuple(estimate_illuminant(image / 255.0).numpy().tolist())
    boat = _textured_mast_boat(
        (100, 100, 100), has_context=True, source_illuminant=light
    )
    rows, columns = np.nonzero(boat.alpha)
    interior = (rows >= 12) & (columns >= 6) & (columns < 18)
    harmonized = _pasted_object(_paste_on(image, boat), boat)
    np.testing.assert_allclose(
        harmonized[interior], boat.image[boat.alpha][interior], atol=1.0
    )

    without_light = replace(boat, source_illuminant=None)
    plain = _paste_on(image, without_light)
    soft = _paste_on(image, without_light, mode="soft_edge")
    np.testing.assert_array_equal(plain.image.numpy(), soft.image.numpy())
    assert dict(plain.records[0].blending)["gains"] is None


def test_estimate_illuminant_follows_light_and_exposure_not_resolution():
    coarse = np.random.default_rng(5).uniform(0.0, 0.75, size=(12, 16, 3))
    scene = tf.image.resize(coarse.astype(np.float32), [96, 128]).numpy()
    plain = estimate_illuminant(linear_to_srgb(scene)).numpy()
    gains = np.array([1.2, 0.9, 0.6], np.float32)
    lit = estimate_illuminant(linear_to_srgb(scene * gains)).numpy()
    np.testing.assert_allclose(lit / plain, gains, rtol=1e-4)
    larger = tf.image.resize(coarse.astype(np.float32), [192, 256]).numpy()
    np.testing.assert_allclose(
        estimate_illuminant(linear_to_srgb(larger)).numpy(), plain, rtol=0.1
    )
    with pytest.raises(tf.errors.InvalidArgumentError, match="edges"):
        estimate_illuminant(np.full([16, 16, 3], 0.5, np.float32))


@pytest.mark.parametrize(
    "illuminant", [(1.0, 1.0), (1.0, 0.0, 1.0), (1.0, float("nan"), 1.0)]
)
def test_instance_rejects_invalid_source_illuminants(illuminant):
    with pytest.raises(ValueError, match="source_illuminant"):
        _composite([_boat(source_illuminant=illuminant)])


def test_each_paste_draws_its_mode_from_the_mix():
    rule = BlendingRule(mix=(("hard", 1.0), ("soft_edge", 1.0), ("harmonized", 2.0)))
    assert [rule.draw_mode(d) for d in (0.0, 0.24, 0.26, 0.49, 0.51, 0.99)] == [
        "hard",
        "hard",
        "soft_edge",
        "soft_edge",
        "harmonized",
        "harmonized",
    ]
    instances = [_boat(source_id=str(i), size=(4, 6)) for i in range(24)]
    placement = PlacementRule(min_visible_pixels=4, allow_overlap=True)
    first = _composite(instances, placement=placement, blending=rule)
    again = _composite(instances, placement=placement, blending=rule)
    modes = [dict(record.blending)["mode"] for record in first.records]
    assert set(modes) == set(BLENDING_MODES)
    assert modes == [dict(record.blending)["mode"] for record in again.records]


def test_horizon_scaling_places_crops_only_where_their_scale_fits():
    horizon = Horizon(HORIZON_ROW, HORIZON_ROW)
    placement = PlacementRule(
        scale="horizon", min_scale=0.8, max_scale=1.2, min_visible_pixels=4
    )
    fits = [
        _boat(source_id=f"fits{i}", source_horizon_offset_px=30.0) for i in range(4)
    ]
    too_far = _boat(source_id="too_far", source_horizon_offset_px=100.0)
    result = _composite(fits + [too_far], placement=placement, horizon=horizon)
    placed = [record for record in result.records if record.status == "placed"]
    assert len(placed) == 4
    assert all(0.8 <= record.scale <= 1.2 for record in placed)
    rejected = result.records[-1]
    assert (rejected.reason, rejected.attempts) == ("scale_out_of_range", 1)

    keep_row = _composite(
        [_boat()], placement=PlacementRule(min_scale=1.5, max_scale=2.0)
    )
    assert keep_row.records[0].reason == "scale_out_of_range"


def test_composite_identity_follows_the_rules():
    first = CompositeDescriptor(PlacementRule(), BlendingRule())
    second = CompositeDescriptor(PlacementRule(), _single("soft_edge"))
    assert first.identity.startswith("composite@2.0.0:sha256:")
    assert first.identity != second.identity
    assert (
        _composite([_boat()]).identity
        == CompositeDescriptor(
            PlacementRule(min_visible_pixels=16), BlendingRule()
        ).identity
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mix": ()},
        {"mix": (("poisson", 1.0),)},
        {"mix": (("hard", 1.0), ("hard", 2.0))},
        {"mix": (("hard", 0.0),)},
        {"edge_sigma_px": 0.0},
        {"harmonization_strength": 1.5},
    ],
)
def test_blending_rule_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError, match="BlendingRule|edge_sigma|harmonization"):
        BlendingRule(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_scale": 2.0, "max_scale": 1.0},
        {"max_scale": float("inf")},
        {"max_scale": 0.0},
    ],
)
def test_placement_rule_rejects_invalid_scale_ranges(kwargs):
    with pytest.raises(ValueError, match="min_scale"):
        PlacementRule(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"placement": PlacementRule(scale="horizon")}, "horizon"),
        ({"seed": tf.constant([1, 2], tf.int64)}, "seed"),
        ({"valid": np.ones([HEIGHT, WIDTH - 1], bool)}, "valid"),
    ],
)
def test_composite_rejects_invalid_calls(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _composite([_boat(source_horizon_offset_px=10.0)], **kwargs)


def test_compositing_runs_eagerly_only():
    image, mask, allowed = _scene()
    boat = _boat()

    @tf.function
    def traced():
        composite_instances(
            image,
            mask,
            [boat],
            allowed=allowed,
            placement=PlacementRule(),
            blending=BlendingRule(),
            seed=SEED,
        )
        return tf.constant(0)

    with pytest.raises(RuntimeError, match="eagerly"):
        traced()


def test_poisson_solver_recovers_a_known_solution():
    rng = np.random.default_rng(2)
    expected = rng.uniform(0, 1, size=(2, 13, 17))
    laplacian = np.zeros_like(expected)
    laplacian[:, 1:-1, 1:-1] = (
        expected[:, :-2, 1:-1]
        + expected[:, 2:, 1:-1]
        + expected[:, 1:-1, :-2]
        + expected[:, 1:-1, 2:]
        - 4 * expected[:, 1:-1, 1:-1]
    )
    solution = solve_poisson_dirichlet(tf.constant(expected), tf.constant(laplacian))
    np.testing.assert_allclose(solution.numpy(), expected, atol=1e-12)


def test_seamless_clone_of_the_destination_keeps_it():
    image, _mask, _allowed = _scene()
    patch_mask = np.zeros([30, 40], bool)
    patch_mask[3:-3, 3:-3] = True
    patch = image[20:50, 30:70]
    cloned = seamless_clone(image, patch, patch_mask, (20, 30)).numpy()
    assert np.abs(cloned.astype(int) - image.astype(int)).max() <= 1
    with pytest.raises(ValueError, match="inside"):
        seamless_clone(image, patch, patch_mask, (HEIGHT - 10, 30))

import json

import numpy as np
import pytest
import tensorflow as tf

from justdata.vision.colorspace import rgb_to_lab
from justdata.vision.compositing import (
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


def test_invalid_pixels_are_never_written():
    image, mask, _allowed = _scene()
    valid = np.ones([HEIGHT, WIDTH], bool)
    valid[:, WIDTH // 2 :] = False
    for mode in ("hard", "feather", "poisson"):
        result = _composite(
            [_boat(source_id=str(i)) for i in range(4)],
            valid=valid,
            blending=BlendingRule(mode=mode),
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


def test_feather_softens_edges_but_keeps_the_label_footprint():
    hard = _composite([_boat()])
    feather = _composite(
        [_boat()], blending=BlendingRule(mode="feather", feather_sigma_px=8.0)
    )
    assert feather.records[0].anchor == hard.records[0].anchor
    hard_footprint = hard.pasted.numpy()
    soft_footprint = feather.pasted.numpy()
    assert np.mean(hard_footprint != soft_footprint) < 0.002
    image, _mask, _allowed = _scene()
    changed = np.any(feather.image.numpy() != image, axis=-1)
    assert np.sum(changed & ~soft_footprint) > 0
    assert dict(feather.records[0].blending)["feather_sigma_px"] == pytest.approx(1.0)


def test_poisson_blending_changes_only_its_region():
    image, _mask, _allowed = _scene()
    result = _composite([_boat()], blending=BlendingRule(mode="poisson"))
    (record,) = result.records
    top, left, bottom, right = record.box
    changed = np.any(result.image.numpy() != image, axis=-1)
    outside = np.ones_like(changed)
    outside[top - 1 : bottom + 1, left - 1 : right + 1] = False
    assert changed.any() and not changed[outside].any()


def test_harmonization_moves_instance_colour_toward_the_background():
    image, _mask, _allowed = _scene()
    bright = _boat()
    bright = Instance(
        image=np.full_like(bright.image, 250),
        alpha=bright.alpha,
        class_id=BOAT,
        source_id="bright",
        anchor_row_fraction=0.75,
    )
    plain = _composite([bright])
    harmonized = _composite([bright], blending=BlendingRule(harmonization_strength=1.0))
    footprint = plain.pasted.numpy()
    # The ring is 8 px at width 1024, so one pixel at this width.
    padded = np.pad(footprint, 1)
    dilated = np.zeros_like(footprint)
    for dy in range(3):
        for dx in range(3):
            dilated |= padded[dy : dy + HEIGHT, dx : dx + WIDTH]

    def lab_mean(pixels):
        return rgb_to_lab(pixels.astype(np.float32) / 255.0).numpy().mean(axis=0)

    ring_mean = lab_mean(image[dilated & ~footprint])
    plain_mean = lab_mean(plain.image.numpy()[footprint])
    harmonized_mean = lab_mean(harmonized.image.numpy()[footprint])
    np.testing.assert_allclose(harmonized_mean, ring_mean, atol=1.0)
    assert np.abs(plain_mean - ring_mean).max() > 20
    assert dict(harmonized.records[0].blending)["harmonized"] is True


def test_composite_identity_follows_the_rules():
    first = CompositeDescriptor(PlacementRule(), BlendingRule())
    second = CompositeDescriptor(PlacementRule(), BlendingRule(mode="feather"))
    assert first.identity.startswith("composite@1.0.0:sha256:")
    assert first.identity != second.identity
    assert (
        _composite([_boat()]).identity
        == CompositeDescriptor(
            PlacementRule(min_visible_pixels=16), BlendingRule()
        ).identity
    )


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

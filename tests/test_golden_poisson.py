"""Compare Poisson cloning with OpenCV ``seamlessClone(NORMAL_CLONE)``.

OpenCV solves in float32 and justdata in float64, and both truncate, so a value
within float32 rounding of an integer may land one level apart.
"""

import numpy as np
import pytest
import tensorflow as tf

from justdata.vision.poisson import clone_bounds, seamless_clone

pytestmark = pytest.mark.golden


def _smooth(rng, shape):
    coarse = rng.uniform(0, 255, size=(shape[0] // 8 + 1, shape[1] // 8 + 1, 3))
    return tf.image.resize(coarse, shape).numpy() + rng.normal(0, 8, (*shape, 3))


@pytest.mark.parametrize(
    ("destination_shape", "source_shape", "offset"),
    [
        ((120, 160), (50, 70), (40, 40)),
        ((300, 400), (90, 140), (100, 100)),
        ((1242, 2208), (300, 500), (414, 552)),
    ],
)
def test_seamless_clone_matches_opencv_normal_clone(
    destination_shape, source_shape, offset
):
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(sum(destination_shape))
    destination = np.clip(_smooth(rng, destination_shape), 0, 255).astype(np.uint8)
    source = np.clip(_smooth(rng, source_shape), 0, 255).astype(np.uint8)
    height, width = source_shape
    rows, columns = np.mgrid[:height, :width]
    mask = (
        ((rows - height / 2) / (height / 2.5)) ** 2
        + ((columns - width / 2) / (width / 2.4)) ** 2
        <= 1
    ).astype(np.uint8) * 255

    top, left, bottom, right = clone_bounds(mask > 0)
    centre = (
        offset[1] + left + (right - left) // 2,
        offset[0] + top + (bottom - top) // 2,
    )
    # OpenCV writes its eroded mask back into the array it is given.
    reference = cv2.seamlessClone(
        source[..., ::-1].copy(),
        destination[..., ::-1].copy(),
        mask.copy(),
        centre,
        cv2.NORMAL_CLONE,
    )[..., ::-1]
    cloned = seamless_clone(destination, source, mask > 0, offset).numpy()

    difference = np.abs(cloned.astype(int) - reference.astype(int))
    assert np.mean(np.any(reference != destination, axis=-1)) > 0.03
    assert difference.max() <= 1
    assert np.mean(difference > 0) < 0.001

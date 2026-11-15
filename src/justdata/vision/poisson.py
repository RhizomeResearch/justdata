"""Gradient-domain (Poisson) cloning that follows OpenCV's ``seamlessClone``.

The steps are those of OpenCV's ``NORMAL_CLONE`` (``modules/photo/src/
seamless_cloning*.cpp``): the mask's outer one-pixel ring is cleared and its
tight bounding box becomes the region of interest; the mask is eroded three
times by a 3 x 3 square; forward-difference gradients come from the masked
source inside the eroded mask and from the destination elsewhere; and the
Poisson equation is solved with the destination on the region's outer ring as
the Dirichlet boundary. The solve uses a type-I discrete sine transform built on
the FFT, in float64, so it is exact up to rounding. The result is clipped to
``[0, 255]`` and truncated toward zero, as OpenCV does.
"""

import numpy as np
import tensorflow as tf

ERODE_ITERATIONS = 3


def _dst1(values: tf.Tensor) -> tf.Tensor:
    """Type-I DST along the last axis: ``y_k = sum_n x_n sin(pi k n / (N + 1))``."""
    zeros = tf.zeros_like(values[..., :1])
    extended = tf.concat([zeros, values, zeros, -tf.reverse(values, [-1])], axis=-1)
    spectrum = tf.signal.fft(tf.complex(extended, tf.zeros_like(extended)))
    count = tf.shape(values)[-1]
    return -tf.math.imag(spectrum[..., 1 : count + 1]) / 2.0


def _dst1_2d(values: tf.Tensor) -> tf.Tensor:
    """DST-I over the last two axes of ``[C, H, W]``."""
    rows = _dst1(values)
    return tf.linalg.matrix_transpose(_dst1(tf.linalg.matrix_transpose(rows)))


def solve_poisson_dirichlet(
    boundary: tf.Tensor,
    laplacian: tf.Tensor,
) -> tf.Tensor:
    """Solve ``4-neighbour Laplacian(u) = laplacian`` inside ``[C, H, W]``.

    ``boundary`` supplies the values on the outer one-pixel ring, which the
    solution keeps; ``laplacian`` is read on the interior only.
    """
    boundary = tf.cast(boundary, tf.float64)
    laplacian = tf.cast(laplacian, tf.float64)
    height, width = tf.shape(boundary)[1], tf.shape(boundary)[2]
    ring = tf.pad(
        tf.zeros([height - 2, width - 2], tf.float64),
        [[1, 1], [1, 1]],
        constant_values=1.0,
    )
    ring_only = boundary * ring
    ring_neighbours = (
        ring_only[:, :-2, 1:-1]
        + ring_only[:, 2:, 1:-1]
        + ring_only[:, 1:-1, :-2]
        + ring_only[:, 1:-1, 2:]
    )
    right_hand_side = laplacian[:, 1:-1, 1:-1] - ring_neighbours

    inner_height = tf.cast(height - 2, tf.float64)
    inner_width = tf.cast(width - 2, tf.float64)
    ky = tf.cast(tf.range(1, height - 1), tf.float64)
    kx = tf.cast(tf.range(1, width - 1), tf.float64)
    eigen_y = 2.0 * tf.cos(np.pi * ky / (inner_height + 1.0)) - 2.0
    eigen_x = 2.0 * tf.cos(np.pi * kx / (inner_width + 1.0)) - 2.0
    eigenvalues = eigen_y[:, tf.newaxis] + eigen_x[tf.newaxis, :]
    spectrum = _dst1_2d(right_hand_side) / eigenvalues
    scale = 4.0 / ((inner_height + 1.0) * (inner_width + 1.0))
    interior = _dst1_2d(spectrum) * scale
    return tf.pad(interior, [[0, 0], [1, 1], [1, 1]]) + ring_only


def _forward_gradients(image: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """``I(x + 1) - I(x)`` and ``I(y + 1) - I(y)``, zero on the last column or row."""
    gradient_x = tf.pad(image[:, :, 1:] - image[:, :, :-1], [[0, 0], [0, 0], [0, 1]])
    gradient_y = tf.pad(image[:, 1:, :] - image[:, :-1, :], [[0, 0], [0, 1], [0, 0]])
    return gradient_x, gradient_y


def _backward_divergence(gradient_x: tf.Tensor, gradient_y: tf.Tensor) -> tf.Tensor:
    """``g(x) - g(x - 1)`` summed over both axes, valid on the interior."""
    divergence_x = tf.pad(
        gradient_x[:, :, 1:] - gradient_x[:, :, :-1], [[0, 0], [0, 0], [1, 0]]
    )
    divergence_y = tf.pad(
        gradient_y[:, 1:, :] - gradient_y[:, :-1, :], [[0, 0], [1, 0], [0, 0]]
    )
    return divergence_x + divergence_y


def _erode(mask: tf.Tensor) -> tf.Tensor:
    """3 x 3 erosion repeated ``ERODE_ITERATIONS`` times, zero outside the region."""
    values = mask[tf.newaxis, :, :, tf.newaxis]
    for _ in range(ERODE_ITERATIONS):
        padded = tf.pad(values, [[0, 0], [1, 1], [1, 1], [0, 0]])
        values = -tf.nn.max_pool2d(-padded, ksize=3, strides=1, padding="VALID")
    return values[0, :, :, 0]


def clone_bounds(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """``(top, left, bottom, right)`` of the mask after clearing its outer ring."""
    inner = np.zeros_like(mask, dtype=bool)
    inner[1:-1, 1:-1] = mask[1:-1, 1:-1]
    inside = np.argwhere(inner)
    if inside.size == 0:
        return None
    top, left = inside.min(axis=0)
    bottom, right = inside.max(axis=0) + 1
    return int(top), int(left), int(bottom), int(right)


def clone_region(
    destination: tf.Tensor,
    source: tf.Tensor,
    mask: tf.Tensor,
) -> tf.Tensor:
    """Float64 ``[h, w, 3]`` solution over one region, before quantization.

    Regions thinner than three pixels have no interior and keep the destination.
    """
    if destination.shape[0] < 3 or destination.shape[1] < 3:
        return tf.cast(destination, tf.float64)
    destination = tf.transpose(tf.cast(destination, tf.float64), [2, 0, 1])
    mask = tf.cast(mask, tf.float64)
    source = tf.transpose(tf.cast(source, tf.float64), [2, 0, 1]) * mask
    eroded = _erode(mask)
    destination_x, destination_y = _forward_gradients(destination)
    source_x, source_y = _forward_gradients(source)
    guidance_x = destination_x * (1.0 - eroded) + source_x * eroded
    guidance_y = destination_y * (1.0 - eroded) + source_y * eroded
    laplacian = _backward_divergence(guidance_x, guidance_y)
    solution = solve_poisson_dirichlet(destination, laplacian)
    return tf.transpose(solution, [1, 2, 0])


def seamless_clone(
    destination: tf.Tensor,
    source: tf.Tensor,
    mask: tf.Tensor,
    offset: tuple[int, int],
) -> tf.Tensor:
    """Clone ``source`` into ``destination`` like OpenCV ``NORMAL_CLONE``.

    ``source`` (``[h, w, 3]``) and ``mask`` (``[h, w]``, non-zero = inside) share a
    frame whose pixel ``(0, 0)`` lands on ``destination`` pixel ``offset``
    ``(row, column)``. Only the bounding box of the mask, after clearing its
    outer ring, changes; it must lie inside ``destination``. Returns ``uint8``.
    """
    output = np.array(destination, dtype=np.uint8)
    mask = np.asarray(mask).astype(bool)
    bounds = clone_bounds(mask)
    if bounds is None:
        return tf.constant(output)
    top, left, bottom, right = bounds
    row, column = offset[0] + top, offset[1] + left
    height, width = bottom - top, right - left
    if (
        row < 0
        or column < 0
        or row + height > output.shape[0]
        or column + width > output.shape[1]
    ):
        raise ValueError("The cloned region must lie inside the destination image.")

    solution = clone_region(
        tf.constant(output[row : row + height, column : column + width]),
        tf.convert_to_tensor(np.asarray(source)[top:bottom, left:right]),
        tf.constant(mask[top:bottom, left:right]),
    )
    output[row : row + height, column : column + width] = np.clip(
        solution.numpy(), 0.0, 255.0
    ).astype(np.uint8)
    return tf.constant(output)


__all__ = ["clone_bounds", "clone_region", "seamless_clone", "solve_poisson_dirichlet"]

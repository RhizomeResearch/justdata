"""Horizon-line geometry for scene-aware corruptions and compositing."""

import math
from dataclasses import dataclass

import tensorflow as tf


@dataclass(frozen=True, slots=True)
class Horizon:
    """A straight horizon line and, optionally, the camera that sees it.

    ``left_row`` and ``right_row`` are the line's rows at the left (``x = 0``)
    and right (``x = width``) image edges, in pixels of the image it is used
    with; pixel centres sit at ``row + 0.5``. Rows may fall outside the image.
    ``camera_height_m`` above the water and ``focal_length_px`` (in pixels of the
    same image) are needed only for distance estimates on flat water.
    """

    left_row: float
    right_row: float
    camera_height_m: float | None = None
    focal_length_px: float | None = None

    def __post_init__(self) -> None:
        for field in ("left_row", "right_row"):
            value = getattr(self, field)
            if isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"Horizon {field} must be a finite number.")
        if self.camera_height_m is None and self.focal_length_px is None:
            return
        if self.camera_height_m is None or self.focal_length_px is None:
            raise ValueError(
                "Horizon camera_height_m and focal_length_px must be given together."
            )
        for value in (self.camera_height_m, self.focal_length_px):
            if not (math.isfinite(value) and value > 0):
                raise ValueError(
                    "Horizon camera_height_m and focal_length_px must be finite and "
                    "positive."
                )


def horizon_coordinates(
    rows: tf.Tensor,
    width: tf.Tensor,
    x: tf.Tensor,
    y: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Return ``(below, along)`` for points ``(x, y)`` in continuous pixel coordinates.

    ``rows`` holds the line's rows at ``x = 0`` and ``x = width``. ``below`` is
    the signed perpendicular distance below the line (positive under it) and
    ``along`` the offset along the line from the image's centre column.
    """
    rows = tf.cast(rows, tf.float32)
    width = tf.cast(width, tf.float32)
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    rise = rows[1] - rows[0]
    norm = tf.sqrt(tf.square(width) + tf.square(rise))
    dx = x - 0.5 * width
    dy = y - 0.5 * (rows[0] + rows[1])
    below = (dy * width - dx * rise) / norm
    along = (dx * width + dy * rise) / norm
    return below, along


def horizon_pixel_coordinates(
    rows: tf.Tensor,
    height: tf.Tensor,
    width: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """``horizon_coordinates`` evaluated at every ``[H, W]`` pixel centre."""
    y = tf.cast(tf.range(height), tf.float32) + 0.5
    x = tf.cast(tf.range(width), tf.float32) + 0.5
    grid_y, grid_x = tf.meshgrid(y, x, indexing="ij")
    return horizon_coordinates(rows, width, grid_x, grid_y)


__all__ = ["Horizon", "horizon_coordinates", "horizon_pixel_coordinates"]

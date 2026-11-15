"""Render a condition that drifts over an episode of frames, and record each frame.

Usage:
    python examples/vision/maritime_conditions.py OUTPUT_DIR FRAME.png [FRAME.png ...]

Lens droplets ramp from strength 0 to 3 over the first half of the frames, then
hold. The droplet layout comes from the episode seed, so droplets stay put while
new ones fade in. Each frame is written as a PNG next to a JSON record.
"""

import json
import sys
from pathlib import Path

import tensorflow as tf

from justdata.vision.corruptions import apply_corruption_at


def drift(index: int, count: int, peak: float = 3.0) -> float:
    ramp_frames = max(1, count // 2)
    return peak * min(1.0, index / ramp_frames)


def main() -> None:
    output_dir = Path(sys.argv[1])
    frames = sys.argv[2:]
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_seed = tf.constant([2026, 1], tf.int32)
    for index, path in enumerate(frames):
        image = tf.io.decode_png(tf.io.read_file(path), channels=3)
        result = apply_corruption_at(
            image,
            name="lens_droplets",
            version="1.0.0",
            strength=drift(index, len(frames)),
            seed=tf.constant([index, 1], tf.int32),
            episode_seed=episode_seed,
        )
        stem = output_dir / Path(path).stem
        tf.io.write_file(f"{stem}.png", tf.io.encode_png(result.image))
        stem.with_suffix(".json").write_text(
            json.dumps(result.record.to_dict(), indent=2, sort_keys=True)
        )


if __name__ == "__main__":
    main()

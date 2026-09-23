"""Build one recorded semantic view with static class-mask targets."""

import tensorflow as tf

import justdata.vision  # noqa: F401
from justdata.core import get_pipeline


def main():
    pipeline = get_pipeline(
        pipeline_name="vision/segmentation",
        apply_presets=False,
        overrides={
            "geometry_kwargs": {
                "class_values": (0, 1, 2),
                "ignore_value": 255,
                "eval_long_side": 8,
                "patch_size": 4,
            },
            "postproc_kwargs": {
                "normalize_image": False,
                "emit_semantic_targets": True,
            },
        },
    )
    preprocess, _, _, postprocess = pipeline.build(is_training=False)
    source = {
        "image": tf.zeros([3, 5, 3], tf.uint8),
        "mask": tf.constant(
            [[0, 0, 255, 2, 2], [0, 1, 255, 2, 2], [0, 0, 255, 2, 2]],
            tf.int32,
        ),
        "metadata": {"example_id": tf.constant("synthetic-1")},
    }
    view = postprocess(preprocess(source))
    print("sample:", view["metadata"]["example_id"].numpy().decode())
    print("class IDs:", view["targets"]["class_ids"].numpy())
    print("present slots:", view["targets"]["target_valid_mask"].numpy())
    print(
        "valid pixels:",
        int(tf.reduce_sum(tf.cast(view["targets"]["pixel_valid_mask"], tf.int32))),
    )


if __name__ == "__main__":
    main()

import tensorflow as tf


@tf.function
def bboxes_to_heatmaps(image_tensor, bboxes_tensor, labels_tensor, num_classes):
    """
    Converts normalized bounding boxes [ymin, xmin, ymax, xmax] to heatmaps.

    Args:
        image_tensor: The image tensor (H, W, C).
        bboxes_tensor: The bounding boxes tensor (N, 4), normalized coordinates.
        labels_tensor: The class labels tensor (N,).
        num_classes: The total number of classes (e.g., 20 for VOC).

    Returns:
        A tensor of heatmaps (H, W, num_classes).
    """
    height = tf.shape(image_tensor)[0]
    width = tf.shape(image_tensor)[1]
    num_objects = tf.shape(bboxes_tensor)[0]

    initial_heatmaps = tf.zeros((height, width, num_classes), dtype=tf.float32)

    # Loop condition and body
    def condition(i, heatmaps):
        return i < num_objects

    def body(i, heatmaps):
        bbox = bboxes_tensor[i]
        label = tf.cast(labels_tensor[i], tf.int32)

        # Denormalize coordinates
        y1_f = bbox[0] * tf.cast(height, tf.float32)
        x1_f = bbox[1] * tf.cast(width, tf.float32)
        y2_f = bbox[2] * tf.cast(height, tf.float32)
        x2_f = bbox[3] * tf.cast(width, tf.float32)

        y1 = tf.cast(tf.math.floor(y1_f), tf.int32)
        x1 = tf.cast(tf.math.floor(x1_f), tf.int32)
        y2 = tf.cast(tf.math.ceil(y2_f), tf.int32)
        x2 = tf.cast(tf.math.ceil(x2_f), tf.int32)

        # Clip coordinates to be within image bounds
        y1 = tf.maximum(0, y1)
        x1 = tf.maximum(0, x1)
        y2 = tf.minimum(height, y2)
        x2 = tf.minimum(width, x2)

        # Update heatmaps if the box is valid
        def update_heatmap():
            rows, cols = tf.meshgrid(tf.range(y1, y2), tf.range(x1, x2), indexing="ij")
            label_indices = tf.fill(tf.shape(rows), label)

            # Shape: (num_pixels_in_box, 3) where each row is [row_idx, col_idx, label]
            indices = tf.stack(
                [
                    tf.reshape(rows, [-1]),
                    tf.reshape(cols, [-1]),
                    tf.reshape(label_indices, [-1]),
                ],
                axis=1,
            )

            # Update the main class_heatmaps tensor at the specific class slice
            updates = tf.ones(tf.shape(indices)[0], dtype=tf.float32)
            updated_heatmaps = tf.tensor_scatter_nd_add(heatmaps, indices, updates)
            return updated_heatmaps

        # Unchanged heatmaps if the box is invalid
        def no_update():
            return heatmaps

        # Conditionally update the heatmap based on box validity
        is_valid_box = tf.logical_and(y1 < y2, x1 < x2)
        updated_heatmaps = tf.cond(is_valid_box, update_heatmap, no_update)

        return i + 1, updated_heatmaps

    _, final_heatmaps = tf.while_loop(
        condition,
        body,
        loop_vars=[tf.constant(0), initial_heatmaps],
        shape_invariants=[
            tf.TensorShape([]),  # Shape of loop counter 'i'
            tf.TensorShape([None, None, num_classes]),  # Shape of heatmaps
        ],
    )

    # Clip the final heatmap values to be between 0 and 1
    final_heatmaps = tf.clip_by_value(final_heatmaps, 0.0, 1.0)

    return final_heatmaps


@tf.function
def bboxes_to_gaussian_heatmaps(
    image_tensor, bboxes_tensor, labels_tensor, num_classes
):
    """
    Converts normalized bounding boxes [ymin, xmin, ymax, xmax] to Gaussian heatmaps
    confined strictly within the bounding box.

    Args:
        image_tensor: The image tensor (H, W, C).
        bboxes_tensor: The bounding boxes tensor (N, 4), normalized coordinates.
        labels_tensor: The class labels tensor (N,).
        num_classes: The total number of classes (e.g., 20 for VOC).

    Returns:
        A tensor of Gaussian heatmaps (H, W, num_classes).
    """
    height = tf.shape(image_tensor)[0]
    width = tf.shape(image_tensor)[1]
    num_objects = tf.shape(bboxes_tensor)[0]

    initial_heatmaps = tf.zeros((height, width, num_classes), dtype=tf.float32)

    y_grid, x_grid = tf.meshgrid(
        tf.range(height, dtype=tf.float32),
        tf.range(width, dtype=tf.float32),
        indexing="ij",
    )

    # Loop condition and body
    def condition(i, heatmaps):
        return i < num_objects

    def body(i, heatmaps):
        bbox = bboxes_tensor[i]
        label = tf.cast(labels_tensor[i], tf.int32)

        # Denormalize coordinates for the original box
        y1 = bbox[0] * tf.cast(height, tf.float32)
        x1 = bbox[1] * tf.cast(width, tf.float32)
        y2 = bbox[2] * tf.cast(height, tf.float32)
        x2 = bbox[3] * tf.cast(width, tf.float32)

        center_y = (y1 + y2) / 2.0
        center_x = (x1 + x2) / 2.0
        box_height = tf.maximum(y2 - y1, 1.0)
        box_width = tf.maximum(x2 - x1, 1.0)

        # Adjust sigma calculation if needed. Dividing by a larger number
        # will make the Gaussian narrower within the box.
        sigma_y = box_height / 4.0
        sigma_x = box_width / 4.0

        # Calculate gaussian values over the entire grid
        squared_dist_y = tf.square(y_grid - center_y)
        squared_dist_x = tf.square(x_grid - center_x)
        # Add small epsilon to avoid division by zero in edge cases (e.g., 1-pixel box)
        epsilon = 1e-6
        gaussian = tf.exp(
            -0.5
            * (
                squared_dist_y / (tf.square(sigma_y) + epsilon)
                + squared_dist_x / (tf.square(sigma_x) + epsilon)
            )
        )

        # Create a mask strictly within the original bounding box
        mask = tf.logical_and(
            tf.logical_and(y_grid >= y1, y_grid <= y2),
            tf.logical_and(x_grid >= x1, x_grid <= x2),
        )
        masked_gaussian = tf.where(mask, gaussian, tf.zeros_like(gaussian))

        class_heatmap = heatmaps[:, :, label]
        updated_class_heatmap = tf.maximum(class_heatmap, masked_gaussian)

        # Updated heatmaps tensor with the new class channel
        indices = tf.stack(
            [
                tf.range(height)[:, tf.newaxis] * tf.ones_like(x_grid, dtype=tf.int32),
                tf.ones_like(y_grid, dtype=tf.int32) * tf.range(width)[tf.newaxis, :],
                tf.ones((height, width), dtype=tf.int32) * label,
            ],
            axis=-1,
        )
        updated_heatmaps = tf.tensor_scatter_nd_update(
            heatmaps, indices, updated_class_heatmap
        )

        return i + 1, updated_heatmaps

    _, final_heatmaps = tf.while_loop(
        condition,
        body,
        loop_vars=[tf.constant(0), initial_heatmaps],
        shape_invariants=[
            tf.TensorShape([]),  # Shape of loop counter 'i'
            tf.TensorShape([None, None, num_classes]),  # Shape of heatmaps
        ],
    )

    return final_heatmaps

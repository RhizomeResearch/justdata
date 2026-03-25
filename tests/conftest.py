import numpy as np
import pytest
import tensorflow as tf


@pytest.fixture
def seed():
    return tf.constant([42, 0], dtype=tf.int32)


@pytest.fixture
def rng():
    return tf.random.Generator.from_seed(42)


@pytest.fixture
def rgb_image_uint8():
    """32x32x3 uint8 image with random pixel values."""
    np.random.seed(0)
    return tf.constant(np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8))


@pytest.fixture
def rgb_image_224_uint8():
    """224x224x3 uint8 image."""
    np.random.seed(0)
    return tf.constant(np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8))


@pytest.fixture
def rgb_batch_uint8():
    """Batch of 4 32x32x3 uint8 images."""
    np.random.seed(0)
    return tf.constant(np.random.randint(0, 256, (4, 32, 32, 3), dtype=np.uint8))


@pytest.fixture
def grayscale_image_uint8():
    """32x32x1 uint8 grayscale image."""
    np.random.seed(0)
    return tf.constant(np.random.randint(0, 256, (32, 32, 1), dtype=np.uint8))


@pytest.fixture
def classification_sample():
    """A sample dict mimicking a classification dataset element."""
    np.random.seed(0)
    return {
        "image": tf.constant(np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)),
        "label": tf.constant(3, dtype=tf.int64),
    }


@pytest.fixture
def segmentation_sample():
    """A sample dict mimicking a segmentation dataset element."""
    np.random.seed(0)
    return {
        "image": tf.constant(np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)),
        "mask": tf.constant(np.random.randint(0, 5, (32, 32), dtype=np.int32)),
        "label": tf.constant(0, dtype=tf.int64),
    }


@pytest.fixture
def synthetic_classification_ds():
    """A tf.data.Dataset of 20 classification samples (32x32 RGB, 10 classes)."""
    np.random.seed(0)
    images = np.random.randint(0, 256, (20, 32, 32, 3), dtype=np.uint8)
    labels = np.random.randint(0, 10, (20,), dtype=np.int64)

    ds = tf.data.Dataset.from_tensor_slices({"image": images, "label": labels})
    ds = ds.apply(tf.data.experimental.assert_cardinality(20))
    return ds


@pytest.fixture
def synthetic_segmentation_ds():
    """A tf.data.Dataset of 10 segmentation samples (32x32 RGB + mask)."""
    np.random.seed(0)
    images = np.random.randint(0, 256, (10, 32, 32, 3), dtype=np.uint8)
    masks = np.random.randint(0, 5, (10, 32, 32), dtype=np.int32)
    labels = np.zeros((10,), dtype=np.int64)

    ds = tf.data.Dataset.from_tensor_slices(
        {"image": images, "mask": masks, "label": labels}
    )
    ds = ds.apply(tf.data.experimental.assert_cardinality(10))
    return ds

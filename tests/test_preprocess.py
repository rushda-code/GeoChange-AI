"""Pytest checks for GeoChange AI preprocessing and tf.data pipeline."""

from pathlib import Path

import numpy as np
import tensorflow as tf

from src.data import (
    EXPECTED_COUNTS,
    augment_triplet,
    decode_pair,
    get_encoded_sample,
    get_split_indices,
    load_data,
    make_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def setup_module():
    """Load encoded Parquet bytes once for this test module."""
    load_data(PROJECT_ROOT / "configs" / "default.yaml")


def test_expected_split_counts():
    assert len(get_split_indices("train")) == EXPECTED_COUNTS["train"] == 7095
    assert len(get_split_indices("val")) == EXPECTED_COUNTS["val"] == 1024
    assert len(get_split_indices("test")) == EXPECTED_COUNTS["test"] == 2048


def test_no_dropped_manifest_row_in_training_indices():
    import csv

    manifest_path = PROJECT_ROOT / "data" / "processed" / "train_manifest.csv"

    with open(manifest_path, "r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    dropped_indices = {
        int(row["row_index"])
        for row in rows
        if row["keep"].strip().lower() not in ("true", "1", "yes")
    }

    train_indices = set(get_split_indices("train"))

    assert len(train_indices) == 7095
    assert train_indices.isdisjoint(dropped_indices)


def test_image_batch_dtype_shape_and_range():
    dataset = make_dataset("train", batch_size=16, training=True, seed=42)
    (image_a, image_b), _mask = next(iter(dataset))

    for image in (image_a, image_b):
        assert image.dtype == tf.float32
        assert tuple(image.shape) == (16, 256, 256, 3)
        assert float(tf.reduce_min(image).numpy()) >= -1.00001
        assert float(tf.reduce_max(image).numpy()) <= 1.00001


def test_mask_batch_dtype_shape_and_binary_values():
    dataset = make_dataset("train", batch_size=16, training=True, seed=42)
    (_image_a, _image_b), mask = next(iter(dataset))

    assert mask.dtype == tf.float32
    assert tuple(mask.shape) == (16, 256, 256, 1)

    values = set(np.unique(mask.numpy()).tolist())
    assert values.issubset({0.0, 1.0})
    assert float(tf.reduce_min(mask).numpy()) == 0.0
    assert float(tf.reduce_max(mask).numpy()) == 1.0


def test_mask_was_not_mobilenet_preprocessed():
    bytes_a, bytes_b, bytes_mask = get_encoded_sample("train", 0)

    _image_a, _image_b, mask = decode_pair(
        bytes_a, bytes_b, bytes_mask
    )

    assert mask.dtype == tf.float32
    assert float(tf.reduce_min(mask).numpy()) == 0.0
    assert float(tf.reduce_max(mask).numpy()) == 1.0
    assert set(np.unique(mask.numpy()).tolist()).issubset({0.0, 1.0})


def test_decoding_same_bytes_twice_is_identical():
    sample = get_encoded_sample("val", 0)

    first = decode_pair(*sample)
    second = decode_pair(*sample)

    for tensor_a, tensor_b in zip(first, second):
        np.testing.assert_array_equal(tensor_a.numpy(), tensor_b.numpy())


def test_augmentation_moves_images_and_mask_together():
    # A's first channel encodes the same binary spatial pattern as the mask.
    # The transform must preserve this relationship exactly.
    pattern = tf.cast(
        tf.random.uniform((256, 256, 1), seed=123) > 0.5,
        tf.float32,
    )
    image_a = tf.concat(
        [pattern * 255.0, tf.zeros_like(pattern), tf.zeros_like(pattern)],
        axis=-1,
    )
    image_b = tf.zeros((256, 256, 3), dtype=tf.float32)
    mask = pattern

    aug_a, aug_b, aug_mask = augment_triplet(image_a, image_b, mask)

    np.testing.assert_array_equal(
        (aug_a[..., 0:1] / 255.0).numpy(),
        aug_mask.numpy(),
    )
    assert tuple(aug_b.shape) == (256, 256, 3)
    assert tuple(aug_mask.shape) == (256, 256, 1)


def _assert_eval_dataset_repeatable(split):
    dataset = make_dataset(split, batch_size=16, training=False, seed=42)

    first_pass = next(iter(dataset))
    second_pass = next(iter(dataset))

    for first_tensor, second_tensor in zip(
        (first_pass[0][0], first_pass[0][1], first_pass[1]),
        (second_pass[0][0], second_pass[0][1], second_pass[1]),
    ):
        np.testing.assert_array_equal(
            first_tensor.numpy(),
            second_tensor.numpy(),
        )


def test_validation_pipeline_is_deterministic():
    _assert_eval_dataset_repeatable("val")


def test_test_pipeline_is_deterministic():
    _assert_eval_dataset_repeatable("test")

"""
GeoChange AI — STEP 2B: TensorFlow input pipeline.

Original Parquet files are read but never modified.
Only encoded JPEG/PNG byte strings are retained in Python memory.
"""

from __future__ import annotations

import csv
import glob
import time
from pathlib import Path

import pyarrow.parquet as pq
import tensorflow as tf
import yaml


# Expected counts from the completed manifest and dataset inspection.
EXPECTED_COUNTS = {"train": 7095, "val": 1024, "test": 2048}

# Encoded bytes only: each row is (imageA_bytes, imageB_bytes, mask_bytes).
_ENCODED_SPLITS = {}
_ROW_INDICES = {}
_CONFIG = None


def _project_root():
    """Project root is the parent directory of src/."""
    return Path(__file__).resolve().parents[1]


def _resolve_one_parquet(pattern, project_root):
    """Resolve a configured glob and require exactly one matching Parquet."""
    full_pattern = str(project_root / pattern)
    matches = sorted(glob.glob(full_pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one Parquet for pattern {full_pattern}; "
            f"found {len(matches)}: {matches}"
        )
    return Path(matches[0])


def _read_keep_indices(manifest_path):
    """Read row_index values whose manifest keep field is true."""
    keep_indices = set()
    manifest_rows = 0

    with open(manifest_path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"row_index", "keep"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"Manifest must contain columns {sorted(required)}"
            )

        for row in reader:
            manifest_rows += 1
            keep_value = row["keep"].strip().lower()
            if keep_value in ("true", "1", "yes"):
                keep_indices.add(int(row["row_index"]))

    if manifest_rows != 7120:
        raise AssertionError(
            f"Expected 7120 manifest rows, found {manifest_rows}"
        )
    if len(keep_indices) != EXPECTED_COUNTS["train"]:
        raise AssertionError(
            f"Expected 7095 kept train rows, found {len(keep_indices)}"
        )

    return keep_indices


def _load_parquet_encoded(parquet_path, split, keep_indices=None):
    """
    Read Parquet in record batches. Retain encoded bytes, never decoded arrays.
    Returned row_indices refer to original Parquet row positions.
    """
    parquet = pq.ParquetFile(parquet_path)
    required_columns = ["imageA", "imageB", "label"]

    encoded_rows = []
    original_indices = []
    row_index = 0

    for batch in parquet.iter_batches(
        batch_size=128,
        columns=required_columns,
    ):
        for row in batch.to_pylist():
            current_index = row_index
            row_index += 1

            # Train rows excluded by the manifest are never added to memory.
            if keep_indices is not None and current_index not in keep_indices:
                continue

            a = row["imageA"]["bytes"]
            b = row["imageB"]["bytes"]
            mask = row["label"]["bytes"]

            if a is None or b is None or mask is None:
                raise ValueError(
                    f"Null encoded bytes in {split} row {current_index}"
                )

            encoded_rows.append((a, b, mask))
            original_indices.append(current_index)

    return encoded_rows, original_indices, row_index


def load_data(config_path="configs/default.yaml", splits=("train", "val", "test")):
    """
    Load encoded bytes for all splits and validate expected sample counts.

    Call once before creating datasets. Recalling this function reloads the
    encoded byte lists from disk.
    """
    global _ENCODED_SPLITS, _ROW_INDICES, _CONFIG

    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = _project_root() / config_path

    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    project_root = _project_root()
    paths = config["paths"]

    manifest_path = Path(paths["train_manifest"])
    if not manifest_path.is_absolute():
        manifest_path = project_root / manifest_path

    train_keep_indices = _read_keep_indices(manifest_path)

    loaded = {}
    row_indices = {}

    requested_splits = tuple(splits)
    if not requested_splits or any(x not in ("train", "val", "test") for x in requested_splits):
        raise ValueError("splits must contain one or more of train, val, test")

    for split in requested_splits:
        parquet_path = _resolve_one_parquet(
            paths[f"{split}_parquet"], project_root
        )

        keep = train_keep_indices if split == "train" else None

        rows, indices, parquet_row_count = _load_parquet_encoded(
            parquet_path, split, keep_indices=keep
        )

        expected = EXPECTED_COUNTS[split]
        if len(rows) != expected:
            raise AssertionError(
                f"{split}: expected {expected} retained rows, got {len(rows)}"
            )

        if split == "train" and parquet_row_count != 7120:
            raise AssertionError(
                f"Expected 7120 original train Parquet rows, "
                f"found {parquet_row_count}"
            )

        loaded[split] = rows
        row_indices[split] = indices

        print(
            f"{split}: loaded {len(rows)} encoded samples "
            f"(original Parquet rows: {parquet_row_count})"
        )

    # Merge requested splits without loading any unrequested split.
    _ENCODED_SPLITS.update(loaded)
    _ROW_INDICES.update(row_indices)
    _CONFIG = config

    return {
        "counts": {split: len(loaded[split]) for split in loaded},
        "row_indices": {split: list(row_indices[split]) for split in row_indices},
    }


def get_split_indices(split):
    """Return original Parquet row indices retained for the requested split."""
    _require_loaded(split)
    if split not in _ROW_INDICES:
        raise ValueError(f"Unknown split: {split}")
    return list(_ROW_INDICES[split])


def get_encoded_sample(split, position=0):
    """Return one sample's encoded bytes (A, B, mask) for tests/debugging."""
    _require_loaded(split)
    if split not in _ENCODED_SPLITS:
        raise ValueError(f"Unknown split: {split}")
    return _ENCODED_SPLITS[split][position]


def _require_loaded(split="train"):
    if split not in _ENCODED_SPLITS:
        # Load only the requested split. In particular, train/val workflows
        # never resolve or read the TEST Parquet unless explicitly requested.
        load_data(splits=(split,))


def decode_pair(bytes_a, bytes_b, bytes_mask, preprocess_images=True):
    """
    Decode one sample using TensorFlow only.

    Images are JPEG. The mask is PNG and any nonzero value means change.
    By default, images are MobileNetV2-preprocessed for app/inference use.

    The training pipeline passes preprocess_images=False so augmentation can
    happen on raw float images first, then preprocesses image channels only.
    """
    image_a = tf.io.decode_jpeg(
        bytes_a, channels=3, dct_method="INTEGER_ACCURATE"
    )
    image_b = tf.io.decode_jpeg(
        bytes_b, channels=3, dct_method="INTEGER_ACCURATE"
    )
    mask = tf.io.decode_png(bytes_mask, channels=1)

    image_a.set_shape((256, 256, 3))
    image_b.set_shape((256, 256, 3))
    mask.set_shape((256, 256, 1))

    image_a = tf.cast(image_a, tf.float32)
    image_b = tf.cast(image_b, tf.float32)
    mask = tf.cast(mask > 0, tf.float32)

    if preprocess_images:
        image_a = tf.keras.applications.mobilenet_v2.preprocess_input(image_a)
        image_b = tf.keras.applications.mobilenet_v2.preprocess_input(image_b)

    image_a.set_shape((256, 256, 3))
    image_b.set_shape((256, 256, 3))
    mask.set_shape((256, 256, 1))

    return image_a, image_b, mask


def augment_triplet(image_a, image_b, mask):
    """
    Apply the same random geometry to A, B and mask.

    Inputs are raw float images (0..255) and a binary float mask (0..1).
    Stacking all seven channels guarantees that the same transform is used.
    """
    joined = tf.concat([image_a, image_b, mask], axis=-1)

    joined = tf.image.random_flip_left_right(joined)
    joined = tf.image.random_flip_up_down(joined)

    k = tf.random.uniform(
        shape=[],
        minval=0,
        maxval=4,
        dtype=tf.int32,
    )
    joined = tf.image.rot90(joined, k=k)

    image_a, image_b, mask = tf.split(joined, [3, 3, 1], axis=-1)

    image_a.set_shape((256, 256, 3))
    image_b.set_shape((256, 256, 3))
    mask.set_shape((256, 256, 1))

    return image_a, image_b, mask


def _preprocess_images(image_a, image_b):
    """Apply MobileNetV2 preprocessing to images, never to the mask."""
    image_a = tf.keras.applications.mobilenet_v2.preprocess_input(image_a)
    image_b = tf.keras.applications.mobilenet_v2.preprocess_input(image_b)
    return image_a, image_b


def make_dataset(split, batch_size=16, training=False, seed=42):
    """
    Build ((imageA, imageB), mask) batches.

    Training: shuffle encoded bytes, decode, augment, preprocess, batch,
    drop incomplete final batch, prefetch.
    Val/test: preserve row order, no augmentation, keep final partial batch.
    """
    if split not in ("train", "val", "test"):
        raise ValueError("split must be 'train', 'val' or 'test'")
    _require_loaded(split)
    if training and split != "train":
        raise ValueError("training=True is only valid for the train split")
    if not training and split == "train":
        raise ValueError("Use training=True for the train split")

    encoded_rows = _ENCODED_SPLITS[split]

    def generator():
        # The generator yields encoded strings, not decoded image arrays.
        for bytes_a, bytes_b, bytes_mask in encoded_rows:
            yield bytes_a, bytes_b, bytes_mask

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.string),
        ),
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(encoded_rows),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    def decode_and_transform(bytes_a, bytes_b, bytes_mask):
        image_a, image_b, mask = decode_pair(
            bytes_a,
            bytes_b,
            bytes_mask,
            preprocess_images=False,
        )

        if training:
            image_a, image_b, mask = augment_triplet(
                image_a, image_b, mask
            )

        image_a, image_b = _preprocess_images(image_a, image_b)

        return (image_a, image_b), mask

    dataset = dataset.map(
        decode_and_transform,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=not training,
    )

    dataset = dataset.batch(
        batch_size,
        drop_remainder=training,
    )

    # Deliberately no cache(): decoded training tensors would use substantial RAM.
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    options = tf.data.Options()
    options.experimental_deterministic = not training
    dataset = dataset.with_options(options)

    return dataset


def describe_batch(dataset):
    """Print shapes, dtypes, ranges and changed-pixel share for one batch."""
    (image_a, image_b), mask = next(iter(dataset))

    for name, tensor in (
        ("imageA", image_a),
        ("imageB", image_b),
        ("mask", mask),
    ):
        print(
            f"{name}: shape={tuple(tensor.shape)}, "
            f"dtype={tensor.dtype.name}, "
            f"min={float(tf.reduce_min(tensor).numpy()):.6f}, "
            f"max={float(tf.reduce_max(tensor).numpy()):.6f}"
        )

    changed_share = tf.reduce_mean(mask)
    print(f"Changed-pixel share in batch: {float(changed_share.numpy()):.6%}")


def time_batches(dataset, batches=50):
    """
    Time up to `batches` batches and report batches/s and images/s.
    Uses actual batch sizes, including any partial evaluation batch.
    """
    iterator = iter(dataset)
    count = 0
    image_count = 0

    start = time.perf_counter()

    while count < batches:
        try:
            (image_a, _), _mask = next(iterator)
        except StopIteration:
            break

        count += 1
        image_count += int(tf.shape(image_a)[0].numpy())

    elapsed = time.perf_counter() - start

    if count == 0 or elapsed <= 0:
        print("Timing unavailable: dataset produced no batches.")
        return None

    batches_per_second = count / elapsed
    images_per_second = image_count / elapsed

    print(
        f"Timing: {count} batches in {elapsed:.3f}s | "
        f"{batches_per_second:.2f} batches/s | "
        f"{images_per_second:.2f} images/s"
    )

    return {
        "batches": count,
        "images": image_count,
        "seconds": elapsed,
        "batches_per_second": batches_per_second,
        "images_per_second": images_per_second,
    }


def make_fixed_batch(split="train", n=16):
    """Return the first N cleaned train rows as one deterministic, unaugmented batch."""
    if split != "train":
        raise ValueError("make_fixed_batch is intended only for train overfit checks")
    _require_loaded("train")
    rows = _ENCODED_SPLITS["train"][:n]
    if len(rows) != n:
        raise ValueError(f"Requested {n} samples, only {len(rows)} available")
    def generator():
        for a, b, mask in rows:
            yield a, b, mask
    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(tf.TensorSpec((), tf.string), tf.TensorSpec((), tf.string), tf.TensorSpec((), tf.string))
    )
    def decode(a, b, mask):
        ia, ib, m = decode_pair(a, b, mask, preprocess_images=True)
        return (ia, ib), m
    return ds.map(decode, num_parallel_calls=tf.data.AUTOTUNE).batch(n).prefetch(1)

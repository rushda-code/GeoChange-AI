# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 14:24:19 2026

@author: Admin
"""

# ============================================================
# GEOCHANGE-AI — STEP 9: SINGLE FINAL TEST EVALUATION
#
# TEST is evaluated once using frozen settings.
# No training, tuning, threshold selection or model modification.
# ============================================================

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import tensorflow as tf
import keras


# ------------------------------------------------------------
# 1. PATHS AND FROZEN SETTINGS
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULT_DIR = ROOT / "experiments" / "results"
STEP8_DIR = RESULT_DIR / "step8"
CACHE_DIR = ROOT / "data" / "processed" / "cache"

DONE_PATH = RESULT_DIR / "TEST_EVAL_DONE.json"
SETTINGS_PATH = RESULT_DIR / "final_settings.json"

TEST_PARQUET = (
    ROOT / "data" / "raw" / "data" /
    "test-00000-of-00001-31d7c3e3444e5b5d.parquet"
)

DICE_MODEL_PATH = (
    ROOT / "experiments" / "runs" /
    "bcedice_v1" / "best_model.keras"
)

BCE_MODEL_PATH = (
    ROOT / "experiments" / "runs" /
    "bce_v1" / "best_model.keras"
)

BATCH_SIZE = 16
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 42

EXPECTED_TEST_ROWS = 2048
EXPECTED_VAL_ROWS = 1024
MIN_AREA = 50

GLOBAL_THRESHOLD = 93

BUCKETS = [
    ("0%", 0.0, 0.0),
    (">0 to <5%", 0.0, 5.0),
    ("5 to 20%", 5.0, 20.0),
    (">20%", 20.0, 100.000001),
]


# ------------------------------------------------------------
# 2. SAFETY CHECKS
# ------------------------------------------------------------

def check_inputs():

    required = [
        SETTINGS_PATH,
        TEST_PARQUET,
        DICE_MODEL_PATH,
        BCE_MODEL_PATH,
        CACHE_DIR / "bce_dice_val_probabilities.npy",
        CACHE_DIR / "bce_only_val_probabilities.npy",
        CACHE_DIR / "val_ground_truth.npy",
    ]

    missing = [str(p) for p in required if not p.exists()]

    if missing:
        raise FileNotFoundError(
            "Required files are missing:\n" +
            "\n".join(missing)
        )

    with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
        settings = json.load(f)

    print("\nFROZEN SETTINGS")
    print(json.dumps(settings, indent=2))

    return settings


def sha256_file(path):

    digest = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


# ------------------------------------------------------------
# 3. CONNECTED COMPONENT CLEANING
# ------------------------------------------------------------

def clean_mask_local(probability, threshold, min_area=50):

    binary = (
        np.asarray(probability) > threshold
    ).astype(np.uint8)

    n_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )

    cleaned = np.zeros_like(binary, dtype=np.uint8)
    region_count = 0

    for label_id in range(1, n_labels):

        area = stats[label_id, cv2.CC_STAT_AREA]

        if area >= min_area:

            cleaned[labels == label_id] = 1
            region_count += 1

    return cleaned, int(region_count)


# ------------------------------------------------------------
# 4. METRICS
# ------------------------------------------------------------

def pixel_counts(gt, pred):

    gt = gt.astype(bool)
    pred = pred.astype(bool)

    tp = int(np.sum(gt & pred))
    fp = int(np.sum(~gt & pred))
    fn = int(np.sum(gt & ~pred))
    tn = int(np.sum(~gt & ~pred))

    return tp, fp, fn, tn


def scores_from_counts(tp, fp, fn, tn):

    precision = (
        tp / (tp + fp)
        if tp + fp > 0 else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0 else 0.0
    )

    f1 = (
        2 * tp / (2 * tp + fp + fn)
        if 2 * tp + fp + fn > 0 else 0.0
    )

    iou = (
        tp / (tp + fp + fn)
        if tp + fp + fn > 0 else 0.0
    )

    accuracy = (
        (tp + tn) / (tp + fp + fn + tn)
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "pixel_accuracy": accuracy,
    }


def region_count(mask):

    binary = np.asarray(mask, dtype=np.uint8)

    n_labels, _, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )

    return sum(
        1
        for label_id in range(1, n_labels)
        if stats[label_id, cv2.CC_STAT_AREA] >= MIN_AREA
    )


# ------------------------------------------------------------
# 5. PER-CROP METRIC COLLECTION
# ------------------------------------------------------------

def evaluate_masks(gt_masks, pred_masks):

    rows = []

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    empty_crops = 0
    empty_alarms = 0

    predicted_region_counts = []
    gt_region_counts = []

    for i in range(len(gt_masks)):

        gt = np.asarray(gt_masks[i], dtype=np.uint8)
        pred = np.asarray(pred_masks[i], dtype=np.uint8)

        tp, fp, fn, tn = pixel_counts(gt, pred)

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

        gt_regions = region_count(gt)
        pred_regions = region_count(pred)

        gt_change_pct = (
            np.mean(gt) * 100.0
        )

        if not np.any(gt):

            empty_crops += 1

            if np.any(pred):
                empty_alarms += 1

        predicted_region_counts.append(pred_regions)
        gt_region_counts.append(gt_regions)

        rows.append({
            "index": i,
            "gt_change_pct": gt_change_pct,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "predicted_regions": pred_regions,
            "gt_regions": gt_regions,
        })

    scores = scores_from_counts(
        total_tp,
        total_fp,
        total_fn,
        total_tn
    )

    count_errors = (
        np.abs(
            np.asarray(predicted_region_counts) -
            np.asarray(gt_region_counts)
        )
    )

    scores.update({

        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "TN": total_tn,

        "empty_crops": empty_crops,
        "empty_crop_alarms": empty_alarms,

        "empty_crop_false_alarm_count": empty_alarms,

        "empty_crop_false_alarm_rate": (
            empty_alarms / empty_crops
            if empty_crops else 0.0
        ),

        "mean_predicted_regions": float(
            np.mean(predicted_region_counts)
        ),

        "mean_gt_regions": float(
            np.mean(gt_region_counts)
        ),

        "count_mae": float(
            np.mean(count_errors)
        ),

        "exact_count_pct": float(
            np.mean(count_errors == 0) * 100.0
        ),

        "_crop_tp": np.asarray(
            [r["TP"] for r in rows],
            dtype=np.int64
        ),

        "_crop_fp": np.asarray(
            [r["FP"] for r in rows],
            dtype=np.int64
        ),

        "_crop_fn": np.asarray(
            [r["FN"] for r in rows],
            dtype=np.int64
        ),

    })

    return scores, pd.DataFrame(rows)


# ------------------------------------------------------------
# 6. CHANGE-PROPORTION BUCKETS
# ------------------------------------------------------------

def bucket_results(gt_masks, pred_masks):

    rows = []

    change_percentages = (
        np.mean(gt_masks, axis=(1, 2)) * 100.0
    )

    for bucket_name, lower, upper in BUCKETS:

        if bucket_name == "0%":

            selected = change_percentages == 0.0

        elif bucket_name == ">0 to <5%":

            selected = (
                (change_percentages > 0.0) &
                (change_percentages < 5.0)
            )

        elif bucket_name == "5 to 20%":

            selected = (
                (change_percentages >= 5.0) &
                (change_percentages <= 20.0)
            )

        else:

            selected = change_percentages > 20.0

        indices = np.where(selected)[0]

        tp = fp = fn = tn = 0
        gt_regions = []

        empty_alarms = 0

        for i in indices:

            gt = gt_masks[i]
            pred = pred_masks[i]

            a, b, c, d = pixel_counts(gt, pred)

            tp += a
            fp += b
            fn += c
            tn += d

            gt_regions.append(region_count(gt))

            if not np.any(gt) and np.any(pred):
                empty_alarms += 1

        scores = scores_from_counts(tp, fp, fn, tn)

        rows.append({

            "gt_change_bucket": bucket_name,
            "number_of_crops": len(indices),

            "precision": scores["precision"],
            "recall": scores["recall"],
            "f1": scores["f1"],
            "iou": scores["iou"],

            "mean_gt_regions": (
                float(np.mean(gt_regions))
                if gt_regions else 0.0
            ),

            "empty_crop_false_alarms": empty_alarms,

            "empty_crop_false_alarm_rate": (
                empty_alarms / len(indices)
                if bucket_name == "0%" and len(indices)
                else np.nan
            ),

        })

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# 7. TEST DATASET — TEST ONLY
# ------------------------------------------------------------

def build_test_dataset():

    parquet = pq.ParquetFile(TEST_PARQUET)

    if parquet.metadata.num_rows != EXPECTED_TEST_ROWS:

        raise AssertionError(
            f"Expected {EXPECTED_TEST_ROWS} test rows, "
            f"found {parquet.metadata.num_rows}"
        )

    def generator():

        row_index = 0

        for batch in parquet.iter_batches(
            batch_size=128,
            columns=["imageA", "imageB", "label"]
        ):

            for row in batch.to_pylist():

                a = row["imageA"]["bytes"]
                b = row["imageB"]["bytes"]
                mask = row["label"]["bytes"]

                if a is None or b is None or mask is None:
                    raise ValueError(
                        f"Null encoded bytes in TEST row {row_index}"
                    )

                row_index += 1

                yield a, b, mask

        if row_index != EXPECTED_TEST_ROWS:
            raise AssertionError(
                f"Read {row_index} rows, expected 2048"
            )

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec((), tf.string),
            tf.TensorSpec((), tf.string),
            tf.TensorSpec((), tf.string),
        )
    )

    def decode(a, b, mask):

        image_a = tf.io.decode_jpeg(
            a, channels=3,
            dct_method="INTEGER_ACCURATE"
        )

        image_b = tf.io.decode_jpeg(
            b, channels=3,
            dct_method="INTEGER_ACCURATE"
        )

        gt = tf.io.decode_png(mask, channels=1)

        image_a = tf.cast(image_a, tf.float32)
        image_b = tf.cast(image_b, tf.float32)

        gt = tf.cast(gt > 0, tf.uint8)

        image_a = (
            tf.keras.applications.mobilenet_v2
            .preprocess_input(image_a)
        )

        image_b = (
            tf.keras.applications.mobilenet_v2
            .preprocess_input(image_b)
        )

        return (image_a, image_b), gt

    dataset = dataset.map(
        decode,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True
    )

    dataset = dataset.batch(
        BATCH_SIZE,
        drop_remainder=False
    )

    return dataset.prefetch(tf.data.AUTOTUNE)


# ------------------------------------------------------------
# 8. MODEL PREDICTION AND CACHING
# ------------------------------------------------------------

def predict_and_cache(
    model,
    dataset,
    output_path,
    gt_path=None
):

    probabilities = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float16,
        shape=(EXPECTED_TEST_ROWS, 256, 256)
    )

    ground_truth = None

    if gt_path is not None:

        ground_truth = np.lib.format.open_memmap(
            gt_path,
            mode="w+",
            dtype=np.uint8,
            shape=(EXPECTED_TEST_ROWS, 256, 256)
        )

    offset = 0

    for (image_a, image_b), batch_gt in dataset:

        prediction = model(
            [image_a, image_b],
            training=False
        )

        prediction = prediction.numpy()[..., 0]

        batch_size = prediction.shape[0]

        probabilities[
            offset:offset + batch_size
        ] = prediction.astype(np.float16)

        if ground_truth is not None:

            ground_truth[
                offset:offset + batch_size
            ] = batch_gt.numpy()[..., 0]

        offset += batch_size

        if offset % 256 == 0 or offset == EXPECTED_TEST_ROWS:

            print(
                f"Predicted {offset}/{EXPECTED_TEST_ROWS}"
            )

    if offset != EXPECTED_TEST_ROWS:

        raise AssertionError(
            f"Prediction count mismatch: {offset}"
        )

    probabilities.flush()

    if ground_truth is not None:
        ground_truth.flush()

    del probabilities

    if ground_truth is not None:
        del ground_truth


# ------------------------------------------------------------
# 9. BASELINE IMAGE DIFFERENCE
# ------------------------------------------------------------

def undo_mobilenet_preprocessing(image):

    image = np.asarray(image, dtype=np.float32)

    image = (image + 1.0) * 127.5

    return np.clip(image, 0, 255).astype(np.uint8)


def difference_image(image_a, image_b):

    image_a = undo_mobilenet_preprocessing(image_a)
    image_b = undo_mobilenet_preprocessing(image_b)

    gray_a = cv2.cvtColor(
        image_a,
        cv2.COLOR_RGB2GRAY
    )

    gray_b = cv2.cvtColor(
        image_b,
        cv2.COLOR_RGB2GRAY
    )

    difference = cv2.absdiff(gray_a, gray_b)

    return cv2.GaussianBlur(
        difference,
        (3, 3),
        0
    )


def make_baseline_prediction(
    image_a,
    image_b,
    method
):

    if method == "zero":

        binary = np.zeros(
            (256, 256),
            dtype=np.uint8
        )

    else:

        difference = difference_image(
            image_a,
            image_b
        )

        if method == "otsu":

            _, binary = cv2.threshold(
                difference,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

        elif method == "global":

            _, binary = cv2.threshold(
                difference,
                GLOBAL_THRESHOLD,
                255,
                cv2.THRESH_BINARY
            )

        else:

            raise ValueError(method)

    cleaned, regions = clean_mask_local(
        binary,
        threshold=0.5,
        min_area=MIN_AREA
    )

    return cleaned


# ------------------------------------------------------------
# 10. BASELINE EVALUATION
# ------------------------------------------------------------

def evaluate_baseline(method):

    dataset = build_test_dataset()

    gt_list = []
    pred_list = []

    for (image_a, image_b), batch_gt in dataset:

        image_a = image_a.numpy()
        image_b = image_b.numpy()
        batch_gt = batch_gt.numpy()[..., 0]

        for i in range(len(batch_gt)):

            prediction = make_baseline_prediction(
                image_a[i],
                image_b[i],
                method
            )

            gt_list.append(batch_gt[i])
            pred_list.append(prediction)

    gt_array = np.asarray(gt_list, dtype=np.uint8)
    pred_array = np.asarray(pred_list, dtype=np.uint8)

    scores, _ = evaluate_masks(
        gt_array,
        pred_array
    )

    return scores


# ------------------------------------------------------------
# 11. MODEL MASK GENERATION
# ------------------------------------------------------------

def probabilities_to_masks(probabilities, threshold):

    masks = np.zeros(
        probabilities.shape,
        dtype=np.uint8
    )

    for i in range(len(probabilities)):

        masks[i], _ = clean_mask_local(
            probabilities[i],
            threshold=threshold,
            min_area=MIN_AREA
        )

    return masks


# ------------------------------------------------------------
# 12. PAIRED BOOTSTRAP
# ------------------------------------------------------------

def pooled_f1_from_counts(tp, fp, fn):

    denominator = 2 * tp + fp + fn

    return np.divide(
        2 * tp,
        denominator,
        out=np.zeros_like(
            np.asarray(denominator, dtype=np.float64)
        ),
        where=denominator > 0
    )


def paired_bootstrap(dice_scores, bce_scores):

    rng = np.random.default_rng(BOOTSTRAP_SEED)

    n = len(dice_scores["_crop_tp"])

    dice_tp = dice_scores["_crop_tp"]
    dice_fp = dice_scores["_crop_fp"]
    dice_fn = dice_scores["_crop_fn"]

    bce_tp = bce_scores["_crop_tp"]
    bce_fp = bce_scores["_crop_fp"]
    bce_fn = bce_scores["_crop_fn"]

    dice_boot = []
    bce_boot = []
    difference_boot = []

    for _ in range(BOOTSTRAP_REPLICATES):

        indices = rng.integers(
            0, n, size=n
        )

        d_f1 = pooled_f1_from_counts(
            np.sum(dice_tp[indices]),
            np.sum(dice_fp[indices]),
            np.sum(dice_fn[indices])
        )

        b_f1 = pooled_f1_from_counts(
            np.sum(bce_tp[indices]),
            np.sum(bce_fp[indices]),
            np.sum(bce_fn[indices])
        )

        dice_boot.append(float(d_f1))
        bce_boot.append(float(b_f1))
        difference_boot.append(float(d_f1 - b_f1))

    def interval(values):

        return [
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5))
        ]

    return {

        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,

        "bce_dice_f1_95_ci": interval(dice_boot),

        "bce_only_f1_95_ci": interval(bce_boot),

        "f1_difference_dice_minus_bce_95_ci": (
            interval(difference_boot)
        ),

        "mean_bootstrap_f1_difference": float(
            np.mean(difference_boot)
        ),

    }


# ------------------------------------------------------------
# 13. TABLE FORMATTING
# ------------------------------------------------------------

METRIC_KEYS = [
    "precision",
    "recall",
    "f1",
    "iou",
    "pixel_accuracy",
    "empty_crop_false_alarm_rate",
    "mean_predicted_regions",
    "mean_gt_regions",
    "count_mae",
    "exact_count_pct",
]


def markdown_comparison_table(results):

    rows = [
        "| Metric | BCE+Dice VAL | BCE+Dice TEST | "
        "BCE-only VAL | BCE-only TEST |",
        "|---|---:|---:|---:|---:|"
    ]

    labels = {
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "iou": "IoU",
        "pixel_accuracy": "Pixel accuracy",
        "empty_crop_false_alarm_rate": "Empty-crop FAR",
        "mean_predicted_regions": "Mean predicted regions",
        "mean_gt_regions": "Mean GT regions",
        "count_mae": "Count MAE",
        "exact_count_pct": "Exact count (%)",
    }

    for key in METRIC_KEYS:

        values = [
            results["dice_val"][key],
            results["dice_test"][key],
            results["bce_val"][key],
            results["bce_test"][key],
        ]

        if key == "exact_count_pct":

            formatted = [f"{v:.2f}%" for v in values]

        elif key == "empty_crop_false_alarm_rate":

            formatted = [f"{v * 100:.2f}%" for v in values]

        else:

            formatted = [f"{v:.4f}" for v in values]

        rows.append(
            f"| {labels[key]} | " +
            " | ".join(formatted) +
            " |"
        )

    return "\n".join(rows)


# ------------------------------------------------------------
# 14. MAIN
# ------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--force",
        action="store_true"
    )

    args = parser.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if DONE_PATH.exists():

        if not args.force:

            raise RuntimeError(
                "TEST_EVAL_DONE.json already exists. "
                "Refusing to evaluate TEST again. "
                "Use --force only if you explicitly accept "
                "that the evaluation is no longer a single "
                "clean evaluation."
            )

        print(
            "\nWARNING: --force supplied. "
            "These numbers will no longer represent "
            "a single clean evaluation.\n"
        )

    settings = check_inputs()

    print("\n" + "=" * 65)
    print("GEOCHANGE-AI — FINAL TEST EVALUATION")
    print("=" * 65)

    print("TEST rows:", EXPECTED_TEST_ROWS)
    print("Batch size:", BATCH_SIZE)
    print("Thresholds and min_area are frozen.")
    print("No augmentation.")
    print("No manifest filtering.")

    # --------------------------------------------------------
    # Frozen settings
    # --------------------------------------------------------

    dice_threshold = 0.30
    bce_threshold = 0.40

    dice_min_area = 50
    bce_min_area = 50

    # Validate that final settings agree with the frozen plan.
    # Supports either nested or flat JSON formats.
    settings_text = json.dumps(settings).lower()

    if "0.3" not in settings_text:
        raise ValueError(
            "final_settings.json does not appear to contain "
            "the frozen BCE+Dice threshold 0.3. "
            "Inspect the file before continuing."
        )

    # --------------------------------------------------------
    # VAL metrics from existing caches
    # --------------------------------------------------------

    print("\nLoading cached VAL predictions.")

    val_gt = np.load(
        CACHE_DIR / "val_ground_truth.npy",
        mmap_mode="r"
    )

    val_dice_prob = np.load(
        CACHE_DIR / "bce_dice_val_probabilities.npy",
        mmap_mode="r"
    )

    val_bce_prob = np.load(
        CACHE_DIR / "bce_only_val_probabilities.npy",
        mmap_mode="r"
    )

    if len(val_gt) != EXPECTED_VAL_ROWS:
        raise AssertionError("Unexpected VAL row count.")

    val_dice_masks = probabilities_to_masks(
        val_dice_prob,
        dice_threshold
    )

    val_bce_masks = probabilities_to_masks(
        val_bce_prob,
        bce_threshold
    )

    dice_val, _ = evaluate_masks(
        val_gt,
        val_dice_masks
    )

    bce_val, _ = evaluate_masks(
        val_gt,
        val_bce_masks
    )

    del val_dice_masks, val_bce_masks

    # --------------------------------------------------------
    # Model loading
    # --------------------------------------------------------

    from src.model import AbsDiff

    custom_objects = {
        "AbsDiff": AbsDiff,
        "GeoChange>AbsDiff": AbsDiff,
    }

    print("\nLoading BCE+Dice model.")

    dice_model = keras.models.load_model(
        DICE_MODEL_PATH,
        compile=False,
        custom_objects=custom_objects
    )

    print("\nLoading BCE-only model.")

    bce_model = keras.models.load_model(
        BCE_MODEL_PATH,
        compile=False,
        custom_objects=custom_objects
    )

    # --------------------------------------------------------
    # TEST prediction caches
    # --------------------------------------------------------

    test_gt_path = CACHE_DIR / "test_ground_truth.npy"

    dice_test_prob_path = (
        CACHE_DIR / "test_bce_dice_probabilities.npy"
    )

    bce_test_prob_path = (
        CACHE_DIR / "test_bce_only_probabilities.npy"
    )

    print("\nCreating TEST dataset.")

    test_ds = build_test_dataset()

    print("\nPredicting with BCE+Dice.")

    predict_and_cache(
        dice_model,
        test_ds,
        dice_test_prob_path,
        gt_path=test_gt_path
    )

    del dice_model

    tf.keras.backend.clear_session()

    print("\nPredicting with BCE-only.")

    bce_model = keras.models.load_model(
        BCE_MODEL_PATH,
        compile=False,
        custom_objects=custom_objects
    )

    test_ds = build_test_dataset()

    predict_and_cache(
        bce_model,
        test_ds,
        bce_test_prob_path
    )

    del bce_model

    tf.keras.backend.clear_session()

    # --------------------------------------------------------
    # Load cached TEST arrays
    # --------------------------------------------------------

    print("\nLoading cached TEST arrays.")

    test_gt = np.load(
        test_gt_path,
        mmap_mode="r"
    )

    dice_test_prob = np.load(
        dice_test_prob_path,
        mmap_mode="r"
    )

    bce_test_prob = np.load(
        bce_test_prob_path,
        mmap_mode="r"
    )

    if len(test_gt) != EXPECTED_TEST_ROWS:
        raise AssertionError("TEST GT count mismatch.")

    # --------------------------------------------------------
    # Model evaluation
    # --------------------------------------------------------

    print("\nPost-processing BCE+Dice TEST predictions.")

    dice_test_masks = probabilities_to_masks(
        dice_test_prob,
        dice_threshold
    )

    dice_test, dice_crop_df = evaluate_masks(
        test_gt,
        dice_test_masks
    )

    print("\nPost-processing BCE-only TEST predictions.")

    bce_test_masks = probabilities_to_masks(
        bce_test_prob,
        bce_threshold
    )

    bce_test, _ = evaluate_masks(
        test_gt,
        bce_test_masks
    )

    # --------------------------------------------------------
    # Per-crop final model CSV
    # --------------------------------------------------------

    crop_csv_path = (
        RESULT_DIR / "test_final_model_per_crop.csv"
    )

    dice_crop_df.to_csv(
        crop_csv_path,
        index=False
    )

    # --------------------------------------------------------
    # Bucket tables
    # --------------------------------------------------------

    dice_buckets = bucket_results(
        test_gt,
        dice_test_masks
    )

    bce_buckets = bucket_results(
        test_gt,
        bce_test_masks
    )

    dice_buckets.to_csv(
        RESULT_DIR / "test_bce_dice_buckets.csv",
        index=False
    )

    bce_buckets.to_csv(
        RESULT_DIR / "test_bce_only_buckets.csv",
        index=False
    )

    # --------------------------------------------------------
    # Baselines
    # --------------------------------------------------------

    print("\nEvaluating all-zero baseline.")

    zero_test = evaluate_baseline("zero")

    print("\nEvaluating per-image Otsu baseline.")

    otsu_test = evaluate_baseline("otsu")

    print("\nEvaluating global threshold 93 baseline.")

    global_test = evaluate_baseline("global")

    baseline_results = pd.DataFrame([
        {"baseline": "All-zero", **zero_test},
        {"baseline": "Per-image Otsu", **otsu_test},
        {"baseline": "Global threshold 93", **global_test},
    ])

    baseline_results.to_csv(
        RESULT_DIR / "test_baseline_results.csv",
        index=False
    )

    # --------------------------------------------------------
    # Paired bootstrap
    # --------------------------------------------------------

    print("\nRunning paired bootstrap.")

    bootstrap = paired_bootstrap(
        dice_test,
        bce_test
    )

    with open(
        RESULT_DIR / "test_bootstrap.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            bootstrap,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("MODEL TEST METRICS")
    print("=" * 65)

    comparison = pd.DataFrame([
        {"model": "BCE+Dice", **dice_test},
        {"model": "BCE-only", **bce_test},
    ])

    print(
        comparison[
            [
                "model",
                "precision",
                "recall",
                "f1",
                "iou",
                "pixel_accuracy",
                "empty_crop_false_alarm_count",
                "empty_crop_false_alarm_rate",
                "mean_predicted_regions",
                "mean_gt_regions",
                "count_mae",
                "exact_count_pct",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 65)
    print("BASELINE TEST METRICS")
    print("=" * 65)

    print(
        baseline_results[
            [
                "baseline",
                "precision",
                "recall",
                "f1",
                "iou",
                "pixel_accuracy",
                "empty_crop_false_alarm_count",
                "empty_crop_false_alarm_rate",
                "mean_predicted_regions",
                "mean_gt_regions",
                "count_mae",
                "exact_count_pct",
            ]
        ].to_string(index=False)
    )

    print("\nBCE+Dice TEST BUCKETS")
    print(dice_buckets.to_string(index=False))

    print("\nBCE-only TEST BUCKETS")
    print(bce_buckets.to_string(index=False))

    print("\nPAIRED BOOTSTRAP — TEST")
    print(json.dumps(bootstrap, indent=2))

    print(
        "\nBootstrap interpretation: "
        "The intervals quantify uncertainty from resampling "
        "the 2,048 test crops with paired indices. They do not "
        "measure variation across independent training seeds."
    )

    # --------------------------------------------------------
    # README-ready report
    # --------------------------------------------------------

    report = []

    report.append("# GeoChange-AI — Final Evaluation")
    report.append("")
    report.append("## Evaluation protocol")
    report.append("")
    report.append(
        "- TEST set: 2,048 rows."
    )
    report.append(
        "- All TEST rows evaluated; no training-manifest filtering."
    )
    report.append(
        "- No augmentation."
    )
    report.append(
        "- Batch size: 16."
    )
    report.append(
        "- Connected-component filtering: minimum area 50 pixels."
    )
    report.append(
        "- BCE+Dice: threshold 0.30."
    )
    report.append(
        "- BCE-only: threshold 0.40."
    )
    report.append(
        "- All pixel metrics calculated after post-processing."
    )
    report.append("")
    report.append("## Model comparison")
    report.append("")
    report.append(markdown_comparison_table({
        "dice_val": dice_val,
        "dice_test": dice_test,
        "bce_val": bce_val,
        "bce_test": bce_test,
    }))
    report.append("")
    report.append("## Baseline comparison — TEST")
    report.append("")
    report.append(
        baseline_results[
            [
                "baseline",
                "precision",
                "recall",
                "f1",
                "iou",
                "pixel_accuracy",
                "empty_crop_false_alarm_rate",
                "mean_predicted_regions",
                "mean_gt_regions",
                "count_mae",
                "exact_count_pct",
            ]
        ].to_markdown(index=False)
    )
    report.append("")
    report.append("## BCE+Dice — TEST change buckets")
    report.append("")
    report.append(dice_buckets.to_markdown(index=False))
    report.append("")
    report.append("## BCE-only — TEST change buckets")
    report.append("")
    report.append(bce_buckets.to_markdown(index=False))
    report.append("")
    report.append("## Paired bootstrap — TEST")
    report.append("")
    report.append("```json")
    report.append(json.dumps(bootstrap, indent=2))
    report.append("```")
    report.append("")
    report.append(
        "The paired bootstrap measures crop-resampling uncertainty "
        "on this test set. It does not measure variation between "
        "independent training seeds."
    )
    report.append("")
    report.append("## Final model")
    report.append("")
    report.append(
        "The declared application model remains BCE+Dice, "
        "threshold 0.30, minimum component area 50 pixels."
    )

    report_path = RESULT_DIR / "test_results.md"

    report_path.write_text(
        "\n".join(report),
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Completion sentinel — only after successful completion
    # --------------------------------------------------------

    completion = {

        "completed_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "test_rows": EXPECTED_TEST_ROWS,

        "batch_size": BATCH_SIZE,

        "models": {

            "bce_dice": {
                "path": str(DICE_MODEL_PATH.relative_to(ROOT)),
                "sha256": sha256_file(DICE_MODEL_PATH),
                "threshold": dice_threshold,
                "min_area": dice_min_area,
            },

            "bce_only": {
                "path": str(BCE_MODEL_PATH.relative_to(ROOT)),
                "sha256": sha256_file(BCE_MODEL_PATH),
                "threshold": bce_threshold,
                "min_area": bce_min_area,
            },

        },

        "tensorflow_version": tf.__version__,
        "keras_version": keras.__version__,

        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,

        "test_results_file": str(
            report_path.relative_to(ROOT)
        ),

    }

    with open(
        DONE_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            completion,
            f,
            indent=2
        )

    print("\n" + "=" * 65)
    print("SAVED OUTPUTS")
    print("=" * 65)

    print("Test predictions:")
    print(dice_test_prob_path)
    print(bce_test_prob_path)

    print("\nPer-crop CSV:")
    print(crop_csv_path)

    print("\nREADME report:")
    print(report_path)

    print("\nBootstrap:")
    print(RESULT_DIR / "test_bootstrap.json")

    print("\nCompletion sentinel:")
    print(DONE_PATH)

    print("\n" + "=" * 65)
    print(
        "TEST HAS NOW BEEN EVALUATED. "
        "DO NOT TUNE ON THESE NUMBERS."
    )
    print("=" * 65)


if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 13:51:31 2026

@author: Admin
"""

"""
GeoChange-AI — STEP 8: VALIDATION ANALYSIS

PART A: Matched region-count evaluation.
PART B: Misalignment with unshifted and shifted labels.
PART C: Ground-truth change buckets and visual error analysis.

STRICT DATA RULE:
- VAL split only.
- Never loads or accesses TEST.
- Uses cached validation probabilities for Parts A and C.
- Part B runs BCE+Dice inference on shifted validation image pairs.
"""

from pathlib import Path
import glob
import json
import random

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import tensorflow as tf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# 1. PATHS AND FIXED SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

CACHE_DIR = ROOT / "data" / "processed" / "cache"
RESULT_DIR = ROOT / "experiments" / "results" / "step8"

VAL_PATTERN = "data/raw/data/val-*.parquet"

PROB_DICE_PATH = CACHE_DIR / "bce_dice_val_probabilities.npy"
PROB_BCE_PATH = CACHE_DIR / "bce_only_val_probabilities.npy"
GT_CACHE_PATH = CACHE_DIR / "val_ground_truth.npy"

DICE_MODEL_PATH = (
    ROOT / "experiments" / "runs" / "bcedice_v1" / "best_model.keras"
)

# Final settings: do not tune these in Step 8.
SETTINGS = {
    "BCE+Dice": {
        "threshold": 0.30,
        "min_area": 50,
        "role": "final_app_model",
    },
    "BCE-only": {
        "threshold": 0.40,
        "min_area": 50,
        "role": "ablation",
    },
}

IMAGE_SIZE = 256
EXPECTED_VAL_CROPS = 1024

RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. VALIDATION-ONLY DATA LOADING
# ============================================================

def resolve_val_parquet():
    matches = sorted(glob.glob(str(ROOT / VAL_PATTERN)))

    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one VAL Parquet file. Found: {matches}"
        )

    return Path(matches[0])


def load_val_images():
    """
    Read imageA and imageB from VAL only.

    TEST is deliberately not included in this function.
    Images are returned as uint8 RGB arrays.
    """

    parquet_path = resolve_val_parquet()

    print("\nReading VAL images only:")
    print(parquet_path)

    parquet = pq.ParquetFile(parquet_path)

    images_a = []
    images_b = []

    for batch in parquet.iter_batches(
        batch_size=128,
        columns=["imageA", "imageB"],
    ):
        for row in batch.to_pylist():

            bytes_a = row["imageA"]["bytes"]
            bytes_b = row["imageB"]["bytes"]

            a = tf.io.decode_jpeg(
                bytes_a,
                channels=3,
                dct_method="INTEGER_ACCURATE",
            ).numpy()

            b = tf.io.decode_jpeg(
                bytes_b,
                channels=3,
                dct_method="INTEGER_ACCURATE",
            ).numpy()

            images_a.append(a)
            images_b.append(b)

    images_a = np.asarray(images_a, dtype=np.uint8)
    images_b = np.asarray(images_b, dtype=np.uint8)

    if len(images_a) != EXPECTED_VAL_CROPS:
        raise AssertionError(
            f"Expected {EXPECTED_VAL_CROPS} VAL crops, "
            f"found {len(images_a)}"
        )

    print("VAL images A:", images_a.shape)
    print("VAL images B:", images_b.shape)

    return images_a, images_b


# ============================================================
# 3. LOAD CACHED VALIDATION ARRAYS
# ============================================================

def load_cached_arrays():

    required = [
        PROB_DICE_PATH,
        PROB_BCE_PATH,
        GT_CACHE_PATH,
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                f"Required cache file missing: {path}"
            )

    prob_dice = np.load(PROB_DICE_PATH)
    prob_bce = np.load(PROB_BCE_PATH)
    gt = np.load(GT_CACHE_PATH)

    # Remove final singleton channel if present.
    if prob_dice.ndim == 4:
        prob_dice = prob_dice[..., 0]

    if prob_bce.ndim == 4:
        prob_bce = prob_bce[..., 0]

    if gt.ndim == 4:
        gt = gt[..., 0]

    gt = (gt > 0).astype(np.uint8)

    expected_shape = (
        EXPECTED_VAL_CROPS,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    for name, array in [
        ("BCE+Dice probabilities", prob_dice),
        ("BCE probabilities", prob_bce),
        ("Ground truth", gt),
    ]:
        if array.shape != expected_shape:
            raise AssertionError(
                f"{name} has shape {array.shape}; "
                f"expected {expected_shape}"
            )

    if not np.all(np.isfinite(prob_dice)):
        raise ValueError("BCE+Dice cache contains invalid values.")

    if not np.all(np.isfinite(prob_bce)):
        raise ValueError("BCE cache contains invalid values.")

    print("\nCached arrays loaded:")
    print("BCE+Dice:", prob_dice.shape, prob_dice.dtype)
    print("BCE-only:", prob_bce.shape, prob_bce.dtype)
    print("GT:", gt.shape, gt.dtype)

    return (
        prob_dice.astype(np.float32),
        prob_bce.astype(np.float32),
        gt,
    )


# ============================================================
# 4. CONNECTED COMPONENT POSTPROCESSING
# ============================================================

def clean_mask(probability, threshold, min_area):
    """
    Threshold probabilities and retain connected components
    whose area is at least min_area.

    Supports min_area=0.
    Uses 8-connectivity.
    """

    binary = (
        np.asarray(probability) > threshold
    ).astype(np.uint8)

    if min_area == 0:
        return binary

    n_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
    )

    cleaned = np.zeros_like(binary)

    for label_id in range(1, n_labels):

        area = stats[
            label_id,
            cv2.CC_STAT_AREA,
        ]

        if area >= min_area:
            cleaned[labels == label_id] = 1

    return cleaned


def count_regions(mask, min_area=50):
    """
    Count connected components after applying the same
    minimum-area rule.
    """

    binary = (np.asarray(mask) > 0).astype(np.uint8)

    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    count = 0

    for label_id in range(1, n_labels):

        area = stats[
            label_id,
            cv2.CC_STAT_AREA,
        ]

        if area >= min_area:
            count += 1

    return int(count)


# ============================================================
# 5. PIXEL METRICS
# ============================================================

def calculate_counts(gt, pred):

    gt = np.asarray(gt).astype(bool)
    pred = np.asarray(pred).astype(bool)

    tp = int(np.sum(gt & pred))
    fp = int(np.sum(~gt & pred))
    fn = int(np.sum(gt & ~pred))
    tn = int(np.sum(~gt & ~pred))

    return tp, fp, fn, tn


def metrics_from_counts(tp, fp, fn, tn):

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

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
    }


def evaluate_masks(gt_masks, pred_masks):

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    for gt, pred in zip(gt_masks, pred_masks):

        tp, fp, fn, tn = calculate_counts(gt, pred)

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

    result = metrics_from_counts(
        total_tp,
        total_fp,
        total_fn,
        total_tn,
    )

    result.update({
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "TN": total_tn,
    })

    return result


def empty_crop_false_alarm(gt_masks, pred_masks):

    empty_indices = [
        i for i, gt in enumerate(gt_masks)
        if not np.any(gt)
    ]

    if not empty_indices:
        return 0.0, 0, 0

    alarms = sum(
        int(np.any(pred_masks[i]))
        for i in empty_indices
    )

    far = alarms / len(empty_indices)

    return far, alarms, len(empty_indices)


# ============================================================
# PART A — MATCHED REGION COUNT SETTINGS
# ============================================================

def run_part_a(prob_dice, prob_bce, gt):

    print("\n" + "=" * 90)
    print("PART A — MATCHED REGION COUNT SETTINGS")
    print("=" * 90)

    model_probabilities = {
        "BCE+Dice": prob_dice,
        "BCE-only": prob_bce,
    }

    rows = []

    for model_name, probabilities in model_probabilities.items():

        threshold = SETTINGS[model_name]["threshold"]
        min_area = SETTINGS[model_name]["min_area"]

        predictions = []
        gt_region_counts = []
        pred_region_counts = []

        for i in range(len(gt)):

            pred = clean_mask(
                probabilities[i],
                threshold,
                min_area,
            )

            predictions.append(pred)

            # Both GT and predictions use min_area=50.
            gt_region_counts.append(
                count_regions(gt[i], min_area)
            )

            pred_region_counts.append(
                count_regions(pred, min_area)
            )

        predictions = np.asarray(predictions, dtype=np.uint8)

        pixel_metrics = evaluate_masks(gt, predictions)

        far, alarms, empty_count = empty_crop_false_alarm(
            gt,
            predictions,
        )

        gt_region_counts = np.asarray(gt_region_counts)
        pred_region_counts = np.asarray(pred_region_counts)

        count_errors = np.abs(
            pred_region_counts - gt_region_counts
        )

        row = {
            "model": model_name,
            "threshold": threshold,
            "min_area": min_area,
            "precision": pixel_metrics["precision"],
            "recall": pixel_metrics["recall"],
            "f1": pixel_metrics["f1"],
            "iou": pixel_metrics["iou"],
            "empty_crop_false_alarm": far,
            "empty_crops": empty_count,
            "empty_crop_alarms": alarms,
            "mean_gt_regions": float(np.mean(gt_region_counts)),
            "mean_predicted_regions": float(np.mean(pred_region_counts)),
            "count_mae": float(np.mean(count_errors)),
            "exact_count_pct": float(
                np.mean(count_errors == 0) * 100
            ),
        }

        rows.append(row)

    table = pd.DataFrame(rows)

    output_path = RESULT_DIR / "part_a_matched_counts.csv"
    table.to_csv(output_path, index=False)

    final_settings = {
        "chosen_model": "BCE+Dice",
        "model_path": str(
            DICE_MODEL_PATH.relative_to(ROOT)
        ),
        "threshold": 0.30,
        "min_area": 50,
        "ablation_model": "BCE-only",
        "ablation_threshold": 0.40,
        "ablation_min_area": 50,
        "split": "validation",
    }

    settings_path = (
        ROOT / "experiments" / "results" / "final_settings.json"
    )

    settings_path.parent.mkdir(parents=True, exist_ok=True)

    with open(settings_path, "w", encoding="utf-8") as f:
        json.dump(final_settings, f, indent=4)

    print("\nPART A RESULTS")
    print(table.to_string(index=False))

    print("\nSaved:", output_path)
    print("Saved:", settings_path)

    return table


# ============================================================
# 6. EDGE-REPLICATION SHIFT
# ============================================================

def shift_with_edge_replication(array, dx, dy):
    """
    Shift an HxWxC or HxW array.

    Positive dx = right.
    Positive dy = down.

    Uses edge replication:
    - Pad the exposed side with its nearest edge.
    - Crop back to the original size.
    - Never wraps pixels around.
    """

    array = np.asarray(array)

    if array.ndim not in (2, 3):
        raise ValueError("Expected HxW or HxWxC array.")

    h, w = array.shape[:2]

    pad_width = (
        (max(dy, 0), max(-dy, 0)),
        (max(dx, 0), max(-dx, 0)),
    )

    if array.ndim == 3:
        pad_width += ((0, 0),)

    padded = np.pad(
        array,
        pad_width,
        mode="edge",
    )

    start_y = max(-dy, 0)
    start_x = max(-dx, 0)

    shifted = padded[
        start_y:start_y + h,
        start_x:start_x + w,
    ]

    return shifted


# ============================================================
# PART B — SHIFTED IMAGE AND SHIFTED LABEL CHECK
# ============================================================

def run_part_b(
    images_a,
    images_b,
    gt,
):

    print("\n" + "=" * 90)
    print("PART B — MISALIGNMENT WITH UNCHANGED AND SHIFTED LABELS")
    print("=" * 90)

    if not DICE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"BCE+Dice model not found: {DICE_MODEL_PATH}"
        )

    print("\nLoading BCE+Dice model for shifted-image inference.")

    # Import custom model component for Keras deserialization.
    from src.model import AbsDiff

    model = tf.keras.models.load_model(
        DICE_MODEL_PATH,
        compile=False,
        custom_objects={
            "AbsDiff": AbsDiff,
            "GeoChange>AbsDiff": AbsDiff,
        },
    )

    threshold = SETTINGS["BCE+Dice"]["threshold"]
    min_area = SETTINGS["BCE+Dice"]["min_area"]

    shifts = {
        "horizontal": {
            0: (0, 0),
            1: (1, 0),
            3: (3, 0),
            5: (5, 0),
        },
        "diagonal": {
            0: (0, 0),
            1: (1, 1),
            3: (3, 3),
            5: (5, 5),
        },
    }

    rows = []

    for direction, shift_map in shifts.items():

        for k, (dx, dy) in shift_map.items():

            print(
                f"\nDirection={direction}, shift={k}px "
                f"(dx={dx}, dy={dy})"
            )

            shifted_images_b = np.asarray([
                shift_with_edge_replication(
                    image,
                    dx=dx,
                    dy=dy,
                )
                for image in images_b
            ], dtype=np.uint8)

            # Convert to float and apply the same preprocessing
            # used by the project's MobileNetV2 input pipeline.
            batch_size = 16

            pred_masks = []

            for start in range(0, len(images_a), batch_size):

                end = min(start + batch_size, len(images_a))

                batch_a = images_a[start:end].astype(np.float32)
                batch_b = shifted_images_b[start:end].astype(np.float32)

                batch_a = tf.keras.applications.mobilenet_v2.preprocess_input(
                    batch_a
                )

                batch_b = tf.keras.applications.mobilenet_v2.preprocess_input(
                    batch_b
                )

                probabilities = model(
                    (batch_a, batch_b),
                    training=False,
                ).numpy()

                for probability in probabilities:

                    pred = clean_mask(
                        probability[..., 0],
                        threshold,
                        min_area,
                    )

                    pred_masks.append(pred)

            pred_masks = np.asarray(pred_masks, dtype=np.uint8)

            # Condition I: compare shifted-image predictions to original GT.
            metrics_original = evaluate_masks(
                gt,
                pred_masks,
            )

            far_original, _, _ = empty_crop_false_alarm(
                gt,
                pred_masks,
            )

            # Condition II: shift GT by exactly the same displacement.
            shifted_gt = np.asarray([
                shift_with_edge_replication(
                    mask,
                    dx=dx,
                    dy=dy,
                )
                for mask in gt
            ], dtype=np.uint8)

            metrics_shifted = evaluate_masks(
                shifted_gt,
                pred_masks,
            )

            far_shifted, _, _ = empty_crop_false_alarm(
                shifted_gt,
                pred_masks,
            )

            rows.append({
                "direction": direction,
                "shift_pixels": k,
                "dx": dx,
                "dy": dy,

                "unshifted_gt_precision": metrics_original["precision"],
                "unshifted_gt_recall": metrics_original["recall"],
                "unshifted_gt_f1": metrics_original["f1"],
                "unshifted_gt_empty_far": far_original,

                "shifted_gt_precision": metrics_shifted["precision"],
                "shifted_gt_recall": metrics_shifted["recall"],
                "shifted_gt_f1": metrics_shifted["f1"],
                "shifted_gt_empty_far": far_shifted,
            })

    table = pd.DataFrame(rows)

    csv_path = RESULT_DIR / "part_b_shifted_label_misalignment.csv"
    table.to_csv(csv_path, index=False)

    print("\nPART B RESULTS")
    print(table.to_string(index=False))

    # Four lines:
    # Horizontal, unshifted GT
    # Horizontal, shifted GT
    # Diagonal, unshifted GT
    # Diagonal, shifted GT

    fig, ax = plt.subplots(figsize=(10, 6))

    line_definitions = [
        (
            "horizontal",
            "unshifted_gt_f1",
            "Horizontal — GT unshifted",
        ),
        (
            "horizontal",
            "shifted_gt_f1",
            "Horizontal — GT shifted",
        ),
        (
            "diagonal",
            "unshifted_gt_f1",
            "Diagonal — GT unshifted",
        ),
        (
            "diagonal",
            "shifted_gt_f1",
            "Diagonal — GT shifted",
        ),
    ]

    for direction, column, label in line_definitions:

        subset = table[
            table["direction"] == direction
        ].sort_values("shift_pixels")

        ax.plot(
            subset["shift_pixels"],
            subset[column],
            marker="o",
            linewidth=2,
            label=label,
        )

    ax.set_title(
        "BCE+Dice: F1 under ImageB Misalignment"
    )
    ax.set_xlabel("ImageB shift (pixels)")
    ax.set_ylabel("F1")
    ax.set_xticks([0, 1, 3, 5])
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    plot_path = RESULT_DIR / "part_b_f1_vs_shift.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    print("\nSaved:", csv_path)
    print("Saved:", plot_path)

    print("\nPART B — INTERPRETATION")
    print(
        "1. Condition (ii), where both imageB and GT are shifted equally, "
        "isolates the effect of changing the image-pair alignment while "
        "keeping the prediction and reference label in the same shifted "
        "coordinate frame."
    )
    print(
        "2. Comparing condition (i) with condition (ii) shows how much of "
        "the measured performance loss comes from evaluating a shifted "
        "prediction against the original, unshifted label."
    )
    print(
        "3. Condition (ii) does not isolate image registration perfectly "
        "or prove that the model is robust to real-world registration "
        "errors; it also shifts the label using edge replication and "
        "therefore changes the evaluation reference near image borders."
    )

    return table


# ============================================================
# PART C — GROUND-TRUTH CHANGE BUCKETS
# ============================================================

def run_part_c_buckets(prob_dice, gt):

    print("\n" + "=" * 90)
    print("PART C1 — ERROR METRICS BY GROUND-TRUTH CHANGE BUCKET")
    print("=" * 90)

    threshold = SETTINGS["BCE+Dice"]["threshold"]
    min_area = SETTINGS["BCE+Dice"]["min_area"]

    predictions = np.asarray([
        clean_mask(
            prob_dice[i],
            threshold,
            min_area,
        )
        for i in range(len(gt))
    ], dtype=np.uint8)

    change_shares = np.mean(
        gt > 0,
        axis=(1, 2),
    )

    buckets = [
        ("0%", change_shares == 0),
        (">0 to <5%", (change_shares > 0) & (change_shares < 0.05)),
        ("5 to 20%", (change_shares >= 0.05) & (change_shares <= 0.20)),
        (">20%", change_shares > 0.20),
    ]

    rows = []

    for bucket_name, indices_mask in buckets:

        indices = np.where(indices_mask)[0]

        if len(indices) == 0:
            continue

        bucket_gt = gt[indices]
        bucket_pred = predictions[indices]

        metrics = evaluate_masks(
            bucket_gt,
            bucket_pred,
        )

        gt_region_counts = [
            count_regions(mask, min_area)
            for mask in bucket_gt
        ]

        row = {
            "gt_change_bucket": bucket_name,
            "number_of_crops": len(indices),
            "pooled_precision": metrics["precision"],
            "pooled_recall": metrics["recall"],
            "pooled_f1": metrics["f1"],
            "pooled_iou": metrics["iou"],
            "mean_gt_regions": float(
                np.mean(gt_region_counts)
            ),
        }

        if bucket_name == "0%":

            far, alarms, empty_count = empty_crop_false_alarm(
                bucket_gt,
                bucket_pred,
            )

            row["false_alarm_rate"] = far
            row["false_alarm_crops"] = alarms
            row["empty_crops"] = empty_count

        else:
            row["false_alarm_rate"] = np.nan
            row["false_alarm_crops"] = np.nan
            row["empty_crops"] = np.nan

        rows.append(row)

    table = pd.DataFrame(rows)

    csv_path = RESULT_DIR / "part_c_change_buckets.csv"
    table.to_csv(csv_path, index=False)

    print("\nCHANGE BUCKET RESULTS")
    print(table.to_string(index=False))

    print("\nSaved:", csv_path)

    return predictions, change_shares, table


# ============================================================
# 7. VISUAL ERROR MAPS AND GRID SAVING
# ============================================================

def make_error_map(gt, pred):
    """
    Error map:
    - Black: correct background / correct prediction.
    - Red: false positive.
    - Blue: false negative.
    - Green: true positive.
    """

    gt = gt.astype(bool)
    pred = pred.astype(bool)

    h, w = gt.shape

    error = np.zeros((h, w, 3), dtype=np.uint8)

    true_positive = gt & pred
    false_positive = ~gt & pred
    false_negative = gt & ~pred

    # RGB display colours.
    error[true_positive] = [0, 180, 0]
    error[false_positive] = [255, 0, 0]
    error[false_negative] = [0, 80, 255]

    return error


def create_grid(
    indices,
    images_a,
    images_b,
    gt,
    predictions,
    change_shares,
    title,
    output_name,
):
    """
    12 crops per grid.

    Each crop has five columns:
    A | B | GT | Prediction | Error map.

    Index and GT change share are printed beneath each image.
    """

    n = len(indices)

    fig, axes = plt.subplots(
        n,
        5,
        figsize=(17, 3.1 * n),
    )

    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    column_titles = [
        "Image A",
        "Image B",
        "Ground Truth",
        "Prediction",
        "Error Map",
    ]

    for col, name in enumerate(column_titles):
        axes[0, col].set_title(
            name,
            fontsize=11,
            fontweight="bold",
        )

    for row, index in enumerate(indices):

        error_map = make_error_map(
            gt[index],
            predictions[index],
        )

        panels = [
            images_a[index],
            images_b[index],
            gt[index],
            predictions[index],
            error_map,
        ]

        for col, panel in enumerate(panels):

            ax = axes[row, col]

            if col in (2, 3):
                ax.imshow(
                    panel,
                    cmap="gray",
                    vmin=0,
                    vmax=1,
                )
            else:
                ax.imshow(panel)

            ax.axis("off")

            ax.set_xlabel(
                f"Index {index}\n"
                f"GT change: {change_shares[index] * 100:.2f}%",
                fontsize=8,
            )

    fig.suptitle(
        title,
        fontsize=15,
        fontweight="bold",
        y=1.002,
    )

    fig.tight_layout()

    output_path = RESULT_DIR / output_name

    fig.savefig(
        output_path,
        dpi=140,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("Saved grid:", output_path)

    return output_path


def run_part_c_grids(
    images_a,
    images_b,
    gt,
    predictions,
    change_shares,
):
    print("\n" + "=" * 90)
    print("PART C2 — VISUAL ERROR GRIDS")
    print("=" * 90)

    fp_counts = np.sum(
        (predictions == 1) & (gt == 0),
        axis=(1, 2),
    )

    fn_counts = np.sum(
        (predictions == 0) & (gt == 1),
        axis=(1, 2),
    )

    fp_indices = np.argsort(fp_counts)[::-1][:12]
    fn_indices = np.argsort(fn_counts)[::-1][:12]

    rng = np.random.default_rng(0)

    random_indices = rng.choice(
        len(gt),
        size=12,
        replace=False,
    )

    print("\nTOP FALSE-POSITIVE CROPS")

    print([
        {
            "index": int(i),
            "fp_pixels": int(fp_counts[i]),
            "gt_change_pct": round(
                float(change_shares[i] * 100), 3
            ),
        }
        for i in fp_indices
    ])

    print("\nTOP FALSE-NEGATIVE CROPS")

    print([
        {
            "index": int(i),
            "fn_pixels": int(fn_counts[i]),
            "gt_change_pct": round(
                float(change_shares[i] * 100), 3
            ),
        }
        for i in fn_indices
    ])

    print("\nRANDOM TYPICAL CROPS — SEED 0")

    print([
        {
            "index": int(i),
            "gt_change_pct": round(
                float(change_shares[i] * 100), 3
            ),
        }
        for i in random_indices
    ])

    create_grid(
        fp_indices,
        images_a,
        images_b,
        gt,
        predictions,
        change_shares,
        "12 Crops with Most False-Positive Pixels",
        "grid_top_false_positives.png",
    )

    create_grid(
        fn_indices,
        images_a,
        images_b,
        gt,
        predictions,
        change_shares,
        "12 Crops with Most False-Negative Pixels",
        "grid_top_false_negatives.png",
    )

    create_grid(
        random_indices,
        images_a,
        images_b,
        gt,
        predictions,
        change_shares,
        "12 Random Validation Crops — Seed 0",
        "grid_random_examples_seed0.png",
    )

    # Save crop selection details.
    selection_rows = []

    for group_name, indices in [
        ("top_false_positive", fp_indices),
        ("top_false_negative", fn_indices),
        ("random_seed_0", random_indices),
    ]:

        for index in indices:

            selection_rows.append({
                "group": group_name,
                "crop_index": int(index),
                "gt_change_share": float(change_shares[index]),
                "gt_change_pct": float(
                    change_shares[index] * 100
                ),
                "fp_pixels": int(fp_counts[index]),
                "fn_pixels": int(fn_counts[index]),
            })

    selection_table = pd.DataFrame(selection_rows)

    selection_path = (
        RESULT_DIR / "part_c_visual_grid_selections.csv"
    )

    selection_table.to_csv(
        selection_path,
        index=False,
    )

    print("\nSaved:", selection_path)

    return fp_counts, fn_counts, selection_table


# ============================================================
# 8. OBSERVED FAILURE PATTERNS
# ============================================================

def print_failure_pattern_candidates(
    fp_indices,
    fn_indices,
    random_indices,
):
    """
    Do not automatically invent image-level observations.

    The grids must be visually inspected before recording
    observed patterns.
    """

    print("\n" + "=" * 90)
    print("PART C3 — FAILURE PATTERNS")
    print("=" * 90)

    print(
        "The following are the grids to inspect:"
    )

    print(
        "1. grid_top_false_positives.png"
    )
    print(
        "2. grid_top_false_negatives.png"
    )
    print(
        "3. grid_random_examples_seed0.png"
    )

    print(
        "\nOBSERVATION RULE:"
    )

    print(
        "Only write a pattern as OBSERVED if it is visibly present "
        "in the saved grids. Do not infer its cause."
    )

    print(
        "\nRecord patterns in this format:"
    )

    print(
        "OBSERVED — [describe the visible FP/FN pattern]."
    )

    print(
        "OBSERVED — [describe another visible pattern]."
    )

    print(
        "OBSERVED — [describe another visible pattern]."
    )

    print(
        "\nNo visual failure pattern is automatically claimed by "
        "this script because the images must be inspected."
    )


# ============================================================
# 9. MAIN
# ============================================================

def main():

    print("\n" + "=" * 90)
    print("GEOCHANGE-AI — STEP 8 VALIDATION ANALYSIS")
    print("=" * 90)

    print("Split: VAL ONLY")
    print("TEST: NEVER LOADED OR ACCESSED")

    print("\nFinal app model:")
    print("BCE+Dice | threshold=0.30 | min_area=50")

    print("\nAblation:")
    print("BCE-only | threshold=0.40 | min_area=50")

    # Load cached probabilities and GT.
    prob_dice, prob_bce, gt = load_cached_arrays()

    # Part A uses only cached arrays.
    part_a_table = run_part_a(
        prob_dice,
        prob_bce,
        gt,
    )

    # Part B needs original VAL images and model inference.
    images_a, images_b = load_val_images()

    part_b_table = run_part_b(
        images_a,
        images_b,
        gt,
    )

    # Part C1 uses cached BCE+Dice probabilities.
    predictions, change_shares, bucket_table = (
        run_part_c_buckets(
            prob_dice,
            gt,
        )
    )

    # Part C2 creates the visual grids.
    fp_counts, fn_counts, selections = run_part_c_grids(
        images_a,
        images_b,
        gt,
        predictions,
        change_shares,
    )

    # Part C3: visual inspection is required.
    print_failure_pattern_candidates(
        np.argsort(fp_counts)[::-1][:12],
        np.argsort(fn_counts)[::-1][:12],
        np.random.default_rng(0).choice(
            len(gt),
            size=12,
            replace=False,
        ),
    )

    print("\n" + "=" * 90)
    print("STEP 8 COMPLETE")
    print("=" * 90)

    print("\nRESULT DIRECTORY:")
    print(RESULT_DIR)

    print("\nOUTPUT FILES:")
    for path in sorted(RESULT_DIR.iterdir()):
        print(path.name)

    print("\nFINAL SETTINGS:")
    print(
        ROOT / "experiments" / "results" / "final_settings.json"
    )

    print("\nTEST SPLIT: NEVER LOADED OR EVALUATED.")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 13:14:01 2026

@author: Admin
"""

# ============================================================
# GeoChange-AI — STEP 7A
# Identical Post-processing + Threshold Tuning + Bootstrap
#
# VALIDATION ONLY.
# TEST IS NEVER LOADED OR ACCESSED.
#
# Run:
# python -m scripts.eval_val_both
# ============================================================

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import glob
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import tensorflow as tf
import keras
import yaml

import matplotlib
matplotlib.use("Agg")

# Importing the model registers the custom AbsDiff layer.
from src.model import AbsDiff


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = ROOT / "configs" / "default.yaml"

CACHE_DIR = ROOT / "data" / "processed" / "cache"

RESULT_DIR = ROOT / "experiments" / "results"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "BCE+Dice": (
        ROOT
        / "experiments"
        / "runs"
        / "bcedice_v1"
        / "best_model.keras"
    ),

    "BCE-only": (
        ROOT
        / "experiments"
        / "runs"
        / "bce_v1"
        / "best_model.keras"
    ),
}

THRESHOLDS = np.round(
    np.arange(0.10, 0.901, 0.05),
    2
)

MIN_AREAS = [0, 10, 20, 50, 100, 200]

BATCH_SIZE = 16

BOOTSTRAP_ITERATIONS = 1000
BOOTSTRAP_SEED = 42

EXPECTED_COUNT = 1024

IMAGE_SIZE = 256


# ============================================================
# 2. LOAD VALIDATION PARQUET ONLY
# ============================================================

def get_val_parquet():

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    pattern = config["paths"]["val_parquet"]

    if not Path(pattern).is_absolute():
        pattern = str(ROOT / pattern)

    matches = sorted(glob.glob(pattern))

    if len(matches) != 1:
        raise FileNotFoundError(
            "Expected exactly one validation Parquet file. "
            f"Found: {matches}"
        )

    path = Path(matches[0])

    if "test" in path.name.lower():
        raise RuntimeError("TEST access is prohibited.")

    print("\nValidation Parquet:")
    print(path)

    return path


def iter_val_rows():

    """
    Read only the validation Parquet.

    Does not import or use src.data.load_data().
    Does not resolve train or test paths.
    """

    parquet_path = get_val_parquet()

    parquet = pq.ParquetFile(parquet_path)

    count = 0

    for batch in parquet.iter_batches(
        batch_size=32,
        columns=["imageA", "imageB", "label"]
    ):

        for row in batch.to_pylist():

            image_a = row["imageA"]["bytes"]
            image_b = row["imageB"]["bytes"]
            mask = row["label"]["bytes"]

            if image_a is None or image_b is None or mask is None:
                raise ValueError(
                    "Null image or mask in validation data."
                )

            count += 1

            yield image_a, image_b, mask

    if count != EXPECTED_COUNT:

        raise AssertionError(
            f"Expected {EXPECTED_COUNT} validation samples, "
            f"found {count}"
        )


# ============================================================
# 3. DECODE AND PREPROCESS
# ============================================================

def decode_sample(bytes_a, bytes_b, bytes_mask):

    image_a = tf.io.decode_jpeg(
        bytes_a,
        channels=3,
        dct_method="INTEGER_ACCURATE"
    )

    image_b = tf.io.decode_jpeg(
        bytes_b,
        channels=3,
        dct_method="INTEGER_ACCURATE"
    )

    mask = tf.io.decode_png(
        bytes_mask,
        channels=1
    )

    image_a = tf.cast(image_a, tf.float32)
    image_b = tf.cast(image_b, tf.float32)

    mask = tf.cast(mask > 0, tf.uint8)

    image_a = tf.keras.applications.mobilenet_v2.preprocess_input(
        image_a
    )

    image_b = tf.keras.applications.mobilenet_v2.preprocess_input(
        image_b
    )

    return (
        image_a.numpy(),
        image_b.numpy(),
        mask.numpy()[..., 0]
    )


# ============================================================
# 4. MODEL LOADING
# ============================================================

def load_trained_model(path):

    if not path.exists():

        raise FileNotFoundError(
            f"Model checkpoint not found: {path}"
        )

    print("\nLoading:", path)

    model = keras.models.load_model(
        path,
        compile=False,
        custom_objects={
            "AbsDiff": AbsDiff,
            "GeoChange>AbsDiff": AbsDiff
        }
    )

    print("Loaded successfully.")
    print("Output shape:", model.output_shape)

    return model


# ============================================================
# 5. VALIDATION PREDICTION CACHE
# ============================================================

def cache_paths(model_name):

    safe_name = (
        model_name.lower()
        .replace("+", "_")
        .replace("-", "_")
    )

    probability_path = (
        CACHE_DIR / f"{safe_name}_val_probabilities.npy"
    )

    return probability_path


GT_CACHE_PATH = CACHE_DIR / "val_ground_truth.npy"


def generate_validation_cache():

    """
    Generate one prediction per model per validation crop.

    Probability cache:
        float16, shape (1024, 256, 256)

    Ground-truth cache:
        uint8, shape (1024, 256, 256)

    Existing caches are reused.
    """

    probability_paths = {
        name: cache_paths(name)
        for name in MODELS
    }

    gt_exists = GT_CACHE_PATH.exists()

    all_prob_exists = all(
        path.exists()
        for path in probability_paths.values()
    )

    if gt_exists and all_prob_exists:

        print("\nAll validation caches already exist.")
        print("Reusing cached arrays.")

        gt = np.load(
            GT_CACHE_PATH,
            mmap_mode="r"
        )

        for name, path in probability_paths.items():

            probabilities = np.load(
                path,
                mmap_mode="r"
            )

            if probabilities.shape != (
                EXPECTED_COUNT,
                IMAGE_SIZE,
                IMAGE_SIZE
            ):

                raise ValueError(
                    f"Invalid cache shape for {name}: "
                    f"{probabilities.shape}"
                )

        if gt.shape != (
            EXPECTED_COUNT,
            IMAGE_SIZE,
            IMAGE_SIZE
        ):

            raise ValueError(
                f"Invalid GT cache shape: {gt.shape}"
            )

        return probability_paths, GT_CACHE_PATH

    print("\nGenerating validation predictions.")

    models = {
        name: load_trained_model(path)
        for name, path in MODELS.items()
    }

    probabilities = {
        name: np.zeros(
            (
                EXPECTED_COUNT,
                IMAGE_SIZE,
                IMAGE_SIZE
            ),
            dtype=np.float16
        )
        for name in MODELS
    }

    ground_truth = np.zeros(
        (
            EXPECTED_COUNT,
            IMAGE_SIZE,
            IMAGE_SIZE
        ),
        dtype=np.uint8
    )

    batch_a = []
    batch_b = []

    position = 0

    def predict_batch():

        nonlocal position

        if not batch_a:
            return

        a = np.asarray(
            batch_a,
            dtype=np.float32
        )

        b = np.asarray(
            batch_b,
            dtype=np.float32
        )

        for name, model in models.items():

            output = model.predict(
                [a, b],
                verbose=0
            )

            output = np.asarray(output)

            if output.ndim == 4:
                output = output[..., 0]

            end = position + len(output)

            probabilities[name][position:end] = (
                output.astype(np.float16)
            )

        position += len(a)

        batch_a.clear()
        batch_b.clear()

    start = time.time()

    for bytes_a, bytes_b, bytes_mask in iter_val_rows():

        image_a, image_b, mask = decode_sample(
            bytes_a,
            bytes_b,
            bytes_mask
        )

        batch_a.append(image_a)
        batch_b.append(image_b)

        ground_truth[position + len(batch_a) - 1] = mask

        if len(batch_a) == BATCH_SIZE:
            predict_batch()

        if position % 160 == 0 and position > 0:

            print(
                f"Processed approximately {position}/"
                f"{EXPECTED_COUNT} crops"
            )

    predict_batch()

    if position != EXPECTED_COUNT:

        raise AssertionError(
            f"Expected {EXPECTED_COUNT} predictions, "
            f"got {position}"
        )

    print(
        f"\nPrediction generation time: "
        f"{time.time() - start:.1f} seconds"
    )

    np.save(
        GT_CACHE_PATH,
        ground_truth
    )

    for name, path in probability_paths.items():

        np.save(
            path,
            probabilities[name]
        )

        print("Saved:", path)

    print("Saved:", GT_CACHE_PATH)

    del models
    del probabilities
    del ground_truth

    return probability_paths, GT_CACHE_PATH


# ============================================================
# 6. CONNECTED COMPONENTS
# ============================================================

def component_labels(binary_mask):

    """
    Calculate connected components exactly once.

    Uses 8-connectivity, matching src.postprocess.clean_mask.
    """

    binary_mask = np.asarray(
        binary_mask,
        dtype=np.uint8
    )

    n_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary_mask,
            connectivity=8
        )
    )

    areas = stats[1:, cv2.CC_STAT_AREA]

    return labels, areas


def filter_components(labels, areas, min_area):

    """
    Reuse previously calculated component labels.

    min_area=0 means no component is removed.
    """

    if min_area == 0:

        return (labels > 0).astype(np.uint8), len(areas)

    keep_ids = (
        np.where(areas >= min_area)[0] + 1
    )

    if len(keep_ids) == 0:

        return np.zeros(
            labels.shape,
            dtype=np.uint8
        ), 0

    kept = np.isin(
        labels,
        keep_ids
    ).astype(np.uint8)

    return kept, len(keep_ids)


def clean_mask_same_rules(
    probability,
    threshold,
    min_area
):

    """
    Same binary thresholding and 8-connected component
    area filtering as src.postprocess.clean_mask.

    Extended to allow min_area=0.
    """

    binary = (
        np.asarray(probability) > threshold
    ).astype(np.uint8)

    labels, areas = component_labels(binary)

    return filter_components(
        labels,
        areas,
        min_area
    )


# ============================================================
# 7. METRICS
# ============================================================

def confusion(gt, pred):

    gt = np.asarray(gt, dtype=bool)
    pred = np.asarray(pred, dtype=bool)

    tp = int(np.sum(gt & pred))
    fp = int(np.sum(~gt & pred))
    fn = int(np.sum(gt & ~pred))
    tn = int(np.sum(~gt & ~pred))

    return tp, fp, fn, tn


def pooled_scores(tp, fp, fn, tn):

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    f1 = (
        2 * tp / (2 * tp + fp + fn)
        if 2 * tp + fp + fn > 0
        else 0.0
    )

    iou = (
        tp / (tp + fp + fn)
        if tp + fp + fn > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou
    }


def evaluate_predictions(predictions, ground_truth):

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    empty_crops = 0
    empty_false_alarms = 0

    region_counts = []

    per_crop_counts = []

    for pred, gt in zip(
        predictions,
        ground_truth
    ):

        tp, fp, fn, tn = confusion(
            gt,
            pred
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

        per_crop_counts.append(
            (tp, fp, fn)
        )

        labels, areas = component_labels(pred)

        _, regions = filter_components(
            labels,
            areas,
            min_area=0
        )

        region_counts.append(regions)

        if not np.any(gt):

            empty_crops += 1

            if np.any(pred):
                empty_false_alarms += 1

    scores = pooled_scores(
        total_tp,
        total_fp,
        total_fn,
        total_tn
    )

    scores["empty_far"] = (
        empty_false_alarms / empty_crops
        if empty_crops
        else 0.0
    )

    scores["mean_regions"] = float(
        np.mean(region_counts)
    )

    scores["TP"] = total_tp
    scores["FP"] = total_fp
    scores["FN"] = total_fn
    scores["TN"] = total_tn

    scores["per_crop_counts"] = per_crop_counts

    return scores


# ============================================================
# 8. THRESHOLD + MIN_AREA GRID
# ============================================================

def evaluate_grid(probabilities, ground_truth, model_name):

    print("\n" + "=" * 70)
    print("GRID SEARCH:", model_name)
    print("=" * 70)

    results = []

    start = time.time()

    for threshold in THRESHOLDS:

        # Store component labels once per crop for this threshold.
        all_labels = []
        all_areas = []

        for probability in probabilities:

            binary = (
                probability.astype(np.float32) > threshold
            ).astype(np.uint8)

            labels, areas = component_labels(binary)

            all_labels.append(labels)
            all_areas.append(areas)

        for min_area in MIN_AREAS:

            total_tp = 0
            total_fp = 0
            total_fn = 0
            total_tn = 0

            empty_crops = 0
            empty_false_alarms = 0

            total_regions = 0

            for i in range(len(ground_truth)):

                pred, regions = filter_components(
                    all_labels[i],
                    all_areas[i],
                    min_area
                )

                gt = ground_truth[i]

                tp, fp, fn, tn = confusion(
                    gt,
                    pred
                )

                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_tn += tn

                total_regions += regions

                if not np.any(gt):

                    empty_crops += 1

                    if np.any(pred):
                        empty_false_alarms += 1

            scores = pooled_scores(
                total_tp,
                total_fp,
                total_fn,
                total_tn
            )

            empty_far = (
                empty_false_alarms / empty_crops
                if empty_crops
                else 0.0
            )

            mean_regions = (
                total_regions / len(ground_truth)
            )

            results.append({

                "model": model_name,

                "threshold": float(threshold),

                "min_area": int(min_area),

                "precision": scores["precision"],

                "recall": scores["recall"],

                "f1": scores["f1"],

                "iou": scores["iou"],

                "empty_far": empty_far,

                "mean_regions": mean_regions,

                "TP": total_tp,
                "FP": total_fp,
                "FN": total_fn,
                "TN": total_tn

            })

    df = pd.DataFrame(results)

    df = df.sort_values(
        "f1",
        ascending=False
    ).reset_index(drop=True)

    print(
        f"\nGrid completed in {time.time() - start:.1f} seconds"
    )

    print("\nTOP 5 SETTINGS:")

    print(
        df.head(5).to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    return df


# ============================================================
# 9. REGION COUNT ACCURACY
# ============================================================

def region_count_metrics(
    probabilities,
    ground_truth,
    threshold,
    min_area
):

    predicted_counts = []
    gt_counts = []

    for probability, gt in zip(
        probabilities,
        ground_truth
    ):

        pred, pred_regions = clean_mask_same_rules(
            probability,
            threshold,
            min_area
        )

        gt_labels, gt_areas = component_labels(gt)

        _, true_regions = filter_components(
            gt_labels,
            gt_areas,
            min_area
        )

        predicted_counts.append(pred_regions)
        gt_counts.append(true_regions)

    predicted_counts = np.asarray(
        predicted_counts
    )

    gt_counts = np.asarray(
        gt_counts
    )

    mae = float(
        np.mean(
            np.abs(predicted_counts - gt_counts)
        )
    )

    exact_percentage = float(
        np.mean(
            predicted_counts == gt_counts
        ) * 100
    )

    return mae, exact_percentage


# ============================================================
# 10. BASELINE IMAGE DIFFERENCE
# ============================================================

def difference_image(image_a, image_b):

    """
    Same grayscale difference and Gaussian blur
    used by the existing baseline implementation.

    Inputs are raw RGB images.
    """

    image_a = np.clip(
        image_a,
        0,
        255
    ).astype(np.uint8)

    image_b = np.clip(
        image_b,
        0,
        255
    ).astype(np.uint8)

    gray_a = cv2.cvtColor(
        image_a,
        cv2.COLOR_RGB2GRAY
    )

    gray_b = cv2.cvtColor(
        image_b,
        cv2.COLOR_RGB2GRAY
    )

    difference = cv2.absdiff(
        gray_a,
        gray_b
    )

    difference = cv2.GaussianBlur(
        difference,
        (3, 3),
        0
    )

    return difference


def evaluate_baselines(ground_truth):

    print("\nEvaluating baselines with identical post-processing.")

    predictions = {
        "All-zero": [],
        "Otsu": [],
        "Global threshold": []
    }

    # Use the existing baseline's fixed global threshold.
    global_threshold = 93

    for bytes_a, bytes_b, bytes_mask in iter_val_rows():

        raw_a = tf.io.decode_jpeg(
            bytes_a,
            channels=3
        ).numpy()

        raw_b = tf.io.decode_jpeg(
            bytes_b,
            channels=3
        ).numpy()

        difference = difference_image(
            raw_a,
            raw_b
        )

        # All-zero.
        predictions["All-zero"].append(
            np.zeros(
                (IMAGE_SIZE, IMAGE_SIZE),
                dtype=np.uint8
            )
        )

        # Per-image Otsu.
        _, otsu_binary = cv2.threshold(
            difference,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        otsu_clean, _ = clean_mask_same_rules(
            otsu_binary,
            threshold=0.5,
            min_area=20
        )

        predictions["Otsu"].append(
            otsu_clean
        )

        # Fixed global grayscale threshold.
        _, global_binary = cv2.threshold(
            difference,
            global_threshold,
            255,
            cv2.THRESH_BINARY
        )

        global_clean, _ = clean_mask_same_rules(
            global_binary,
            threshold=0.5,
            min_area=20
        )

        predictions["Global threshold"].append(
            global_clean
        )

    rows = []

    for name, preds in predictions.items():

        scores = evaluate_predictions(
            preds,
            ground_truth
        )

        mae, exact = region_count_metrics(
            np.asarray(preds),
            ground_truth,
            threshold=0.5,
            min_area=0
        )

        rows.append({

            "model": name,

            "threshold": (
                "—"
                if name == "All-zero"
                else (
                    "per-image Otsu"
                    if name == "Otsu"
                    else str(global_threshold)
                )
            ),

            "min_area": 20,

            "precision": scores["precision"],

            "recall": scores["recall"],

            "f1": scores["f1"],

            "iou": scores["iou"],

            "empty_far": scores["empty_far"],

            "mean_regions": scores["mean_regions"],

            "count_mae": mae,

            "exact_count_pct": exact

        })

    return rows


# ============================================================
# 11. PAIRED BOOTSTRAP
# ============================================================

def bootstrap_f1(tp, fp, fn, indices):

    """
    Calculate pooled F1 for each bootstrap resample.
    """

    results = []

    for sample_indices in indices:

        sampled_tp = int(
            np.sum(tp[sample_indices])
        )

        sampled_fp = int(
            np.sum(fp[sample_indices])
        )

        sampled_fn = int(
            np.sum(fn[sample_indices])
        )

        denominator = (
            2 * sampled_tp
            + sampled_fp
            + sampled_fn
        )

        f1 = (
            2 * sampled_tp / denominator
            if denominator > 0
            else 0.0
        )

        results.append(f1)

    return np.asarray(results)


def run_paired_bootstrap(
    bcedice_counts,
    bce_counts
):

    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    n = EXPECTED_COUNT

    # Generate resamples once.
    # SAME indices used for both models.
    bootstrap_indices = rng.integers(
        low=0,
        high=n,
        size=(BOOTSTRAP_ITERATIONS, n)
    )

    bcedice_counts = np.asarray(
        bcedice_counts,
        dtype=np.int64
    )

    bce_counts = np.asarray(
        bce_counts,
        dtype=np.int64
    )

    bcedice_f1 = bootstrap_f1(
        bcedice_counts[:, 0],
        bcedice_counts[:, 1],
        bcedice_counts[:, 2],
        bootstrap_indices
    )

    bce_f1 = bootstrap_f1(
        bce_counts[:, 0],
        bce_counts[:, 1],
        bce_counts[:, 2],
        bootstrap_indices
    )

    difference = bcedice_f1 - bce_f1

    result = {

        "BCE+Dice F1 lower": float(
            np.percentile(bcedice_f1, 2.5)
        ),

        "BCE+Dice F1 upper": float(
            np.percentile(bcedice_f1, 97.5)
        ),

        "BCE-only F1 lower": float(
            np.percentile(bce_f1, 2.5)
        ),

        "BCE-only F1 upper": float(
            np.percentile(bce_f1, 97.5)
        ),

        "Difference lower": float(
            np.percentile(difference, 2.5)
        ),

        "Difference upper": float(
            np.percentile(difference, 97.5)
        ),

        "Difference mean": float(
            np.mean(difference)
        )

    }

    print("\n" + "=" * 70)
    print("PAIRED BOOTSTRAP — 1,000 RESAMPLES")
    print("=" * 70)

    print(
        "BCE+Dice F1 95% CI: "
        f"[{result['BCE+Dice F1 lower']:.4f}, "
        f"{result['BCE+Dice F1 upper']:.4f}]"
    )

    print(
        "BCE-only F1 95% CI: "
        f"[{result['BCE-only F1 lower']:.4f}, "
        f"{result['BCE-only F1 upper']:.4f}]"
    )

    print(
        "F1 difference (BCE+Dice minus BCE-only) "
        "95% CI: "
        f"[{result['Difference lower']:.4f}, "
        f"{result['Difference upper']:.4f}]"
    )

    print(
        "Mean bootstrap difference:",
        f"{result['Difference mean']:.4f}"
    )

    print(
        "\nThis interval measures uncertainty from sampling "
        "validation crops. It does not measure variation "
        "between independent training seeds."
    )

    return result


# ============================================================
# 12. MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("GEOCHANGE-AI — STEP 7A")
    print("VALIDATION-ONLY EVALUATION")
    print("=" * 70)

    print("\nTEST SPLIT ACCESS: DISABLED")

    # --------------------------------------------------------
    # CACHE
    # --------------------------------------------------------

    probability_paths, gt_path = (
        generate_validation_cache()
    )

    ground_truth = np.load(
        gt_path
    )

    # --------------------------------------------------------
    # GRID SEARCH
    # --------------------------------------------------------

    all_grid_results = []

    best_settings = {}

    per_crop_counts = {}

    final_model_rows = []

    for model_name, probability_path in (
        probability_paths.items()
    ):

        probabilities = np.load(
            probability_path,
            mmap_mode="r"
        )

        grid = evaluate_grid(
            probabilities,
            ground_truth,
            model_name
        )

        all_grid_results.append(grid)

        best = grid.iloc[0]

        threshold = float(
            best["threshold"]
        )

        min_area = int(
            best["min_area"]
        )

        best_settings[model_name] = {
            "threshold": threshold,
            "min_area": min_area
        }

        # Recreate selected masks for detailed metrics.
        selected_predictions = []

        for probability in probabilities:

            pred, _ = clean_mask_same_rules(
                probability,
                threshold,
                min_area
            )

            selected_predictions.append(pred)

        scores = evaluate_predictions(
            selected_predictions,
            ground_truth
        )

        mae, exact = region_count_metrics(
            probabilities,
            ground_truth,
            threshold,
            min_area
        )

        final_model_rows.append({

            "model": model_name,

            "threshold": threshold,

            "min_area": min_area,

            "precision": scores["precision"],

            "recall": scores["recall"],

            "f1": scores["f1"],

            "iou": scores["iou"],

            "empty_far": scores["empty_far"],

            "mean_regions": scores["mean_regions"],

            "count_mae": mae,

            "exact_count_pct": exact

        })

        per_crop_counts[model_name] = np.asarray(
            scores["per_crop_counts"],
            dtype=np.int64
        )

        print("\nSelected setting:", model_name)
        print("Threshold:", threshold)
        print("Minimum area:", min_area)
        print("Region count MAE:", f"{mae:.4f}")
        print("Exact region count:", f"{exact:.2f}%")

        del selected_predictions

    # --------------------------------------------------------
    # SAVE FULL GRID
    # --------------------------------------------------------

    full_grid = pd.concat(
        all_grid_results,
        ignore_index=True
    )

    grid_path = (
        RESULT_DIR / "postprocess_grid_val.csv"
    )

    full_grid.to_csv(
        grid_path,
        index=False
    )

    print("\nSaved grid:", grid_path)

    # --------------------------------------------------------
    # BASELINES
    # --------------------------------------------------------

    baseline_rows = evaluate_baselines(
        ground_truth
    )

    # --------------------------------------------------------
    # FINAL TABLE
    # --------------------------------------------------------

    final_rows = (
        baseline_rows + final_model_rows
    )

    final_table = pd.DataFrame(
        final_rows
    )

    # Save the exact selected settings for Part C.
    settings_path = (
        RESULT_DIR / "selected_settings_val.json"
    )

    with open(
        settings_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            best_settings,
            f,
            indent=4
        )

    final_path = (
        RESULT_DIR / "postprocessed_comparison_val.csv"
    )

    final_table.to_csv(
        final_path,
        index=False
    )

    print("\n" + "=" * 100)
    print("FINAL VALIDATION COMPARISON")
    print("EVERY ROW USES POST-PROCESSED PREDICTIONS")
    print("=" * 100)

    columns = [
        "model",
        "threshold",
        "min_area",
        "precision",
        "recall",
        "f1",
        "iou",
        "empty_far",
        "mean_regions",
        "count_mae"
    ]

    print(
        final_table[columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\nSaved:", final_path)
    print("Saved:", settings_path)

    # --------------------------------------------------------
    # BOOTSTRAP
    # --------------------------------------------------------

    bootstrap_result = run_paired_bootstrap(
        per_crop_counts["BCE+Dice"],
        per_crop_counts["BCE-only"]
    )

    bootstrap_path = (
        RESULT_DIR / "bootstrap_f1_val.json"
    )

    with open(
        bootstrap_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            bootstrap_result,
            f,
            indent=4
        )

    print("\nSaved:", bootstrap_path)

    print("\nSTEP 7A COMPLETE.")
    print("TEST WAS NEVER LOADED OR EVALUATED.")


if __name__ == "__main__":
    main()
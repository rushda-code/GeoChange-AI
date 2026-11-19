# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 13:15:06 2026

@author: Admin
"""

# ============================================================
# GeoChange-AI — STEP 7C
# Image Misalignment Robustness
#
# VALIDATION ONLY.
# TEST IS NEVER LOADED OR ACCESSED.
#
# Run:
# python -m scripts.misalignment
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
import matplotlib.pyplot as plt

# Register custom model layer.
from src.model import AbsDiff


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = ROOT / "configs" / "default.yaml"

RESULT_DIR = ROOT / "experiments" / "results"

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

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

SHIFTS = [0, 1, 3, 5]

BATCH_SIZE = 16

IMAGE_SIZE = 256

EXPECTED_COUNT = 1024


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
            f"Expected one validation Parquet. Found: {matches}"
        )

    path = Path(matches[0])

    if "test" in path.name.lower():
        raise RuntimeError("TEST access is prohibited.")

    return path


def iter_val_rows():

    parquet = pq.ParquetFile(
        get_val_parquet()
    )

    count = 0

    for batch in parquet.iter_batches(
        batch_size=32,
        columns=["imageA", "imageB", "label"]
    ):

        for row in batch.to_pylist():

            count += 1

            yield (
                row["imageA"]["bytes"],
                row["imageB"]["bytes"],
                row["label"]["bytes"]
            )

    if count != EXPECTED_COUNT:

        raise AssertionError(
            f"Expected {EXPECTED_COUNT} validation crops, got {count}"
        )


# ============================================================
# 3. DECODE IMAGES
# ============================================================

def decode_raw(bytes_a, bytes_b, bytes_mask):

    image_a = tf.io.decode_jpeg(
        bytes_a,
        channels=3,
        dct_method="INTEGER_ACCURATE"
    ).numpy()

    image_b = tf.io.decode_jpeg(
        bytes_b,
        channels=3,
        dct_method="INTEGER_ACCURATE"
    ).numpy()

    mask = tf.io.decode_png(
        bytes_mask,
        channels=1
    ).numpy()

    mask = (mask[..., 0] > 0).astype(
        np.uint8
    )

    return image_a, image_b, mask


def preprocess_batch(images):

    images = tf.convert_to_tensor(
        images,
        dtype=tf.float32
    )

    return tf.keras.applications.mobilenet_v2.preprocess_input(
        images
    ).numpy()


# ============================================================
# 4. SHIFT IMAGE B
# ============================================================

def shift_image_edge(image, k, direction):

    """
    Shift an image using edge replication.

    No wrap-around.

    Horizontal:
        k pixels right.

    Diagonal:
        k pixels right AND k pixels down.
    """

    if k == 0:
        return image.copy()

    if direction == "horizontal":

        padded = np.pad(
            image,
            (
                (0, 0),
                (k, 0),
                (0, 0)
            ),
            mode="edge"
        )

        shifted = padded[:, :IMAGE_SIZE, :]

    elif direction == "diagonal":

        padded = np.pad(
            image,
            (
                (k, 0),
                (k, 0),
                (0, 0)
            ),
            mode="edge"
        )

        shifted = padded[
            :IMAGE_SIZE,
            :IMAGE_SIZE,
            :
        ]

    else:

        raise ValueError(
            f"Unknown shift direction: {direction}"
        )

    return shifted


# ============================================================
# 5. COMPONENT CLEANING
# ============================================================

def clean_prediction(
    probability,
    threshold,
    min_area
):

    binary = (
        probability > threshold
    ).astype(np.uint8)

    n_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )

    output = np.zeros_like(
        binary,
        dtype=np.uint8
    )

    regions = 0

    for label_id in range(1, n_labels):

        area = stats[
            label_id,
            cv2.CC_STAT_AREA
        ]

        if area >= min_area:

            output[
                labels == label_id
            ] = 1

            regions += 1

    return output, regions


# ============================================================
# 6. METRICS
# ============================================================

def confusion(gt, pred):

    gt = gt.astype(bool)
    pred = pred.astype(bool)

    tp = int(np.sum(gt & pred))
    fp = int(np.sum(~gt & pred))
    fn = int(np.sum(gt & ~pred))
    tn = int(np.sum(~gt & ~pred))

    return tp, fp, fn, tn


def calculate_metrics(
    total_tp,
    total_fp,
    total_fn,
    empty_crops,
    false_alarm_crops,
    total_regions,
    total_crops
):

    precision = (
        total_tp / (total_tp + total_fp)
        if total_tp + total_fp > 0
        else 0.0
    )

    recall = (
        total_tp / (total_tp + total_fn)
        if total_tp + total_fn > 0
        else 0.0
    )

    f1 = (
        2 * total_tp /
        (2 * total_tp + total_fp + total_fn)
        if 2 * total_tp + total_fp + total_fn > 0
        else 0.0
    )

    iou = (
        total_tp /
        (total_tp + total_fp + total_fn)
        if total_tp + total_fp + total_fn > 0
        else 0.0
    )

    empty_far = (
        false_alarm_crops / empty_crops
        if empty_crops > 0
        else 0.0
    )

    mean_regions = (
        total_regions / total_crops
        if total_crops > 0
        else 0.0
    )

    return {

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "iou": iou,

        "empty_far": empty_far,

        "mean_regions": mean_regions

    }


# ============================================================
# 7. LOAD SELECTED SETTINGS
# ============================================================

def load_selected_settings():

    path = (
        RESULT_DIR / "selected_settings_val.json"
    )

    if not path.exists():

        raise FileNotFoundError(
            "Run eval_val_both.py first. "
            "selected_settings_val.json is missing."
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        settings = json.load(f)

    print("\nSelected validation settings:")

    print(
        json.dumps(
            settings,
            indent=4
        )
    )

    return settings


# ============================================================
# 8. LOAD MODELS
# ============================================================

def load_models():

    models = {}

    for name, path in MODELS.items():

        if not path.exists():

            raise FileNotFoundError(path)

        print("\nLoading:", name)

        models[name] = keras.models.load_model(
            path,
            compile=False,
            custom_objects={
                "AbsDiff": AbsDiff,
                "GeoChange>AbsDiff": AbsDiff
            }
        )

    return models


# ============================================================
# 9. RUN MISALIGNMENT EXPERIMENT
# ============================================================

def run_experiment():

    print("\n" + "=" * 70)
    print("STEP 7C — MISALIGNMENT ROBUSTNESS")
    print("VALIDATION ONLY")
    print("TEST SPLIT ACCESS: DISABLED")
    print("=" * 70)

    settings = load_selected_settings()

    models = load_models()

    # Each model/direction/k gets separate pooled counts.
    accumulators = {}

    for model_name in MODELS:

        for direction in [
            "horizontal",
            "diagonal"
        ]:

            for k in SHIFTS:

                accumulators[
                    (model_name, direction, k)
                ] = {

                    "tp": 0,
                    "fp": 0,
                    "fn": 0,

                    "empty_crops": 0,
                    "false_alarm_crops": 0,

                    "regions": 0,
                    "crops": 0

                }

    # --------------------------------------------------------
    # PROCESS VALIDATION IN BATCHES
    # --------------------------------------------------------

    batch_a = []
    batch_b = []
    batch_gt = []

    processed = 0

    start = time.time()

    def process_batch():

        nonlocal processed

        if not batch_a:
            return

        raw_a = np.asarray(
            batch_a,
            dtype=np.float32
        )

        raw_b = np.asarray(
            batch_b,
            dtype=np.float32
        )

        gt_batch = np.asarray(
            batch_gt,
            dtype=np.uint8
        )

        # Preprocess A once.
        input_a = preprocess_batch(raw_a)

        # Every model is tested at the same shifts.
        for direction in [
            "horizontal",
            "diagonal"
        ]:

            for k in SHIFTS:

                shifted_raw_b = np.stack([

                    shift_image_edge(
                        image,
                        k,
                        direction
                    )

                    for image in raw_b

                ])

                shifted_b = preprocess_batch(
                    shifted_raw_b
                )

                for model_name, model in models.items():

                    threshold = float(
                        settings[model_name]["threshold"]
                    )

                    min_area = int(
                        settings[model_name]["min_area"]
                    )

                    probabilities = model.predict(
                        [input_a, shifted_b],
                        verbose=0
                    )

                    probabilities = np.asarray(
                        probabilities
                    )

                    if probabilities.ndim == 4:
                        probabilities = probabilities[..., 0]

                    key = (
                        model_name,
                        direction,
                        k
                    )

                    acc = accumulators[key]

                    for probability, gt in zip(
                        probabilities,
                        gt_batch
                    ):

                        pred, regions = clean_prediction(
                            probability,
                            threshold,
                            min_area
                        )

                        tp, fp, fn, tn = confusion(
                            gt,
                            pred
                        )

                        acc["tp"] += tp
                        acc["fp"] += fp
                        acc["fn"] += fn

                        acc["regions"] += regions
                        acc["crops"] += 1

                        if not np.any(gt):

                            acc["empty_crops"] += 1

                            if np.any(pred):

                                acc["false_alarm_crops"] += 1

        processed += len(batch_a)

        if processed % 128 == 0:

            print(
                f"Processed {processed}/{EXPECTED_COUNT} crops"
            )

        batch_a.clear()
        batch_b.clear()
        batch_gt.clear()

    # --------------------------------------------------------
    # STREAM VALIDATION ONLY
    # --------------------------------------------------------

    for bytes_a, bytes_b, bytes_mask in iter_val_rows():

        image_a, image_b, gt = decode_raw(
            bytes_a,
            bytes_b,
            bytes_mask
        )

        batch_a.append(image_a)
        batch_b.append(image_b)
        batch_gt.append(gt)

        if len(batch_a) == BATCH_SIZE:

            process_batch()

    process_batch()

    if processed != EXPECTED_COUNT:

        raise AssertionError(
            f"Expected {EXPECTED_COUNT} samples, got {processed}"
        )

    print(
        f"\nEvaluation time: {time.time() - start:.1f} seconds"
    )

    # --------------------------------------------------------
    # BUILD RESULTS TABLE
    # --------------------------------------------------------

    rows = []

    for key, acc in accumulators.items():

        model_name, direction, k = key

        scores = calculate_metrics(
            acc["tp"],
            acc["fp"],
            acc["fn"],
            acc["empty_crops"],
            acc["false_alarm_crops"],
            acc["regions"],
            acc["crops"]
        )

        rows.append({

            "model": model_name,

            "direction": direction,

            "shift_pixels": k,

            "threshold": settings[model_name]["threshold"],

            "min_area": settings[model_name]["min_area"],

            **scores

        })

    results = pd.DataFrame(rows)

    results = results.sort_values(
        [
            "model",
            "direction",
            "shift_pixels"
        ]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    csv_path = (
        RESULT_DIR / "misalignment_val.csv"
    )

    results.to_csv(
        csv_path,
        index=False
    )

    print("\nSaved:", csv_path)

    # --------------------------------------------------------
    # F1 LINE PLOT
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    styles = {
        ("BCE+Dice", "horizontal"): "-",
        ("BCE+Dice", "diagonal"): "--",
        ("BCE-only", "horizontal"): "-",
        ("BCE-only", "diagonal"): "--"
    }

    for model_name in MODELS:

        for direction in [
            "horizontal",
            "diagonal"
        ]:

            subset = results[
                (results["model"] == model_name)
                &
                (results["direction"] == direction)
            ].sort_values("shift_pixels")

            ax.plot(
                subset["shift_pixels"],
                subset["f1"],
                marker="o",
                linestyle=styles[
                    (model_name, direction)
                ],
                label=f"{model_name} — {direction}"
            )

    ax.set_title(
        "GeoChange-AI — Sensitivity to Image Misalignment"
    )

    ax.set_xlabel(
        "Image B shift (pixels)"
    )

    ax.set_ylabel(
        "Pooled F1"
    )

    ax.set_xticks(SHIFTS)

    ax.set_ylim(0, 1)

    ax.grid(
        True,
        alpha=0.3
    )

    ax.legend()

    fig.tight_layout()

    plot_path = (
        RESULT_DIR / "misalignment_f1_val.png"
    )

    fig.savefig(
        plot_path,
        dpi=160,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Saved:", plot_path)

    # --------------------------------------------------------
    # PRINT TABLE
    # --------------------------------------------------------

    print("\n" + "=" * 110)
    print("MISALIGNMENT RESULTS — VALIDATION")
    print("=" * 110)

    columns = [

        "model",

        "direction",

        "shift_pixels",

        "precision",

        "recall",

        "f1",

        "iou",

        "empty_far",

        "mean_regions"

    ]

    print(
        results[columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\nSTEP 7C COMPLETE.")
    print("TEST WAS NEVER LOADED OR EVALUATED.")

    return results


if __name__ == "__main__":

    run_experiment()
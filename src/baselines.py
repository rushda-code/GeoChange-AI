# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 13:45:52 2026

@author: Admin
"""

# src/baselines.py

import os
import cv2
import numpy as np
import pandas as pd

from src.data import (
    load_data,
    make_dataset,
    get_encoded_sample
)

from src.metrics import (
    confusion_counts,
    scores_from_counts,
    empty_crop_false_alarm_rate
)

from src.postprocess import clean_mask


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

SEED = 42

TRAIN_SAMPLES = 1000

MIN_AREA = 20

RESULT_DIR = "experiments/results"


# --------------------------------------------------
# 1. CONVERT PREPROCESSED IMAGE BACK TO RGB
# --------------------------------------------------

def undo_mobilenet_preprocessing(image):

    image = np.asarray(
        image,
        dtype=np.float32
    )

    image = (image + 1.0) * 127.5

    image = np.clip(
        image,
        0,
        255
    )

    return image.astype(np.uint8)


# --------------------------------------------------
# 2. DIFFERENCE IMAGE
# --------------------------------------------------

def difference_image(image_a, image_b):

    image_a = undo_mobilenet_preprocessing(
        image_a
    )

    image_b = undo_mobilenet_preprocessing(
        image_b
    )

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

    # Light smoothing reduces isolated noise.
    difference = cv2.GaussianBlur(
        difference,
        (3, 3),
        0
    )

    return difference


# --------------------------------------------------
# 3. PER-IMAGE OTSU
# --------------------------------------------------

def otsu_prediction(
    image_a,
    image_b,
    min_area=MIN_AREA
):

    difference = difference_image(
        image_a,
        image_b
    )

    threshold_value, binary = cv2.threshold(
        difference,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    cleaned, regions, area_pct = clean_mask(
        binary,
        threshold=0.5,
        min_area=min_area
    )

    return cleaned, regions, area_pct, threshold_value


# --------------------------------------------------
# 4. GLOBAL THRESHOLD SELECTION
# --------------------------------------------------

def select_global_threshold(
    n_samples=TRAIN_SAMPLES,
    seed=SEED
):
    """
    Select one threshold using a fixed random sample
    from the cleaned training indices.

    Uses pooled pixel F1.
    Does not access validation or test data.

    Histogram accumulation avoids storing all image arrays.
    """

    rng = np.random.default_rng(seed)

    train_count = 7095

    selected_positions = rng.choice(
        train_count,
        size=n_samples,
        replace=False
    )

    positive_hist = np.zeros(
        256,
        dtype=np.int64
    )

    negative_hist = np.zeros(
        256,
        dtype=np.int64
    )

    total_positive = 0

    for position in selected_positions:

        bytes_a, bytes_b, bytes_mask = (
            get_encoded_sample(
                "train",
                int(position)
            )
        )

        # Decode only this sample.
        image_a = cv2.imdecode(
            np.frombuffer(bytes_a, dtype=np.uint8),
            cv2.IMREAD_COLOR
        )

        image_b = cv2.imdecode(
            np.frombuffer(bytes_b, dtype=np.uint8),
            cv2.IMREAD_COLOR
        )

        gt = cv2.imdecode(
            np.frombuffer(bytes_mask, dtype=np.uint8),
            cv2.IMREAD_GRAYSCALE
        )

        if image_a is None or image_b is None or gt is None:
            raise ValueError("Failed to decode training sample.")

        # Convert BGR to RGB before using the shared difference function.
        image_a = cv2.cvtColor(
            image_a,
            cv2.COLOR_BGR2RGB
        )

        image_b = cv2.cvtColor(
            image_b,
            cv2.COLOR_BGR2RGB
        )

        difference = cv2.absdiff(
            cv2.cvtColor(image_a, cv2.COLOR_RGB2GRAY),
            cv2.cvtColor(image_b, cv2.COLOR_RGB2GRAY)
        )

        difference = cv2.GaussianBlur(
            difference,
            (3, 3),
            0
        )

        gt = gt > 0

        total_positive += int(np.sum(gt))

        positive_hist += np.bincount(
            difference[gt],
            minlength=256
        )

        negative_hist += np.bincount(
            difference[~gt],
            minlength=256
        )

    # Predicted positive means difference > threshold.
    positive_reverse = np.cumsum(
        positive_hist[::-1]
    )[::-1]

    negative_reverse = np.cumsum(
        negative_hist[::-1]
    )[::-1]

    best_threshold = 0
    best_f1 = -1.0

    for threshold in range(256):

        tp = int(
            positive_reverse[threshold + 1]
        ) if threshold < 255 else 0

        fp = int(
            negative_reverse[threshold + 1]
        ) if threshold < 255 else 0

        fn = total_positive - tp

        denominator = 2 * tp + fp + fn

        f1 = (
            2 * tp / denominator
            if denominator > 0
            else 0.0
        )

        if f1 > best_f1:

            best_f1 = f1
            best_threshold = threshold

    print("\nGLOBAL THRESHOLD SELECTION")
    print("Training samples:", n_samples)
    print("Fixed seed:", seed)
    print("Selected threshold:", best_threshold)
    print("Training pooled F1:", best_f1)

    return best_threshold


# --------------------------------------------------
# 5. GLOBAL THRESHOLD PREDICTION
# --------------------------------------------------

def global_prediction(
    image_a,
    image_b,
    threshold,
    min_area=MIN_AREA
):

    difference = difference_image(
        image_a,
        image_b
    )

    _, binary = cv2.threshold(
        difference,
        threshold,
        255,
        cv2.THRESH_BINARY
    )

    cleaned, regions, area_pct = clean_mask(
        binary,
        threshold=0.5,
        min_area=min_area
    )

    return cleaned, regions, area_pct


# --------------------------------------------------
# 6. EVALUATE ONE BASELINE ON VALIDATION
# --------------------------------------------------

def evaluate_baseline(
    name,
    val_ds,
    method,
    threshold=None
):

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    empty_crops = 0
    empty_false_alarms = 0

    total_regions = 0
    total_crops = 0

    for (batch_a, batch_b), batch_gt in val_ds:

        batch_a = batch_a.numpy()
        batch_b = batch_b.numpy()
        batch_gt = batch_gt.numpy()

        for i in range(len(batch_gt)):

            gt = (
                batch_gt[i, ..., 0] > 0
            ).astype(np.uint8)

            if method == "zero":

                pred = np.zeros_like(
                    gt,
                    dtype=np.uint8
                )

                regions = 0

            elif method == "otsu":

                pred, regions, area_pct, _ = (
                    otsu_prediction(
                        batch_a[i],
                        batch_b[i]
                    )
                )

            elif method == "global":

                pred, regions, area_pct = (
                    global_prediction(
                        batch_a[i],
                        batch_b[i],
                        threshold
                    )
                )

            else:

                raise ValueError(
                    f"Unknown baseline: {method}"
                )

            tp, fp, fn, tn = confusion_counts(
                gt,
                pred
            )

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn

            total_regions += regions
            total_crops += 1

            if np.all(gt == 0):

                empty_crops += 1

                if np.any(pred > 0):
                    empty_false_alarms += 1

    scores = scores_from_counts(
        total_tp,
        total_fp,
        total_fn,
        total_tn
    )

    false_alarm_rate = (
        empty_false_alarms / empty_crops
        if empty_crops > 0
        else 0.0
    )

    mean_regions = (
        total_regions / total_crops
        if total_crops > 0
        else 0.0
    )

    return {
        "baseline": name,
        "threshold": (
            threshold if threshold is not None else "per_image"
        ),
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "TN": total_tn,
        "precision": scores["precision"],
        "recall": scores["recall"],
        "f1": scores["f1"],
        "iou": scores["iou"],
        "pixel_accuracy": scores["pixel_accuracy"],
        "undefined": ",".join(scores["undefined"]),
        "empty_crop_false_alarm_rate": false_alarm_rate,
        "mean_regions_per_crop": mean_regions
    }


# --------------------------------------------------
# 7. RUN ALL BASELINES ON VALIDATION
# --------------------------------------------------

def run_baselines():

    # Load the already-cleaned encoded dataset.
    load_data()

    # Only validation is used for baseline evaluation.
    val_ds = make_dataset(
        "val",
        batch_size=16,
        training=False,
        seed=SEED
    )

    print("\nRunning all-zero baseline...")

    zero_result = evaluate_baseline(
        name="All-zero",
        val_ds=val_ds,
        method="zero"
    )

    print("\nRunning per-image Otsu baseline...")

    otsu_result = evaluate_baseline(
        name="Per-image Otsu",
        val_ds=val_ds,
        method="otsu"
    )

    print("\nSelecting global threshold from training...")

    selected_threshold = select_global_threshold(
        n_samples=1000,
        seed=SEED
    )

    print("\nRunning global-threshold baseline...")

    global_result = evaluate_baseline(
        name="Global threshold",
        val_ds=val_ds,
        method="global",
        threshold=selected_threshold
    )

    results = pd.DataFrame([
        zero_result,
        otsu_result,
        global_result
    ])

    os.makedirs(
        RESULT_DIR,
        exist_ok=True
    )

    output_path = os.path.join(
        RESULT_DIR,
        "baselines_val.csv"
    )

    results.to_csv(
        output_path,
        index=False
    )

    print("\n========== VALIDATION BASELINES ==========")

    print(
        results[
            [
                "baseline",
                "threshold",
                "precision",
                "recall",
                "f1",
                "iou",
                "pixel_accuracy",
                "empty_crop_false_alarm_rate",
                "mean_regions_per_crop"
            ]
        ].to_string(index=False)
    )

    print("\nSaved:", output_path)

    return results


if __name__ == "__main__":

    run_baselines()
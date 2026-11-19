# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 13:44:29 2026

@author: Admin
"""

# src/metrics.py

import numpy as np
import pandas as pd
import tensorflow as tf
import keras


# --------------------------------------------------
# 1. CONFUSION MATRIX COUNTS
# --------------------------------------------------

def confusion_counts(gt, pred_binary):
    """
    Calculate pixel-level TP, FP, FN and TN.

    gt: Ground-truth binary mask.
    pred_binary: Predicted binary mask.
    """

    gt = np.asarray(gt).astype(bool)
    pred = np.asarray(pred_binary).astype(bool)

    if gt.shape != pred.shape:
        raise ValueError("Ground truth and prediction shapes differ.")

    tp = int(np.sum(gt & pred))
    fp = int(np.sum(~gt & pred))
    fn = int(np.sum(gt & ~pred))
    tn = int(np.sum(~gt & ~pred))

    return tp, fp, fn, tn


# --------------------------------------------------
# 2. METRICS FROM POOLED COUNTS
# --------------------------------------------------

def scores_from_counts(TP, FP, FN, TN):

    TP = int(TP)
    FP = int(FP)
    FN = int(FN)
    TN = int(TN)

    undefined = []

    def safe_divide(numerator, denominator, name):

        if denominator == 0:
            undefined.append(name)
            return 0.0

        return float(numerator / denominator)

    precision = safe_divide(
        TP, TP + FP, "precision"
    )

    recall = safe_divide(
        TP, TP + FN, "recall"
    )

    f1 = safe_divide(
        2 * TP,
        2 * TP + FP + FN,
        "f1"
    )

    iou = safe_divide(
        TP,
        TP + FP + FN,
        "iou"
    )

    pixel_accuracy = safe_divide(
        TP + TN,
        TP + FP + FN + TN,
        "pixel_accuracy"
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "pixel_accuracy": pixel_accuracy,
        "undefined": undefined
    }


# --------------------------------------------------
# 3. THRESHOLD SWEEP
# --------------------------------------------------

def sweep_thresholds(
    model,
    ds,
    thresholds=np.arange(0.05, 0.96, 0.05)
):
    """
    Evaluate multiple probability thresholds.

    Processes the dataset once.
    Does not store all predictions.
    """

    thresholds = np.asarray(thresholds)

    counts = {
        float(t): [0, 0, 0, 0]
        for t in thresholds
    }

    for (image_a, image_b), masks in ds:

        probabilities = model(
            (image_a, image_b),
            training=False
        )

        probabilities = probabilities.numpy()
        masks = masks.numpy()

        for threshold in thresholds:

            pred = probabilities >= threshold

            tp, fp, fn, tn = confusion_counts(
                masks,
                pred
            )

            current = counts[float(threshold)]

            current[0] += tp
            current[1] += fp
            current[2] += fn
            current[3] += tn

    rows = []

    for threshold, values in counts.items():

        tp, fp, fn, tn = values

        scores = scores_from_counts(
            tp, fp, fn, tn
        )

        rows.append({
            "threshold": threshold,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            **scores
        })

    return pd.DataFrame(rows)


# --------------------------------------------------
# 4. STREAMING KERAS F1
# --------------------------------------------------

@keras.saving.register_keras_serializable(
    package="GeoChangeAI"
)
class ChangeF1(keras.metrics.Metric):

    def __init__(
        self,
        threshold=0.5,
        name="change_f1",
        **kwargs
    ):

        super().__init__(name=name, **kwargs)

        self.threshold = threshold

        self.tp = self.add_weight(
            name="tp",
            initializer="zeros"
        )

        self.fp = self.add_weight(
            name="fp",
            initializer="zeros"
        )

        self.fn = self.add_weight(
            name="fn",
            initializer="zeros"
        )

    def update_state(
        self,
        y_true,
        y_pred,
        sample_weight=None
    ):

        y_true = tf.cast(
            y_true > 0.5,
            tf.float32
        )

        y_pred = tf.cast(
            y_pred >= self.threshold,
            tf.float32
        )

        tp = y_true * y_pred
        fp = (1.0 - y_true) * y_pred
        fn = y_true * (1.0 - y_pred)

        if sample_weight is not None:

            weights = tf.cast(
                sample_weight,
                tf.float32
            )

            tp *= weights
            fp *= weights
            fn *= weights

        self.tp.assign_add(tf.reduce_sum(tp))
        self.fp.assign_add(tf.reduce_sum(fp))
        self.fn.assign_add(tf.reduce_sum(fn))

    def result(self):

        denominator = (
            2.0 * self.tp +
            self.fp +
            self.fn
        )

        return tf.math.divide_no_nan(
            2.0 * self.tp,
            denominator
        )

    def reset_state(self):

        self.tp.assign(0.0)
        self.fp.assign(0.0)
        self.fn.assign(0.0)

    def get_config(self):

        config = super().get_config()

        config.update({
            "threshold": self.threshold
        })

        return config


# --------------------------------------------------
# 5. STREAMING KERAS IOU
# --------------------------------------------------

@keras.saving.register_keras_serializable(
    package="GeoChangeAI"
)
class ChangeIoU(keras.metrics.Metric):

    def __init__(
        self,
        threshold=0.5,
        name="change_iou",
        **kwargs
    ):

        super().__init__(name=name, **kwargs)

        self.threshold = threshold

        self.tp = self.add_weight(
            name="tp",
            initializer="zeros"
        )

        self.fp = self.add_weight(
            name="fp",
            initializer="zeros"
        )

        self.fn = self.add_weight(
            name="fn",
            initializer="zeros"
        )

    def update_state(
        self,
        y_true,
        y_pred,
        sample_weight=None
    ):

        y_true = tf.cast(
            y_true > 0.5,
            tf.float32
        )

        y_pred = tf.cast(
            y_pred >= self.threshold,
            tf.float32
        )

        tp = y_true * y_pred
        fp = (1.0 - y_true) * y_pred
        fn = y_true * (1.0 - y_pred)

        if sample_weight is not None:

            weights = tf.cast(
                sample_weight,
                tf.float32
            )

            tp *= weights
            fp *= weights
            fn *= weights

        self.tp.assign_add(tf.reduce_sum(tp))
        self.fp.assign_add(tf.reduce_sum(fp))
        self.fn.assign_add(tf.reduce_sum(fn))

    def result(self):

        denominator = (
            self.tp +
            self.fp +
            self.fn
        )

        return tf.math.divide_no_nan(
            self.tp,
            denominator
        )

    def reset_state(self):

        self.tp.assign(0.0)
        self.fp.assign(0.0)
        self.fn.assign(0.0)

    def get_config(self):

        config = super().get_config()

        config.update({
            "threshold": self.threshold
        })

        return config


# --------------------------------------------------
# 6. TRAINING METRICS
# --------------------------------------------------

def get_training_metrics():

    return [
        ChangeF1(threshold=0.5),
        ChangeIoU(threshold=0.5),

        keras.metrics.Precision(
            thresholds=0.5,
            name="precision"
        ),

        keras.metrics.Recall(
            thresholds=0.5,
            name="recall"
        )
    ]


# --------------------------------------------------
# 7. EMPTY CROP FALSE ALARM RATE
# --------------------------------------------------

def empty_crop_false_alarm_rate(
    pred_masks_after_postprocess,
    gt_masks
):

    predictions = np.asarray(
        pred_masks_after_postprocess
    )

    ground_truth = np.asarray(gt_masks)

    if predictions.shape != ground_truth.shape:
        raise ValueError("Prediction and GT shapes differ.")

    if predictions.ndim < 3:
        raise ValueError("Expected batched masks.")

    predictions = predictions.reshape(
        predictions.shape[0], -1
    )

    ground_truth = ground_truth.reshape(
        ground_truth.shape[0], -1
    )

    empty_indices = np.all(
        ground_truth == 0,
        axis=1
    )

    empty_count = int(np.sum(empty_indices))

    if empty_count == 0:
        return 0.0

    false_alarms = np.any(
        predictions[empty_indices] > 0,
        axis=1
    )

    return float(np.mean(false_alarms))
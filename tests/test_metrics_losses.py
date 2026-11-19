# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 13:46:15 2026

@author: Admin
"""

# tests/test_metrics_losses.py

import numpy as np
import tensorflow as tf

from src.metrics import (
    confusion_counts,
    scores_from_counts,
    ChangeF1
)

from src.losses import BceDiceLoss


# --------------------------------------------------
# TEST 1: PERFECT PREDICTION
# --------------------------------------------------

def test_perfect_prediction():

    gt = np.array([
        [0, 1],
        [1, 0]
    ])

    pred = gt.copy()

    tp, fp, fn, tn = confusion_counts(
        gt, pred
    )

    scores = scores_from_counts(
        tp, fp, fn, tn
    )

    assert scores["f1"] == 1.0
    assert scores["iou"] == 1.0
    assert scores["precision"] == 1.0
    assert scores["recall"] == 1.0
    assert scores["pixel_accuracy"] == 1.0


# --------------------------------------------------
# TEST 2: ALL-ZERO PREDICTION
# --------------------------------------------------

def test_all_zero_prediction():

    gt = np.array([
        [0, 1],
        [1, 0]
    ])

    pred = np.zeros_like(gt)

    tp, fp, fn, tn = confusion_counts(
        gt, pred
    )

    scores = scores_from_counts(
        tp, fp, fn, tn
    )

    assert scores["precision"] == 0.0
    assert scores["recall"] == 0.0
    assert scores["f1"] == 0.0
    assert scores["iou"] == 0.0

    # Two unchanged pixels out of four.
    assert scores["pixel_accuracy"] == 0.5

    assert "precision" in scores["undefined"]


# --------------------------------------------------
# TEST 3: BCE + DICE LOSS
# --------------------------------------------------

def test_bce_dice_perfect_prediction():

    loss_fn = BceDiceLoss()

    gt = tf.ones((2, 16, 16, 1))

    pred = tf.ones((2, 16, 16, 1))

    loss = loss_fn(gt, pred)

    assert np.isfinite(loss.numpy())

    assert float(loss.numpy()) < 1e-5


def test_bce_dice_empty_ground_truth():

    loss_fn = BceDiceLoss()

    gt = tf.zeros((2, 16, 16, 1))

    pred = tf.fill(
        (2, 16, 16, 1),
        0.5
    )

    loss = loss_fn(gt, pred)

    assert np.isfinite(loss.numpy())

    assert float(loss.numpy()) > 0.7


# --------------------------------------------------
# TEST 4: CUSTOM F1 MATCHES POOLED F1
# --------------------------------------------------

def test_custom_f1_matches_scores_from_counts():

    rng = np.random.default_rng(42)

    gt = rng.integers(
        0, 2,
        size=(4, 32, 32, 1)
    ).astype(np.float32)

    pred = rng.random(
        size=(4, 32, 32, 1)
    ).astype(np.float32)

    pred_binary = pred >= 0.5

    tp, fp, fn, tn = confusion_counts(
        gt, pred_binary
    )

    expected = scores_from_counts(
        tp, fp, fn, tn
    )["f1"]

    metric = ChangeF1(
        threshold=0.5
    )

    metric.update_state(
        gt,
        pred
    )

    actual = float(
        metric.result().numpy()
    )

    assert np.isclose(
        actual,
        expected,
        atol=1e-7
    )


# --------------------------------------------------
# TEST 5: RESET STATE
# --------------------------------------------------

def test_change_f1_reset():

    metric = ChangeF1()

    gt = tf.ones((1, 4, 4, 1))
    pred = tf.ones((1, 4, 4, 1))

    metric.update_state(gt, pred)

    assert float(metric.result().numpy()) == 1.0

    metric.reset_state()

    assert float(metric.result().numpy()) == 0.0
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 13:44:54 2026

@author: Admin
"""

# src/losses.py

import tensorflow as tf
import keras


@keras.saving.register_keras_serializable(
    package="GeoChangeAI"
)
class BceDiceLoss(keras.losses.Loss):

    def __init__(
        self,
        bce_weight=0.5,
        dice_weight=0.5,
        smooth=1.0,
        name="bce_dice_loss",
        **kwargs
    ):

        super().__init__(
            name=name,
            **kwargs
        )

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def call(self, y_true, y_pred):

        # Use float32 for stable calculations.
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        eps = tf.keras.backend.epsilon()

        y_pred = tf.clip_by_value(
            y_pred,
            eps,
            1.0 - eps
        )

        # ---------------------------------
        # Binary cross-entropy
        # ---------------------------------

        bce = keras.losses.binary_crossentropy(
            y_true,
            y_pred
        )

        bce = tf.reduce_mean(bce)

        # ---------------------------------
        # Batch-level soft Dice
        # ---------------------------------

        intersection = tf.reduce_sum(
            y_true * y_pred
        )

        denominator = (
            tf.reduce_sum(y_true) +
            tf.reduce_sum(y_pred)
        )

        dice_loss = 1.0 - (
            (2.0 * intersection + self.smooth) /
            (denominator + self.smooth)
        )

        # ---------------------------------
        # Combined loss
        # ---------------------------------

        total_loss = (
            self.bce_weight * bce +
            self.dice_weight * dice_loss
        )

        return total_loss

    def get_config(self):

        config = super().get_config()

        config.update({
            "bce_weight": self.bce_weight,
            "dice_weight": self.dice_weight,
            "smooth": self.smooth
        })

        return config


# --------------------------------------------------
# LOSS SELECTION
# --------------------------------------------------

def get_loss(loss_name="bce_dice"):

    if loss_name == "bce_dice":

        return BceDiceLoss(
            bce_weight=0.5,
            dice_weight=0.5,
            smooth=1.0
        )

    elif loss_name == "bce":

        return keras.losses.BinaryCrossentropy(
            name="binary_crossentropy"
        )

    else:

        raise ValueError(
            f"Unknown loss function: {loss_name}"
        )
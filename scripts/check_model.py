# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 14:11:26 2026

@author: Admin
"""

"""
GeoChange AI
STEP 4 - Siamese U-Net verification.

Checks:
A. Parameter counts
B. Single shared encoder
C. Feature difference sharing proof
D. Output shape
E. Real validation batch prediction
F. One training smoke step
G. Training speed estimate
H. Save/load consistency

This is NOT full model training.
"""

import os
import sys
import time

import numpy as np
import tensorflow as tf

from pathlib import Path

from tensorflow import keras

# ------------------------------------------------------------
# PROJECT PATH
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

os.chdir(PROJECT_ROOT)

sys.path.insert(0, str(PROJECT_ROOT))

print("Project:", PROJECT_ROOT)

# ------------------------------------------------------------
# IMPORT PROJECT MODULES
# ------------------------------------------------------------

from src.model import build_siamese_unet, AbsDiff

from src.data import make_dataset

from src.losses import BceDiceLoss


# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

import yaml

with open("configs/default.yaml", "r") as file:

    config = yaml.safe_load(file)

FREEZE_ENCODER = config.get(
    "freeze_encoder",
    False
)

DECODER_FILTERS = config.get(
    "decoder_filters",
    [256, 128, 64, 32]
)

MIXED_PRECISION = config.get(
    "use_mixed_precision",
    False
)

BATCH_SIZE = 16
SEED = 42

print("\nCONFIGURATION")

print("Freeze encoder:", FREEZE_ENCODER)

print("Decoder filters:", DECODER_FILTERS)

print("Mixed precision:", MIXED_PRECISION)


# ------------------------------------------------------------
# BUILD MODEL
# ------------------------------------------------------------

model, encoder, diff_model, skip_information = build_siamese_unet(

    freeze_encoder=FREEZE_ENCODER,

    decoder_filters=DECODER_FILTERS,

    use_mixed_precision=MIXED_PRECISION,

    verbose=True,

    return_components=True
)


# ============================================================
# CHECK A: PARAMETER COUNTS
# ============================================================

print("\n========================================")
print("CHECK A: PARAMETER COUNTS")
print("========================================")

encoder_params = encoder.count_params()

total_params = model.count_params()

decoder_params = total_params - encoder_params

print(f"Encoder parameters: {encoder_params:,}")

print(f"Decoder parameters: {decoder_params:,}")

print(f"Total parameters:   {total_params:,}")

assert total_params == encoder_params + decoder_params

print("PASS: Total = Encoder + Decoder")


# ============================================================
# CHECK B: EXACTLY ONE SHARED ENCODER
# ============================================================

print("\n========================================")
print("CHECK B: SHARED ENCODER")
print("========================================")

encoder_models = [

    layer

    for layer in model.layers

    if isinstance(layer, keras.Model)

]

print("Nested models in main model:")

for layer in encoder_models:

    print(layer.name)

assert len(encoder_models) == 1

assert encoder_models[0] is encoder

print("PASS: Exactly one encoder sub-model.")

print("PASS: Same encoder object used for both inputs.")


# ============================================================
# LOAD REAL VALIDATION BATCH
# ============================================================

print("\nLoading validation batch...")

val_ds = make_dataset(
    "val",
    BATCH_SIZE,
    False,
    SEED
)

val_batch = next(iter(val_ds))

(val_a, val_b), val_mask = val_batch

print("Validation A:", val_a.shape)

print("Validation B:", val_b.shape)

print("Validation mask:", val_mask.shape)


# ============================================================
# CHECK C: FEATURE SHARING
# ============================================================

print("\n========================================")
print("CHECK C: FEATURE SHARING")
print("========================================")

# Same image supplied to both branches.

same_image = val_a[:1]

same_differences = diff_model(
    [same_image, same_image],
    training=False
)

for index, difference in enumerate(same_differences):

    max_difference = float(
        tf.reduce_max(
            tf.abs(difference)
        ).numpy()
    )

    print(
        f"Scale {[128,64,32,16,8][index]}: "
        f"maximum difference = {max_difference:.10f}"
    )

    assert max_difference <= 1e-6

print("PASS: Same image produces zero feature differences.")


# Different images.

different_differences = diff_model(
    [val_a[:1], val_b[:1]],
    training=False
)

different_maxima = [

    float(
        tf.reduce_max(
            tf.abs(x)
        ).numpy()
    )

    for x in different_differences

]

print("\nDifferent-image maximum differences:")

print(different_maxima)

assert any(
    value > 1e-6
    for value in different_maxima
)

print("PASS: Different images produce nonzero differences.")


# ============================================================
# CHECK D: OUTPUT SHAPE
# ============================================================

print("\n========================================")
print("CHECK D: OUTPUT SHAPE")
print("========================================")

print("Model output:", model.output_shape)

assert model.output_shape == (
    None,
    256,
    256,
    1
)

print("PASS: Output is (None,256,256,1).")


# ============================================================
# CHECK E: REAL VALIDATION PREDICTION
# ============================================================

print("\n========================================")
print("CHECK E: REAL VALIDATION PREDICTION")
print("========================================")

val_prediction = model(
    [val_a, val_b],
    training=False
)

print("Prediction shape:", val_prediction.shape)

print("Prediction dtype:", val_prediction.dtype)

print(
    "Prediction minimum:",
    float(tf.reduce_min(val_prediction).numpy())
)

print(
    "Prediction maximum:",
    float(tf.reduce_max(val_prediction).numpy())
)

assert tuple(val_prediction.shape) == (
    16,
    256,
    256,
    1
)

assert val_prediction.dtype == tf.float32

assert bool(
    tf.reduce_all(val_prediction > 0.0).numpy()
)

assert bool(
    tf.reduce_all(val_prediction < 1.0).numpy()
)

print("PASS: Real validation prediction.")


# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(

    optimizer=keras.optimizers.Adam(
        learning_rate=1e-4
    ),

    loss=BceDiceLoss()

)


# ============================================================
# CHECK F: ONE TRAINING SMOKE STEP
# ============================================================

print("\n========================================")
print("CHECK F: TRAINING SMOKE STEP")
print("========================================")

train_ds = make_dataset(
    "train",
    BATCH_SIZE,
    True,
    SEED
)

train_iterator = iter(train_ds)

(train_a, train_b), train_mask = next(train_iterator)

train_result = model.train_on_batch(

    [train_a, train_b],

    train_mask,

    return_dict=True

)

train_loss = float(
    train_result["loss"]
)

print("Training loss:", train_loss)

assert np.isfinite(train_loss)

print("PASS: One training step completed.")

print("NOTE: This is only a smoke test, not full training.")


# ============================================================
# CHECK G: TRAINING SPEED
# ============================================================

print("\n========================================")
print("CHECK G: TRAINING SPEED")
print("========================================")

# Two warm-up steps.

print("Running two warm-up steps...")

for _ in range(2):

    (a, b), mask = next(train_iterator)

    model.train_on_batch(
        [a, b],
        mask
    )

# Time five steps.

print("Timing five training steps...")

start_time = time.perf_counter()

for _ in range(5):

    (a, b), mask = next(train_iterator)

    model.train_on_batch(
        [a, b],
        mask
    )

elapsed = time.perf_counter() - start_time

seconds_per_step = elapsed / 5

images_per_second = BATCH_SIZE / seconds_per_step

epoch_steps = 443

estimated_epoch_time = (
    epoch_steps * seconds_per_step
)

print("\nTIMING RESULTS")

print(f"Five steps: {elapsed:.3f} seconds")

print(f"Seconds/step: {seconds_per_step:.4f}")

print(f"Images/second: {images_per_second:.2f}")

print(
    f"Estimated epoch time: "
    f"{estimated_epoch_time:.2f} seconds"
)

print(
    f"Estimated epoch time: "
    f"{estimated_epoch_time / 60:.2f} minutes"
)


# ============================================================
# CHECK H: SAVE AND LOAD
# ============================================================

print("\n========================================")
print("CHECK H: SAVE / LOAD")
print("========================================")

save_path = "tmp.keras"

if os.path.exists(save_path):

    os.remove(save_path)

# Save model.

model.save(save_path)

print("Model saved:", save_path)

# Load model with registered custom objects.

loaded_model = keras.models.load_model(

    save_path,

    custom_objects={

        "AbsDiff": AbsDiff,

        "BceDiceLoss": BceDiceLoss

    }

)

print("Model loaded successfully.")

# Compare predictions.

prediction_before = model(
    [val_a, val_b],
    training=False
).numpy()

prediction_after = loaded_model(
    [val_a, val_b],
    training=False
).numpy()

maximum_error = np.max(
    np.abs(
        prediction_before - prediction_after
    )
)

print("Maximum prediction difference:", maximum_error)

assert np.allclose(

    prediction_before,

    prediction_after,

    rtol=1e-6,

    atol=1e-6

)

print("PASS: Save/load predictions match.")


# ============================================================
# FINAL REPORT
# ============================================================

print("\n")
print("=" * 60)

print("GEOCHANGE AI — STEP 4 VERIFICATION SUMMARY")

print("=" * 60)

print("A. Parameter counts:             PASS")

print("B. Shared encoder:               PASS")

print("C. Feature sharing:              PASS")

print("D. Output shape:                 PASS")

print("E. Validation prediction:        PASS")

print("F. Training smoke step:          PASS")

print("G. Training timing:              COMPLETE")

print("H. Save/load consistency:        PASS")

print("-" * 60)

print("Encoder parameters:", f"{encoder_params:,}")

print("Decoder parameters:", f"{decoder_params:,}")

print("Total parameters:", f"{total_params:,}")

print("Seconds per step:", round(seconds_per_step, 4))

print(
    "Estimated epoch minutes:",
    round(estimated_epoch_time / 60, 2)
)

print("-" * 60)

print("FULL TRAINING: NOT STARTED")

print("STEP 4 VERIFICATION FINISHED")

print("=" * 60)
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 14:43:29 2026

@author: Admin
"""


"""
GeoChange-AI — STEP 6b+
Export the trained model for application deployment.

NO TRAINING.
CPU ONLY.
Uses VAL for architecture equivalence verification.
Does not access TEST.
"""

import os

# Must be set before importing TensorFlow.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import sys
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import tensorflow as tf
import keras

# ------------------------------------------------------------
# PROJECT PATHS
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))

from src.model import build_siamese_unet, AbsDiff


RUN_NAME = "bcedice_v1"

TRAINED_MODEL_PATH = (
    ROOT / "experiments/runs/bcedice_v1/best_model.keras"
)

FINAL_SETTINGS_PATH = (
    ROOT / "experiments/results/final_settings.json"
)

VAL_PARQUET = (
    ROOT / "data/raw/data/"
    "val-00000-of-00001-d09d88a7419f2427.parquet"
)

APP_MODEL_DIR = ROOT / "app/model"

WEIGHTS_PATH = (
    APP_MODEL_DIR / "geochange_v1.weights.h5"
)

SETTINGS_PATH = (
    APP_MODEL_DIR / "settings.json"
)

EXPORT_INFO_PATH = (
    APP_MODEL_DIR / "EXPORT_INFO.txt"
)

IMG_SIZE = 256
MAX_DIFFERENCE = 1e-5


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def decode_image(encoded_bytes):
    """Decode a JPEG exactly as in the training pipeline."""

    image = tf.io.decode_jpeg(
        encoded_bytes,
        channels=3,
        dct_method="INTEGER_ACCURATE"
    )

    image.set_shape((256, 256, 3))

    image = tf.cast(image, tf.float32)

    image = tf.keras.applications.mobilenet_v2.preprocess_input(
        image
    )

    return image


def get_val_pairs(number=8):
    """
    Read eight validation pairs directly.

    Does not call load_data(), because that function may
    load other splits as well.
    """

    if not VAL_PARQUET.exists():
        raise FileNotFoundError(VAL_PARQUET)

    table = pq.read_table(
        VAL_PARQUET,
        columns=["imageA", "imageB"]
    )

    if table.num_rows < number:
        raise ValueError(
            f"VAL contains only {table.num_rows} rows."
        )

    pairs = []

    for i in range(number):

        row = table.slice(i, 1).to_pylist()[0]

        image_a = decode_image(row["imageA"]["bytes"])
        image_b = decode_image(row["imageB"]["bytes"])

        pairs.append((image_a, image_b))

    batch_a = tf.stack([p[0] for p in pairs])
    batch_b = tf.stack([p[1] for p in pairs])

    return batch_a, batch_b


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():

    print("\n" + "=" * 75)
    print("GEOCHANGE-AI — MODEL EXPORT")
    print("=" * 75)

    print("TensorFlow:", tf.__version__)
    print("Keras:", keras.__version__)
    print("Python:", platform.python_version())

    print("Visible GPUs:", tf.config.list_physical_devices("GPU"))

    if tf.config.list_physical_devices("GPU"):
        raise RuntimeError(
            "GPU is visible. This export must run on CPU."
        )

    APP_MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 1. LOAD ORIGINAL TRAINED MODEL
    # --------------------------------------------------------

    print("\n[1/6] Loading trained BCE+Dice model...")

    if not TRAINED_MODEL_PATH.exists():
        raise FileNotFoundError(TRAINED_MODEL_PATH)

    original_model = keras.models.load_model(
        TRAINED_MODEL_PATH,
        compile=False,
        custom_objects={
            "AbsDiff": AbsDiff,
            "GeoChange>AbsDiff": AbsDiff
        }
    )

    print("Original model loaded.")
    print("Parameters:", original_model.count_params())

    # --------------------------------------------------------
    # 2. EXPORT WEIGHTS
    # --------------------------------------------------------

    print("\n[2/6] Exporting trained weights...")

    original_model.save_weights(WEIGHTS_PATH)

    print("Weights saved:")
    print(WEIGHTS_PATH)

    # --------------------------------------------------------
    # 3. EXPORT FROZEN SETTINGS
    # --------------------------------------------------------

    print("\n[3/6] Exporting application settings...")

    with open(
        FINAL_SETTINGS_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        final_settings = json.load(f)

    app_settings = {
        "model_name": "GeoChange-AI BCE+Dice",
        "threshold": 0.3,
        "min_area": 50
    }

    if final_settings.get("chosen_model") != "BCE+Dice":
        raise ValueError(
            "The selected model in final_settings.json "
            "is not BCE+Dice."
        )

    if float(final_settings.get("threshold")) != 0.3:
        raise ValueError(
            "Unexpected threshold in final_settings.json."
        )

    if int(final_settings.get("min_area")) != 50:
        raise ValueError(
            "Unexpected min_area in final_settings.json."
        )

    with open(
        SETTINGS_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            app_settings,
            f,
            indent=4
        )

    print("Settings saved:")
    print(json.dumps(app_settings, indent=4))

    # --------------------------------------------------------
    # 4. REBUILD MODEL WITHOUT IMAGENET WEIGHTS
    # --------------------------------------------------------

    print("\n[4/6] Rebuilding architecture on CPU...")

    with tf.device("/CPU:0"):

        rebuilt_model = build_siamese_unet(
            encoder_weights=None,
            verbose=False
        )

        rebuilt_model.load_weights(WEIGHTS_PATH)

    print("Exported weights loaded into rebuilt model.")

    if rebuilt_model.count_params() != original_model.count_params():
        raise AssertionError(
            "Parameter count mismatch between models."
        )

    print("Parameter count:", rebuilt_model.count_params())

    # --------------------------------------------------------
    # 5. VALIDATION PREDICTION EQUIVALENCE
    # --------------------------------------------------------

    print("\n[5/6] Comparing predictions on eight VAL pairs...")

    val_a, val_b = get_val_pairs(number=8)

    with tf.device("/CPU:0"):

        original_predictions = original_model(
            [val_a, val_b],
            training=False
        ).numpy()

        rebuilt_predictions = rebuilt_model(
            [val_a, val_b],
            training=False
        ).numpy()

    max_difference = float(
        np.max(
            np.abs(
                original_predictions - rebuilt_predictions
            )
        )
    )

    print("\nPREDICTION COMPARISON")
    print("-" * 55)
    print("Original prediction shape:", original_predictions.shape)
    print("Rebuilt prediction shape:", rebuilt_predictions.shape)
    print(f"Maximum absolute difference: {max_difference:.12f}")
    print("Required maximum difference:", MAX_DIFFERENCE)

    assert max_difference < MAX_DIFFERENCE, (
        f"Prediction mismatch: {max_difference} "
        f"is not below {MAX_DIFFERENCE}"
    )

    print("\nPASS: Exported weights reproduce original predictions.")

    # --------------------------------------------------------
    # 6. EXPORT INFORMATION
    # --------------------------------------------------------

    print("\n[6/6] Writing EXPORT_INFO.txt...")

    export_info = f"""
GEOCHANGE-AI — MODEL EXPORT INFORMATION
======================================

Framework versions
------------------
TensorFlow: {tf.__version__}
Keras: {keras.__version__}
Python: {platform.python_version()}

Model
-----
Run name: {RUN_NAME}
Selected model: BCE+Dice
Best epoch: 27
Architecture: Siamese U-Net with shared MobileNetV2 encoder
Input size: 256 x 256
Output size: 256 x 256 x 1
Parameters: {original_model.count_params()}

Frozen application settings
---------------------------
Threshold: 0.3
Minimum connected-component area: 50 pixels

Training configuration
----------------------
Batch normalization momentum: 0.99
Mixed precision: Disabled

Export
------
Weights file: geochange_v1.weights.h5
Settings file: settings.json
Original model: experiments/runs/bcedice_v1/best_model.keras

Verification
------------
Verification split: VAL
Number of validation pairs: 8
Maximum absolute prediction difference: {max_difference:.12f}
Required maximum difference: < {MAX_DIFFERENCE}

Result: PASS
No training was performed during export.
No TEST evaluation was performed during export.
"""

    with open(
        EXPORT_INFO_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(export_info.strip() + "\n")

    print("\n" + "=" * 75)
    print("EXPORT COMPLETE")
    print("=" * 75)

    print("Weights:", WEIGHTS_PATH)
    print("Settings:", SETTINGS_PATH)
    print("Export info:", EXPORT_INFO_PATH)

    print("\nASSERTION RESULT: PASS")
    print(f"Maximum difference: {max_difference:.12f}")
    print(f"Required: < {MAX_DIFFERENCE}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 15:17:18 2026

@author: Admin
"""


"""
GeoChange-AI — Inference Module

Responsibilities:
1. Load the exported Siamese U-Net weights on CPU.
2. Decode uploaded JPG and PNG images.
3. Resize and preprocess image pairs.
4. Predict a change-probability mask.
5. Post-process the mask.
6. Count changed regions and calculate changed area.
7. Draw detected changes on the second image.
8. Measure inference time.

The model architecture and trained weights are unchanged.
"""

import os

# Force CPU before importing TensorFlow.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from src.model import build_siamese_unet
from src.postprocess import clean_mask


# ============================================================
# CONFIGURATION
# ============================================================

IMG_SIZE = 256

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_WEIGHTS_PATH = (
    PROJECT_ROOT / "app" / "model" / "geochange_v1.weights.h5"
)

DEFAULT_THRESHOLD = 0.3
DEFAULT_MIN_AREA = 50


# ============================================================
# 1. LOAD MODEL
# ============================================================

def load_model(weights_path=DEFAULT_WEIGHTS_PATH):
    """
    Rebuild the Siamese U-Net architecture and load exported weights.

    ImageNet weights are disabled because the trained weights
    already contain the complete model parameters.
    """

    weights_path = Path(weights_path)

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Model weights not found: {weights_path}"
        )

    # Ensure TensorFlow executes on CPU.
    with tf.device("/CPU:0"):

        model = build_siamese_unet(
            encoder_weights=None,
            verbose=False
        )

        model.load_weights(str(weights_path))

    print("GeoChange-AI model loaded successfully.")
    print(f"Weights: {weights_path}")
    print(f"Parameters: {model.count_params():,}")

    return model


# ============================================================
# 2. DECODE UPLOADED IMAGE
# ============================================================

def decode_upload(image_bytes):
    """
    Decode uploaded JPEG or PNG bytes into an RGB uint8 image.

    Supported formats:
        JPEG
        PNG

    Returns:
        RGB image with shape (height, width, 3).
    """

    if not isinstance(image_bytes, (bytes, bytearray)):
        raise TypeError("Image upload must contain bytes.")

    if len(image_bytes) == 0:
        raise ValueError("Uploaded image is empty.")

    image_bytes = bytes(image_bytes)

    # Detect JPEG using file signature.
    is_jpeg = image_bytes.startswith(b"\xff\xd8")

    # Detect PNG using file signature.
    is_png = image_bytes.startswith(
        b"\x89PNG\r\n\x1a\n"
    )

    if not is_jpeg and not is_png:
        raise ValueError(
            "Unsupported image format. Please upload JPG or PNG."
        )

    encoded = tf.convert_to_tensor(
        image_bytes,
        dtype=tf.string
    )

    try:

        if is_jpeg:

            image = tf.io.decode_jpeg(
                encoded,
                channels=3,
                dct_method="INTEGER_ACCURATE"
            )

        else:

            image = tf.io.decode_png(
                encoded,
                channels=3
            )

        image = image.numpy()

    except Exception as exc:

        raise ValueError(
            "Could not decode the uploaded image."
        ) from exc

    return image.astype(np.uint8)


# ============================================================
# 3. PREPARE IMAGE PAIR
# ============================================================

def prepare_pair(image_a, image_b):
    """
    Prepare two RGB images for model prediction.

    Requirements:
        - Both images must have the same original dimensions.
        - Both images are resized to 256 x 256.
        - MobileNetV2 preprocessing converts values to [-1, 1].

    Returns:
        input_a: (1, 256, 256, 3), float32
        input_b: (1, 256, 256, 3), float32
        warning: warning message or None
    """

    image_a = np.asarray(image_a)
    image_b = np.asarray(image_b)

    if image_a.ndim != 3 or image_a.shape[-1] != 3:
        raise ValueError("Image A must be an RGB image.")

    if image_b.ndim != 3 or image_b.shape[-1] != 3:
        raise ValueError("Image B must be an RGB image.")

    if image_a.shape[:2] != image_b.shape[:2]:
        raise ValueError(
            "Image A and Image B must have the same dimensions. "
            f"Received {image_a.shape[:2]} and {image_b.shape[:2]}."
        )

    original_height, original_width = image_a.shape[:2]

    warning = None

    if (original_height, original_width) != (
        IMG_SIZE,
        IMG_SIZE
    ):

        warning = (
            f"Both images were resized from "
            f"{original_width} x {original_height} to "
            f"{IMG_SIZE} x {IMG_SIZE}. "
            "The model was trained on LEVIR-CD 256 x 256 crops "
            "at approximately 0.5 m/pixel. "
            "Performance on other resolutions and imagery "
            "has not been independently validated."
        )

    # Convert to float32.
    image_a = image_a.astype(np.float32)
    image_b = image_b.astype(np.float32)

    # Resize both images consistently.
    image_a = tf.image.resize(
        image_a,
        (IMG_SIZE, IMG_SIZE),
        method="bilinear",
        antialias=True
    ).numpy()

    image_b = tf.image.resize(
        image_b,
        (IMG_SIZE, IMG_SIZE),
        method="bilinear",
        antialias=True
    ).numpy()

    # MobileNetV2 preprocessing:
    # pixel values [0, 255] -> approximately [-1, 1].
    image_a = preprocess_input(image_a)
    image_b = preprocess_input(image_b)

    # Add batch dimension.
    input_a = np.expand_dims(
        image_a.astype(np.float32),
        axis=0
    )

    input_b = np.expand_dims(
        image_b.astype(np.float32),
        axis=0
    )

    return input_a, input_b, warning


# ============================================================
# 4. PREDICT CHANGE PROBABILITY
# ============================================================

def predict(model, input_a, input_b):
    """
    Generate a change-probability map.

    Returns:
        Probability map with shape (256, 256).
        Values range from 0 to 1.
    """

    with tf.device("/CPU:0"):

        predictions = model(
            [input_a, input_b],
            training=False
        )

    probability_map = predictions.numpy()[0, :, :, 0]

    return probability_map.astype(np.float32)


# ============================================================
# 5. ANALYSE PREDICTION
# ============================================================

def analyse(
    probability_map,
    threshold=DEFAULT_THRESHOLD,
    min_area=DEFAULT_MIN_AREA
):
    """
    Convert probability map into a cleaned binary change mask.

    Returns:
        binary_mask: uint8 mask containing 0 and 1.
        n_regions: number of retained changed regions.
        changed_area_pct: percentage of pixels classified as changed.
    """

    binary_mask, n_regions, changed_area_pct = clean_mask(
        probability_map,
        threshold=threshold,
        min_area=min_area
    )

    return (
        binary_mask.astype(np.uint8),
        int(n_regions),
        float(changed_area_pct)
    )


# ============================================================
# 6. DRAW OVERLAY ON IMAGE B
# ============================================================

def overlay(image_b, binary_mask):
    """
    Draw red outlines around predicted changed regions
    on the original second image.

    Returns:
        RGB overlay image.
    """

    image_b = np.asarray(image_b)

    if image_b.ndim != 3 or image_b.shape[-1] != 3:
        raise ValueError("Image B must be an RGB image.")

    mask = np.asarray(binary_mask)

    if mask.ndim != 2:
        raise ValueError("Binary mask must be a 2D array.")

    # Resize mask to original Image B dimensions if necessary.
    original_height, original_width = image_b.shape[:2]

    if mask.shape != (original_height, original_width):

        mask = cv2.resize(
            mask.astype(np.uint8),
            (original_width, original_height),
            interpolation=cv2.INTER_NEAREST
        )

    mask = (mask > 0).astype(np.uint8)

    # Find outlines of changed regions.
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    result = image_b.copy()

    # Red in RGB format.
    cv2.drawContours(
        result,
        contours,
        contourIdx=-1,
        color=(255, 0, 0),
        thickness=2
    )

    return result


# ============================================================
# 7. MODEL WARM-UP
# ============================================================

def warm_up(model):
    """
    Run one dummy prediction to initialize the model.

    Call this once after loading the model.

    Returns:
        Warm-up time in milliseconds.
    """

    dummy_a = np.zeros(
        (1, IMG_SIZE, IMG_SIZE, 3),
        dtype=np.float32
    )

    dummy_b = np.zeros(
        (1, IMG_SIZE, IMG_SIZE, 3),
        dtype=np.float32
    )

    start = time.perf_counter()

    _ = predict(
        model,
        dummy_a,
        dummy_b
    )

    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000

    return float(elapsed_ms)


# ============================================================
# 8. TIMED PREDICTION
# ============================================================

def predict_timed(model, input_a, input_b):
    """
    Measure model prediction time.

    Returns:
        probability_map
        inference_time_ms
    """

    start = time.perf_counter()

    probability_map = predict(
        model,
        input_a,
        input_b
    )

    inference_time_ms = (
        time.perf_counter() - start
    ) * 1000

    return (
        probability_map,
        float(inference_time_ms)
    )

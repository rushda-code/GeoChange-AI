# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 18:51:11 2026

@author: Admin
"""



import os

# Force CPU before importing TensorFlow or the inference module.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import sys
import json
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image


# ============================================================
# PATHS AND IMPORTS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.infer import (
    load_model,
    decode_upload,
    prepare_pair,
    predict_timed,
    analyse,
    overlay,
    warm_up,
)


APP_DIR = Path(__file__).resolve().parent

MODEL_DIR = APP_DIR / "model"

WEIGHTS_PATH = MODEL_DIR / "geochange_v1.weights.h5"

SETTINGS_PATH = MODEL_DIR / "settings.json"

DEMO_DIR = APP_DIR / "demo_pairs"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="GeoChange-AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# LOAD SETTINGS
# ============================================================

@st.cache_data
def load_settings():

    if not SETTINGS_PATH.exists():
        raise FileNotFoundError(
            f"Settings file not found: {SETTINGS_PATH}"
        )

    with open(SETTINGS_PATH, "r", encoding="utf-8") as file:
        settings = json.load(file)

    return settings


# ============================================================
# LOAD MODEL ONCE
# ============================================================

@st.cache_resource
def get_model():

    model = load_model(WEIGHTS_PATH)

    warmup_ms = warm_up(model)

    return model, warmup_ms


# ============================================================
# DEMO PAIR DISCOVERY
# ============================================================

def get_demo_pairs():

    if not DEMO_DIR.exists():
        return {}

    demo_pairs = {}

    for folder in sorted(DEMO_DIR.iterdir()):

        if not folder.is_dir():
            continue

        image_a = folder / "A.jpg"
        image_b = folder / "B.jpg"
        gt_mask = folder / "gt.png"

        if image_a.exists() and image_b.exists():

            demo_pairs[folder.name] = {
                "image_a": image_a,
                "image_b": image_b,
                "gt_mask": gt_mask if gt_mask.exists() else None
            }

    return demo_pairs


# ============================================================
# LOAD DEMO IMAGE
# ============================================================

def read_demo_image(path):

    try:

        with open(path, "rb") as file:
            image_bytes = file.read()

        return decode_upload(image_bytes)

    except Exception as exc:

        raise ValueError(
            f"Could not read demo image: {path.name}"
        ) from exc


# ============================================================
# LOAD GROUND-TRUTH MASK
# ============================================================

def read_ground_truth(path):

    if path is None or not path.exists():
        return None

    try:

        gt_image = Image.open(path).convert("L")

        gt_array = np.asarray(gt_image)

        # Original masks use non-zero values for changed pixels.
        gt_binary = (gt_array > 0).astype(np.uint8)

        return gt_binary

    except Exception as exc:

        raise ValueError(
            "Could not read the demo ground-truth mask."
        ) from exc


# ============================================================
# GROUND-TRUTH F1
# ============================================================

def calculate_pixel_f1(prediction, ground_truth):

    prediction = (prediction > 0).astype(np.uint8)
    ground_truth = (ground_truth > 0).astype(np.uint8)

    if prediction.shape != ground_truth.shape:

        ground_truth = cv2.resize(
            ground_truth,
            (prediction.shape[1], prediction.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    tp = int(np.sum((prediction == 1) & (ground_truth == 1)))
    fp = int(np.sum((prediction == 1) & (ground_truth == 0)))
    fn = int(np.sum((prediction == 0) & (ground_truth == 1)))

    denominator = (2 * tp) + fp + fn

    if denominator == 0:
        return 1.0

    return float((2 * tp) / denominator)


# ============================================================
# PROBABILITY HEATMAP
# ============================================================

def make_heatmap(probability_map):

    probability_uint8 = np.clip(
        probability_map * 255,
        0,
        255
    ).astype(np.uint8)

    heatmap = cv2.applyColorMap(
        probability_uint8,
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    return heatmap


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    st.title("🛰️ GeoChange-AI")

    st.subheader(
        "Satellite Infrastructure Change Detection"
    )

    st.write(
        "Compare satellite images captured at two different "
        "times to identify areas where infrastructure has changed."
    )

    st.info(
        "GeoChange-AI is an independent academic portfolio project. "
        "The model was trained on the LEVIR-CD building-change "
        "dataset, using 256 × 256 image crops at approximately "
        "0.5 m/pixel. Performance on other imagery, sensors, "
        "resolutions and geographic regions has not been "
        "independently validated."
    )

    st.divider()

    # --------------------------------------------------------
    # LOAD SETTINGS AND MODEL
    # --------------------------------------------------------

    try:

        settings = load_settings()

        model, warmup_ms = get_model()

    except Exception as exc:

        st.error(
            f"Could not initialize GeoChange-AI: {exc}"
        )

        st.stop()

    default_threshold = float(
        settings.get("threshold", 0.3)
    )

    default_min_area = int(
        settings.get("min_area", 50)
    )

    # --------------------------------------------------------
    # SIDEBAR
    # --------------------------------------------------------

    st.sidebar.header("Detection settings")

    st.sidebar.caption(
        "Default values were selected using the validation set."
    )

    threshold = st.sidebar.slider(
        "Detection threshold",
        min_value=0.05,
        max_value=0.95,
        value=default_threshold,
        step=0.05,
        help=(
            "A pixel is initially classified as changed "
            "when its predicted probability exceeds this value."
        )
    )

    min_area = st.sidebar.slider(
        "Minimum region area (pixels)",
        min_value=1,
        max_value=500,
        value=default_min_area,
        step=1,
        help=(
            "Connected regions smaller than this area "
            "are removed from the final change mask."
        )
    )

    st.sidebar.divider()

    show_heatmap = st.sidebar.checkbox(
        "Show probability heatmap",
        value=False
    )

    st.sidebar.caption(
        f"Model: {settings.get('model_name', 'GeoChange-AI')}"
    )

    st.sidebar.caption(
        f"Warm-up time: {warmup_ms:.1f} ms"
    )

    # --------------------------------------------------------
    # INPUT SELECTION
    # --------------------------------------------------------

    st.header("1. Select satellite images")

    demo_pairs = get_demo_pairs()

    demo_options = [
        "Upload your own images"
    ] + list(demo_pairs.keys())

    selected_demo = st.selectbox(
        "Choose a demonstration pair or upload your own",
        demo_options
    )

    image_a = None
    image_b = None
    gt_mask = None

    if selected_demo != "Upload your own images":

        selected = demo_pairs[selected_demo]

        try:

            image_a = read_demo_image(
                selected["image_a"]
            )

            image_b = read_demo_image(
                selected["image_b"]
            )

            gt_mask = read_ground_truth(
                selected["gt_mask"]
            )

            st.success(
                f"Loaded demonstration pair: {selected_demo}"
            )

        except Exception as exc:

            st.error(str(exc))
            st.stop()

    else:

        col_a, col_b = st.columns(2)

        with col_a:

            upload_a = st.file_uploader(
                "Image A — Earlier image",
                type=["jpg", "jpeg", "png"],
                key="image_a"
            )

        with col_b:

            upload_b = st.file_uploader(
                "Image B — Later image",
                type=["jpg", "jpeg", "png"],
                key="image_b"
            )

        if upload_a is not None and upload_b is not None:

            try:

                image_a = decode_upload(
                    upload_a.getvalue()
                )

                image_b = decode_upload(
                    upload_b.getvalue()
                )

            except Exception as exc:

                st.error(str(exc))
                st.stop()

        elif upload_a is not None or upload_b is not None:

            st.warning(
                "Please upload both Image A and Image B."
            )

    # --------------------------------------------------------
    # IMAGE PREVIEW AND PREDICTION
    # --------------------------------------------------------

    if image_a is None or image_b is None:

        st.info(
            "Select a demonstration pair or upload two images "
            "to begin change detection."
        )

        st.stop()

    st.divider()

    st.header("2. Image comparison")

    preview_a, preview_b = st.columns(2)

    with preview_a:

        st.image(
            image_a,
            caption="Image A — Earlier",
            use_container_width=True
        )

    with preview_b:

        st.image(
            image_b,
            caption="Image B — Later",
            use_container_width=True
        )

    # --------------------------------------------------------
    # RUN DETECTION
    # --------------------------------------------------------

    st.divider()

    st.header("3. Detect infrastructure changes")

    if st.button(
        "Run change detection",
        type="primary",
        use_container_width=True
    ):

        try:

            # Prepare image pair.
            input_a, input_b, warning = prepare_pair(
                image_a,
                image_b
            )

            if warning:
                st.warning(warning)

            # Predict.
            with st.spinner(
                "Analysing satellite images..."
            ):

                probability_map, inference_ms = predict_timed(
                    model,
                    input_a,
                    input_b
                )

            # Post-process.
            binary_mask, n_regions, changed_area_pct = analyse(
                probability_map,
                threshold=threshold,
                min_area=min_area
            )

            # Draw overlay.
            overlay_image = overlay(
                image_b,
                binary_mask
            )

            # ------------------------------------------------
            # RESULTS
            # ------------------------------------------------

            st.success(
                "Change detection completed successfully."
            )

            st.header("4. Detection results")

            metric1, metric2, metric3 = st.columns(3)

            metric1.metric(
                "Changed regions",
                f"{n_regions:,}"
            )

            metric2.metric(
                "Changed area",
                f"{changed_area_pct:.2f}%"
            )

            metric3.metric(
                "Inference time",
                f"{inference_ms:.1f} ms"
            )

            # ------------------------------------------------
            # RESULT IMAGES
            # ------------------------------------------------

            st.subheader("Visual results")

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.image(
                    image_a,
                    caption="Image A",
                    use_container_width=True
                )

            with col2:

                st.image(
                    image_b,
                    caption="Image B",
                    use_container_width=True
                )

            with col3:

                st.image(
                    binary_mask * 255,
                    caption="Predicted binary change mask",
                    clamp=True,
                    use_container_width=True
                )

            with col4:

                st.image(
                    overlay_image,
                    caption="Detected regions on Image B",
                    use_container_width=True
                )

            # ------------------------------------------------
            # OPTIONAL PROBABILITY HEATMAP
            # ------------------------------------------------

            if show_heatmap:

                st.subheader(
                    "Change probability heatmap"
                )

                st.image(
                    make_heatmap(probability_map),
                    caption=(
                        "Higher heatmap intensity indicates "
                        "higher predicted change probability."
                    ),
                    use_container_width=True
                )

            # ------------------------------------------------
            # OPTIONAL GROUND-TRUTH COMPARISON
            # ------------------------------------------------

            if gt_mask is not None:

                st.divider()

                st.subheader(
                    "Ground-truth comparison"
                )

                gt_f1 = calculate_pixel_f1(
                    binary_mask,
                    gt_mask
                )

                gt_col1, gt_col2 = st.columns(2)

                with gt_col1:

                    st.image(
                        gt_mask * 255,
                        caption="Ground-truth change mask",
                        clamp=True,
                        use_container_width=True
                    )

                with gt_col2:

                    st.metric(
                        "Pixel-level F1 score",
                        f"{gt_f1:.4f}"
                    )

                st.caption(
                    "This F1 score compares the current prediction "
                    "with the demonstration pair's ground-truth "
                    "mask. It is not a new evaluation of the "
                    "complete test dataset."
                )

            # ------------------------------------------------
            # EXPLANATION
            # ------------------------------------------------

            st.divider()

            with st.expander("How GeoChange-AI works"):

                st.markdown(
                    """
                    **Step 1 — Image input**

                    Two satellite images of the same location,
                    captured at different times, are provided.

                    **Step 2 — Preprocessing**

                    Both images are resized to 256 × 256 pixels
                    and normalized using MobileNetV2 preprocessing.

                    **Step 3 — Feature extraction**

                    A shared MobileNetV2 encoder extracts
                    features from both images.

                    **Step 4 — Change detection**

                    The Siamese U-Net compares the extracted
                    feature maps and predicts a pixel-level
                    change-probability map.

                    **Step 5 — Post-processing**

                    A probability threshold and minimum-region
                    area filter produce the final binary mask.

                    **Final output**

                    The application displays detected regions,
                    changed-area percentage and inference time.
                    """
                )

        except Exception as exc:

            st.error(
                f"Change detection failed: {exc}"
            )

    # --------------------------------------------------------
    # FOOTER
    # --------------------------------------------------------

    st.divider()

    st.caption(
        "GeoChange-AI | Independent academic portfolio project | "
        "TensorFlow / Keras | CPU inference"
    )


if __name__ == "__main__":
    main()

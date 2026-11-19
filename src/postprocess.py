# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 13:45:25 2026

@author: Admin
"""

# src/postprocess.py

import numpy as np
import cv2


def clean_mask(
    prob_or_binary,
    threshold=0.5,
    min_area=20
):
    """
    Convert probability/binary mask into a cleaned binary mask.

    Returns:
        kept_mask: uint8 mask containing 0 and 1.
        n_regions: number of retained connected components.
        changed_area_pct: percentage of image classified as changed.
    """

    mask = np.asarray(
        prob_or_binary
    )

    if mask.ndim != 2:
        raise ValueError(
            "Expected a 2D mask (H, W)."
        )

    if min_area < 1:
        raise ValueError(
            "min_area must be at least 1."
        )

    if not np.all(np.isfinite(mask)):
        raise ValueError(
            "Mask contains NaN or infinite values."
        )

    # Convert probabilities into binary predictions.
    binary = (mask > threshold).astype(
        np.uint8
    )

    # Find connected regions.
    n_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )

    kept_mask = np.zeros_like(
        binary,
        dtype=np.uint8
    )

    kept_regions = 0

    # Label 0 represents the background.
    for label_id in range(1, n_labels):

        area = stats[
            label_id,
            cv2.CC_STAT_AREA
        ]

        if area >= min_area:

            kept_mask[
                labels == label_id
            ] = 1

            kept_regions += 1

    # Percentage of pixels retained as changed.
    changed_area_pct = (
        np.sum(kept_mask) /
        kept_mask.size
    ) * 100.0

    return (
        kept_mask,
        int(kept_regions),
        float(changed_area_pct)
    )
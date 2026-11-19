# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 13:46:35 2026

@author: Admin
"""

# tests/test_postprocess.py

import numpy as np

from src.postprocess import clean_mask


def test_clean_mask_removes_tiny_region():

    mask = np.zeros(
        (100, 100),
        dtype=np.float32
    )

    # Blob 1: 5 x 5 = 25 pixels
    mask[5:10, 5:10] = 1.0

    # Blob 2: 6 x 6 = 36 pixels
    mask[20:26, 20:26] = 1.0

    # Blob 3: 4 x 8 = 32 pixels
    mask[40:44, 40:48] = 1.0

    # Blob 4: 2 x 2 = 4 pixels
    # This should be removed.
    mask[70:72, 70:72] = 1.0

    cleaned, n_regions, area_pct = clean_mask(
        mask,
        threshold=0.5,
        min_area=20
    )

    # Three regions should remain.
    assert n_regions == 3

    # Retained pixels = 25 + 36 + 32 = 93.
    assert np.sum(cleaned) == 93

    # 93 / 10000 * 100 = 0.93%.
    assert np.isclose(
        area_pct,
        0.93
    )

    assert cleaned.dtype == np.uint8

    assert set(np.unique(cleaned)).issubset(
        {0, 1}
    )

    # Tiny component should disappear.
    assert np.sum(cleaned[70:72, 70:72]) == 0
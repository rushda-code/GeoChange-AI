# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 14:44:29 2026

@author: Admin
"""


"""
GeoChange-AI — STEP 6b+
Create six demonstration pairs.

TEST is read ONLY to select and copy demo examples.
No metrics are recomputed.
No predictions are generated.
No evaluation files are modified.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import sys
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

TEST_PARQUET = (
    ROOT / "data/raw/data/"
    "test-00000-of-00001-31d7c3e3444e5b5d.parquet"
)

TEST_RESULTS = (
    ROOT / "experiments/results/"
    "test_final_model_per_crop.csv"
)

OUTPUT_DIR = ROOT / "app/demo_pairs"

RANDOM_SEED = 0


# ------------------------------------------------------------
# COLUMN HELPERS
# ------------------------------------------------------------

def find_column(df, candidates, description):

    for name in candidates:

        if name in df.columns:
            return name

    raise KeyError(
        f"Could not find {description} column.\n"
        f"Available columns: {list(df.columns)}"
    )


def get_bytes(value):

    """
    Hugging Face image columns are commonly stored as
    dictionaries containing a 'bytes' field.
    """

    if isinstance(value, dict):

        if value.get("bytes") is not None:
            return value["bytes"]

        raise ValueError(
            "Image dictionary does not contain encoded bytes."
        )

    if isinstance(value, bytes):
        return value

    if isinstance(value, bytearray):
        return bytes(value)

    raise TypeError(
        f"Unexpected encoded image type: {type(value)}"
    )


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():

    print("\n" + "=" * 75)
    print("GEOCHANGE-AI — DEMO PAIR EXPORT")
    print("=" * 75)

    if not TEST_PARQUET.exists():
        raise FileNotFoundError(TEST_PARQUET)

    if not TEST_RESULTS.exists():
        raise FileNotFoundError(TEST_RESULTS)

    # --------------------------------------------------------
    # 1. READ EXISTING TEST RESULTS
    # --------------------------------------------------------

    results = pd.read_csv(TEST_RESULTS)

    index_col = find_column(
        results,
        ["index", "crop_index", "test_index", "row_index"],
        "crop index"
    )

    change_col = find_column(
        results,
        ["gt_change_pct", "ground_truth_change_pct", "gt_change_percent"],
        "ground-truth change percentage"
    )

    fp_col = find_column(
      results,
      ["fp_pixels", "false_positive_pixels", "false_positives", "FP"],
      "false-positive pixel count"
    )

    results[index_col] = results[index_col].astype(int)
    results[change_col] = results[change_col].astype(float)
    results[fp_col] = results[fp_col].astype(int)

    print("Existing TEST result rows:", len(results))
    print("Columns used:")
    print("  Index:", index_col)
    print("  GT change:", change_col)
    print("  FP pixels:", fp_col)

    # --------------------------------------------------------
    # 2. SELECT SIX ROWS
    # --------------------------------------------------------

    rng = np.random.default_rng(RANDOM_SEED)

    moderate = results[
        (results[change_col] >= 5.0) &
        (results[change_col] <= 20.0)
    ]

    large = results[
        results[change_col] > 20.0
    ]

    small = results[
        (results[change_col] > 0.0) &
        (results[change_col] < 5.0)
    ]

    empty = results[
        results[change_col] == 0.0
    ]

    empty_correct = empty[
        empty[fp_col] == 0
    ]

    if len(moderate) < 2:
        raise ValueError(
            "Fewer than two crops have 5–20% GT change."
        )

    if len(large) == 0:
        raise ValueError("No crop has more than 20% GT change.")

    if len(small) == 0:
        raise ValueError(
            "No crop has more than 0% and less than 5% GT change."
        )

    if len(empty_correct) == 0:
        raise ValueError(
            "No empty-GT crop has zero false-positive pixels."
        )

    if len(empty) == 0:
        raise ValueError("No empty-GT crops found.")

    selected_moderate = moderate.iloc[
        rng.choice(len(moderate), size=2, replace=False)
    ]

    selected_large = large.sort_values(
        change_col,
        ascending=False
    ).iloc[0]

    selected_small = small.sort_values(
        change_col,
        ascending=False
    ).iloc[0]

    selected_empty = empty_correct.iloc[0]

    selected_failure = empty.sort_values(
        fp_col,
        ascending=False
    ).iloc[0]

    selections = [
        (
            "moderate_change_1",
            selected_moderate.iloc[0],
            "Random example with 5–20% ground-truth change."
        ),
        (
            "moderate_change_2",
            selected_moderate.iloc[1],
            "Second random example with 5–20% ground-truth change."
        ),
        (
            "large_change",
            selected_large,
            "Example with more than 20% ground-truth change."
        ),
        (
            "small_change",
            selected_small,
            "Example with more than 0% and less than 5% ground-truth change."
        ),
        (
            "no_change",
            selected_empty,
            "Empty ground truth and zero false-positive pixels."
        ),
        (
            "honest_failure",
            selected_failure,
            "Empty ground truth with the highest false-positive pixel count."
        )
    ]

    # --------------------------------------------------------
    # 3. READ ORIGINAL TEST IMAGE BYTES
    # --------------------------------------------------------

    print("\nReading TEST Parquet for selected examples only...")

    table = pq.read_table(TEST_PARQUET)

    required_columns = {"imageA", "imageB"}

    if not required_columns.issubset(set(table.column_names)):
        raise KeyError(
            f"Expected imageA and imageB. "
            f"Found: {table.column_names}"
        )

    mask_candidates = [
        "label",
        "mask",
        "gt",
        "ground_truth"
    ]

    mask_col = next(
        (
            name for name in mask_candidates
            if name in table.column_names
        ),
        None
    )

    if mask_col is None:
        raise KeyError(
            "Could not find the ground-truth mask column. "
            f"Available columns: {table.column_names}"
        )

    if len(table) != 2048:
        raise ValueError(
            f"Expected 2048 TEST rows, found {len(table)}."
        )

    if table.num_rows <= int(results[index_col].max()):
        raise IndexError(
            "A crop index in the results CSV is outside the TEST Parquet."
        )

    # --------------------------------------------------------
    # 4. COPY SIX ORIGINAL ENCODED PAIRS
    # --------------------------------------------------------

    if OUTPUT_DIR.exists():

        print("\nRemoving previous demo pair folder...")

        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    readme_rows = []

    for name, result_row, reason in selections:

        crop_index = int(result_row[index_col])

        row = table.slice(
            crop_index,
            1
        ).to_pylist()[0]

        image_a_bytes = get_bytes(row["imageA"])
        image_b_bytes = get_bytes(row["imageB"])
        mask_bytes = get_bytes(row[mask_col])

        pair_dir = OUTPUT_DIR / name

        pair_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # Write original bytes directly.
        # No decoding, resizing or re-encoding.

        (pair_dir / "A.jpg").write_bytes(
            image_a_bytes
        )

        (pair_dir / "B.jpg").write_bytes(
            image_b_bytes
        )

        (pair_dir / "gt.png").write_bytes(
            mask_bytes
        )

        readme_rows.append({
            "name": name,
            "test_crop_index": crop_index,
            "gt_change_pct": float(result_row[change_col]),
            "fp_pixels": int(result_row[fp_col]),
            "reason": reason
        })

        print(
            f"{name:20s} | "
            f"Index={crop_index:4d} | "
            f"GT change={float(result_row[change_col]):7.3f}% | "
            f"FP={int(result_row[fp_col])}"
        )

    # --------------------------------------------------------
    # 5. WRITE README
    # --------------------------------------------------------

    readme_path = OUTPUT_DIR / "README.txt"

    with open(
        readme_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "GEOCHANGE-AI — DEMO PAIRS\n"
            "=========================\n\n"
        )

        f.write(
            "These six examples were selected from the already "
            "evaluated TEST split.\n\n"
        )

        f.write(
            "The files A.jpg, B.jpg and gt.png contain the "
            "original encoded bytes from the Parquet dataset.\n"
        )

        f.write(
            "They were not re-encoded or resized.\n\n"
        )

        f.write(
            "No metrics were recalculated and no model evaluation "
            "was performed by this script.\n\n"
        )

        f.write(
            "Dataset imagery is for local demonstration only. "
            "Do not redistribute it.\n\n"
        )

        for item in readme_rows:

            f.write("-" * 65 + "\n")

            f.write(f"Demo: {item['name']}\n")
            f.write(
                f"TEST crop index: {item['test_crop_index']}\n"
            )
            f.write(
                f"GT change percentage: "
                f"{item['gt_change_pct']:.3f}%\n"
            )
            f.write(
                f"False-positive pixels: {item['fp_pixels']}\n"
            )
            f.write(f"Selection reason: {item['reason']}\n\n")

    # --------------------------------------------------------
    # 6. FINAL TABLE
    # --------------------------------------------------------

    output_df = pd.DataFrame(readme_rows)

    print("\n" + "=" * 75)
    print("DEMO PAIR SELECTION TABLE")
    print("=" * 75)

    print(
        output_df.to_string(
            index=False
        )
    )

    print("\nSaved demo pairs to:")
    print(OUTPUT_DIR)

    print("\nSaved README:")
    print(readme_path)

    print("\nSTEP 6b+ DEMO EXPORT COMPLETE.")
    print("No training performed.")
    print("No metrics recomputed.")
    print("TEST_EVAL_DONE.json was not accessed or modified.")


if __name__ == "__main__":
    main()

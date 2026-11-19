"""
GeoChange AI — STEP 2A.2: Reproducible cleaning manifest.

This script reads the original Parquet files without modifying them.
It writes:
  data/processed/train_manifest.csv
  data/processed/dup_report_val_test.csv
  data/processed/duplicate_groups_all.csv
  data/processed/partial_image_matches.csv
  data/processed/file_signature_counts.csv

Run from the project root:
  python src/manifest.py --data-dir data --output-dir data/processed
"""

import argparse
import csv
import hashlib
import json
from collections import defaultdict, Counter
from pathlib import Path

import pyarrow.parquet as pq


SPLITS = ("train", "val", "test")
COLUMNS = ("imageA", "imageB", "label")
SIGNATURES = {
    "JPEG": b"\xff\xd8\xff",
    "PNG": b"\x89PNG",
}


def sha1_bytes(value):
    """Return a stable SHA1 hex digest for encoded file bytes."""
    if value is None:
        raise ValueError("Found a null encoded value; cannot hash it.")
    return hashlib.sha1(value).hexdigest()


def classify_signature(value):
    """Classify bytes by their leading JPEG/PNG signature."""
    if value.startswith(SIGNATURES["JPEG"]):
        return "JPEG"
    if value.startswith(SIGNATURES["PNG"]):
        return "PNG"
    return "other"


def find_split_files(data_dir):
    """Find exactly one Parquet file for each split under data_dir."""
    found = {}
    for path in sorted(Path(data_dir).rglob("*.parquet")):
        lower_name = path.name.lower()
        matched = [split for split in SPLITS if split in lower_name]
        if len(matched) == 1:
            split = matched[0]
            if split in found:
                raise RuntimeError(
                    f"More than one Parquet file matched split '{split}': "
                    f"{found[split]} and {path}"
                )
            found[split] = path

    missing = set(SPLITS) - set(found)
    if missing:
        raise FileNotFoundError(
            f"Could not find Parquet files for splits {sorted(missing)} "
            f"under {data_dir}"
        )
    return found


def load_split_records(parquet_path, split):
    """
    Read encoded bytes and paths from Parquet in batches.
    No image decoding is performed. Each record stores hashes, not image arrays.
    """
    parquet_file = pq.ParquetFile(parquet_path)
    required = ["imageA", "imageB", "label"]
    missing = set(required) - set(parquet_file.schema_arrow.names)
    if missing:
        raise ValueError(f"{parquet_path} is missing columns: {sorted(missing)}")

    records = []
    signature_counts = {
        column: Counter({"JPEG": 0, "PNG": 0, "other": 0})
        for column in COLUMNS
    }

    row_index = 0
    for batch in parquet_file.iter_batches(
        batch_size=128,
        columns=required
    ):
        for row in batch.to_pylist():
            record = {
                "split": split,
                "row_index": row_index,
            }

            for column in COLUMNS:
                value = row[column]
                if not isinstance(value, dict):
                    raise TypeError(
                        f"{split} row {row_index} column {column} "
                        f"was expected to be a struct, got {type(value).__name__}"
                    )

                encoded = value.get("bytes")
                path = value.get("path")

                if encoded is None:
                    raise ValueError(
                        f"Missing encoded bytes: {split} row {row_index} {column}"
                    )

                record[f"{column}_hash"] = sha1_bytes(encoded)
                record[f"{column}_path"] = path
                signature_counts[column][classify_signature(encoded)] += 1

            record["pair_key"] = (
                record["imageA_hash"],
                record["imageB_hash"],
            )
            record["group_id"] = hashlib.sha1(
                (record["imageA_hash"] + record["imageB_hash"]).encode("ascii")
            ).hexdigest()

            records.append(record)
            row_index += 1

    return records, signature_counts


def member_summary(record):
    """Small, JSON-serializable member description for reports."""
    return {
        "split": record["split"],
        "row_index": record["row_index"],
        "imageA_path": record["imageA_path"],
        "imageB_path": record["imageB_path"],
        "label_path": record["label_path"],
        "label_sha1": record["label_hash"],
    }


def duplicate_groups_for_comparison(records_by_split, left, right):
    """
    Find repeated pair-key groups for one requested split comparison.
    For within-split comparisons, each group must contain at least two rows.
    For cross-split comparisons, each group must contain members from both.
    """
    combined = records_by_split[left]
    if right != left:
        combined = combined + records_by_split[right]

    by_pair = defaultdict(list)
    for record in combined:
        by_pair[record["pair_key"]].append(record)

    output = []
    for pair_key, members in by_pair.items():
        if left == right:
            qualifies = len(members) > 1
        else:
            has_left = any(m["split"] == left for m in members)
            has_right = any(m["split"] == right for m in members)
            qualifies = has_left and has_right

        if not qualifies:
            continue

        label_hashes = {m["label_hash"] for m in members}
        output.append({
            "comparison": f"{left}-{right}",
            "group_id": members[0]["group_id"],
            "group_size": len(members),
            "label_hashes_all_equal": len(label_hashes) == 1,
            "members": [member_summary(m) for m in sorted(
                members, key=lambda x: (x["split"], x["row_index"])
            )],
        })

    output.sort(key=lambda x: (x["comparison"], x["group_id"]))
    return output


def partial_image_matches_for_comparison(records_by_split, left, right):
    """
    Report same imageA with differing imageB, and same imageB with differing
    imageA. These are partial matches, not necessarily leakage.
    """
    combined = records_by_split[left]
    if right != left:
        combined = combined + records_by_split[right]

    results = []

    for matched_column, differing_column, description in (
        ("imageA_hash", "imageB_hash", "same_imageA_different_imageB"),
        ("imageB_hash", "imageA_hash", "same_imageB_different_imageA"),
    ):
        groups = defaultdict(list)
        for record in combined:
            groups[record[matched_column]].append(record)

        for matched_hash, members in groups.items():
            differing_hashes = {m[differing_column] for m in members}
            if len(differing_hashes) <= 1:
                continue

            if left == right:
                qualifies = len(members) > 1
            else:
                qualifies = (
                    any(m["split"] == left for m in members)
                    and any(m["split"] == right for m in members)
                )

            if not qualifies:
                continue

            results.append({
                "comparison": f"{left}-{right}",
                "match_type": description,
                "matched_component_sha1": matched_hash,
                "group_size": len(members),
                "members": [member_summary(m) for m in sorted(
                    members, key=lambda x: (x["split"], x["row_index"])
                )],
            })

    results.sort(key=lambda x: (
        x["comparison"], x["match_type"], x["matched_component_sha1"]
    ))
    return results


def write_json_report(path, rows, fieldnames):
    """Write deterministic CSV with nested member lists encoded as JSON."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            converted = dict(row)
            if "members" in converted:
                converted["members"] = json.dumps(
                    converted["members"],
                    sort_keys=True,
                    separators=(",", ":"),
                )
            writer.writerow(converted)


def build_train_manifest(records_by_split, output_dir, expected_train_rows=7120):
    """Apply the requested train-only exclusion rules in a deterministic order."""
    train_records = records_by_split["train"]

    val_test_pair_keys = {
        record["pair_key"]
        for split in ("val", "test")
        for record in records_by_split[split]
    }

    train_groups = defaultdict(list)
    for record in train_records:
        train_groups[record["pair_key"]].append(record)

    manifest_rows = []

    for pair_key, members in train_groups.items():
        members = sorted(members, key=lambda x: x["row_index"])
        group_id = members[0]["group_id"]

        # Rule A has highest priority: any pair key found in val or test.
        if pair_key in val_test_pair_keys:
            for record in members:
                manifest_rows.append({
                    "row_index": record["row_index"],
                    "path": record["imageA_path"],
                    "keep": False,
                    "reason": "cross_split_copy",
                    "group_id": group_id,
                })
            continue

        # Rule C: if this pair key has multiple label hashes, exclude ALL
        # members. This group-level check prevents keeping one arbitrary label.
        label_hashes = {record["label_hash"] for record in members}
        if len(label_hashes) > 1:
            for record in members:
                manifest_rows.append({
                    "row_index": record["row_index"],
                    "path": record["imageA_path"],
                    "keep": False,
                    "reason": "conflicting_labels",
                    "group_id": group_id,
                })
            continue

        # Rule B: labels agree, so keep the lowest row index only.
        kept_record = members[0]
        manifest_rows.append({
            "row_index": kept_record["row_index"],
            "path": kept_record["imageA_path"],
            "keep": True,
            "reason": "kept",
            "group_id": group_id,
        })

        for record in members[1:]:
            manifest_rows.append({
                "row_index": record["row_index"],
                "path": record["imageA_path"],
                "keep": False,
                "reason": f"duplicate_of_{kept_record['row_index']}",
                "group_id": group_id,
            })

    manifest_rows.sort(key=lambda x: x["row_index"])

    if len(manifest_rows) != len(train_records):
        raise AssertionError(
            f"Manifest has {len(manifest_rows)} rows but train has "
            f"{len(train_records)} rows."
        )

    if len(train_records) != expected_train_rows:
        raise AssertionError(
            f"Expected {expected_train_rows} training rows from the inspected "
            f"dataset, found {len(train_records)}. Check the input files."
        )

    kept = sum(bool(row["keep"]) for row in manifest_rows)
    dropped = len(manifest_rows) - kept
    assert kept + dropped == 7120, (
        f"Kept + dropped must equal 7120; got {kept} + {dropped}"
    )

    output_path = Path(output_dir) / "train_manifest.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["row_index", "path", "keep", "reason", "group_id"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    reason_counts = Counter(
        row["reason"] for row in manifest_rows if not row["keep"]
    )

    return manifest_rows, {
        "train_rows": len(train_records),
        "kept": kept,
        "dropped": dropped,
        "dropped_by_reason": dict(sorted(reason_counts.items())),
        "manifest_path": str(output_path),
    }


def build_manifests(data_dir, output_dir):
    """Run signature audit, hashing, duplicate reports and train manifest."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_files = find_split_files(data_dir)

    records_by_split = {}
    signatures_by_split = {}

    print("INPUT PARQUET FILES")
    for split in SPLITS:
        print(f"  {split}: {split_files[split]}")

    # Read encoded bytes in batches, hash each component, and count signatures.
    for split in SPLITS:
        print(f"\nHashing {split}...")
        records, signature_counts = load_split_records(
            split_files[split], split
        )
        records_by_split[split] = records
        signatures_by_split[split] = signature_counts
        print(f"  rows hashed: {len(records)}")

    # 1. File signature report.
    signature_rows = []
    print("\nFILE SIGNATURE COUNTS")
    for split in SPLITS:
        for column in COLUMNS:
            counts = signatures_by_split[split][column]
            row = {
                "split": split,
                "column": column,
                "JPEG": counts["JPEG"],
                "PNG": counts["PNG"],
                "other": counts["other"],
                "total": sum(counts.values()),
            }
            signature_rows.append(row)
            print(
                f"  {split:5s} {column:6s}: "
                f"JPEG={row['JPEG']}, PNG={row['PNG']}, "
                f"other={row['other']}, total={row['total']}"
            )

    signature_path = output_dir / "file_signature_counts.csv"
    with open(signature_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "column", "JPEG", "PNG", "other", "total"],
        )
        writer.writeheader()
        writer.writerows(signature_rows)

    # 2-3. Duplicate pair-key reports for all six requested combinations.
    comparisons = (
        ("train", "train"),
        ("val", "val"),
        ("test", "test"),
        ("train", "val"),
        ("train", "test"),
        ("val", "test"),
    )

    all_duplicate_groups = []
    all_partial_matches = []

    print("\nDUPLICATE PAIR-KEY GROUPS")
    for left, right in comparisons:
        groups = duplicate_groups_for_comparison(
            records_by_split, left, right
        )
        all_duplicate_groups.extend(groups)
        print(f"\n  {left}-{right}: {len(groups)} duplicate groups")

        for group in groups:
            print(
                f"    group_id={group['group_id'][:12]} "
                f"size={group['group_size']} "
                f"all_label_hashes_equal={group['label_hashes_all_equal']}"
            )
            for member in group["members"]:
                print(
                    f"      ({member['split']}, {member['row_index']}, "
                    f"{member['imageA_path']})"
                )

        partial = partial_image_matches_for_comparison(
            records_by_split, left, right
        )
        all_partial_matches.extend(partial)
        print(
            f"    partial image matches (same A/different B or "
            f"same B/different A): {len(partial)} groups"
        )

    write_json_report(
        output_dir / "duplicate_groups_all.csv",
        all_duplicate_groups,
        [
            "comparison", "group_id", "group_size",
            "label_hashes_all_equal", "members"
        ],
    )

    write_json_report(
        output_dir / "partial_image_matches.csv",
        all_partial_matches,
        [
            "comparison", "match_type", "matched_component_sha1",
            "group_size", "members"
        ],
    )

    # 5. Val/test duplicate report only. These rows are report-only.
    val_test_groups = [
        group for group in all_duplicate_groups
        if group["comparison"] in ("val-val", "test-test", "val-test")
    ]
    write_json_report(
        output_dir / "dup_report_val_test.csv",
        val_test_groups,
        [
            "comparison", "group_id", "group_size",
            "label_hashes_all_equal", "members"
        ],
    )

    # 4-6. Apply train-only rules and save deterministic manifest.
    manifest_rows, manifest_summary = build_train_manifest(
        records_by_split, output_dir
    )

    # Report whether any duplicate pair-key group has >2 members.
    groups_over_two = [
        group for group in all_duplicate_groups
        if group["group_size"] > 2
    ]
    any_group_over_two = bool(groups_over_two)

    print("\nTRAIN MANIFEST SUMMARY")
    print(f"  original train rows: {manifest_summary['train_rows']}")
    print(f"  rows kept: {manifest_summary['kept']}")
    print(f"  rows dropped: {manifest_summary['dropped']}")
    for reason, count in manifest_summary["dropped_by_reason"].items():
        print(f"  dropped — {reason}: {count}")
    print(f"  kept + dropped: {manifest_summary['kept'] + manifest_summary['dropped']}")
    print(f"  any duplicate group with >2 members: {any_group_over_two}")
    print(f"  groups with >2 members: {len(groups_over_two)}")
    print(f"  manifest: {manifest_summary['manifest_path']}")
    print(f"  val/test duplicate report: {output_dir / 'dup_report_val_test.csv'}")
    print(f"  all duplicate groups: {output_dir / 'duplicate_groups_all.csv'}")
    print(f"  partial image matches: {output_dir / 'partial_image_matches.csv'}")
    print(f"  file signatures: {signature_path}")

    return {
        "split_files": split_files,
        "signature_rows": signature_rows,
        "duplicate_groups": all_duplicate_groups,
        "partial_matches": all_partial_matches,
        "val_test_duplicate_groups": val_test_groups,
        "manifest_rows": manifest_rows,
        "manifest_summary": manifest_summary,
        "any_group_over_two": any_group_over_two,
        "groups_over_two": groups_over_two,
        "output_dir": str(output_dir),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build GeoChange AI reproducible train cleaning manifest."
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing raw Parquet files (searched recursively).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory for CSV reports and manifest.",
    )
    args = parser.parse_args()
    build_manifests(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Pair individual-QK and cumulative-Q1:QK trajectory metrics."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from .aggregate_rotations import prepare_output_dir, read_csv, write_csv


PAIR_FIELDS = (
    "rotation", "test_fold", "codec", "depth", "seed",
    "group_type", "group_value", "metric",
)


def index_rows(path: Path, expected_mode: str) -> dict[tuple, dict]:
    rows = read_csv(path / "trajectory_long.csv")
    indexed = {}
    for row in rows:
        mode = row.get("rvq_mode")
        if not mode:
            mode = "cumulative" if expected_mode == "cumulative" else ""
        if mode != expected_mode:
            raise ValueError(f"Expected {expected_mode} rows in {path}, found {mode!r}")
        key = tuple(row[field] for field in PAIR_FIELDS)
        if key in indexed:
            raise ValueError(f"Duplicate comparison key {key} in {path}")
        indexed[key] = row
    if not indexed:
        raise ValueError(f"No trajectory rows found in {path}")
    return indexed


def compare(cumulative_root: Path, individual_root: Path) -> list[dict]:
    cumulative = index_rows(cumulative_root, "cumulative")
    individual = index_rows(individual_root, "individual")
    if set(cumulative) != set(individual):
        missing_individual = sorted(set(cumulative) - set(individual))[:5]
        missing_cumulative = sorted(set(individual) - set(cumulative))[:5]
        raise ValueError(
            "Representation rows do not pair exactly: "
            f"missing_individual={missing_individual}, "
            f"missing_cumulative={missing_cumulative}"
        )
    rows = []
    for key in sorted(cumulative):
        cumulative_row, individual_row = cumulative[key], individual[key]
        cumulative_value = float(cumulative_row["value"])
        individual_value = float(individual_row["value"])
        base = {field: value for field, value in zip(PAIR_FIELDS, key)}
        cumulative_condition = cumulative_row.get("condition") or (
            "cumulative_q1" if base["depth"] == "1"
            else f"cumulative_q1_{base['depth']}"
        )
        rows.append({
            **base,
            "individual_condition": individual_row.get("condition")
            or f"individual_q{base['depth']}",
            "cumulative_condition": cumulative_condition,
            "individual_value": individual_value,
            "cumulative_value": cumulative_value,
            "delta_individual_minus_cumulative": individual_value - cumulative_value,
        })
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    rotations = defaultdict(set)
    for row in rows:
        key = (
            row["codec"], int(row["depth"]), row["group_type"],
            row["group_value"], row["metric"],
        )
        grouped[key].append(float(row["delta_individual_minus_cumulative"]))
        rotations[key].add(int(row["rotation"]))
    output = []
    for key in sorted(grouped):
        values = grouped[key]
        output.append({
            "codec": key[0], "depth": key[1], "group_type": key[2],
            "group_value": key[3], "metric": key[4],
            "mean_delta_individual_minus_cumulative": statistics.fmean(values),
            "sd": statistics.stdev(values) if len(values) > 1 else "",
            "n_valid": len(values), "n_rotations": len(rotations[key]),
        })
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cumulative-root", type=Path, required=True)
    parser.add_argument("--individual-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(args: argparse.Namespace) -> None:
    rows = compare(args.cumulative_root, args.individual_root)
    summary = summarize(rows)
    prepare_output_dir(args.output_dir)
    write_csv(
        args.output_dir / "representation_comparison_long.csv", rows,
        (*PAIR_FIELDS, "individual_condition", "cumulative_condition",
         "individual_value", "cumulative_value", "delta_individual_minus_cumulative"),
    )
    write_csv(
        args.output_dir / "representation_comparison_summary.csv", summary,
        ("codec", "depth", "group_type", "group_value", "metric",
         "mean_delta_individual_minus_cumulative", "sd", "n_valid", "n_rotations"),
    )
    audit = {
        "status": "valid", "paired_rows": len(rows), "summary_rows": len(summary),
        "notes": [
            "Negative deltas favor individual QK for error metrics.",
            "WER and CER are ASR performance metrics, not clinical intelligibility.",
        ],
    }
    (args.output_dir / "comparison_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main(parse_args())

#!/usr/bin/env python3
"""Paired utterance bootstrap for WER/CER differences between two conditions."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from evaluate_with_faster_whisper import rvq_condition_from_row


def load_pairs(path: Path, condition_a: str, condition_b: str) -> dict[str, list[tuple[dict, dict]]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    indexed = {(rvq_condition_from_row(row), row["utt_id"]): row for row in rows}
    speakers = defaultdict(list)
    for row in rows:
        if rvq_condition_from_row(row) != condition_a:
            continue
        partner = indexed.get((condition_b, row["utt_id"]))
        if partner is not None:
            speakers[row["speaker_id"]].append((row, partner))
    if not speakers:
        raise ValueError(f"No paired {condition_a}/{condition_b} utterances found")
    return speakers


def corpus_rate(rows: list[dict], edits_key: str, units_key: str) -> float:
    edits = sum(row[edits_key] for row in rows)
    units = sum(row[units_key] for row in rows)
    return edits / units if units else 0.0


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap_speaker(
    pairs: list[tuple[dict, dict]], condition_a: str, condition_b: str,
    samples: int, seed: int,
) -> dict:
    rng = random.Random(seed)
    a_rows = [pair[0] for pair in pairs]
    b_rows = [pair[1] for pair in pairs]
    results = {
        "speaker_id": a_rows[0]["speaker_id"],
        "severity": a_rows[0]["severity"],
        "utterances": len(pairs),
        "condition_a": condition_a,
        "condition_b": condition_b,
    }
    for metric, edits_key, units_key in (
        ("wer", "word_edits", "reference_words"),
        ("cer", "character_edits", "reference_characters"),
    ):
        rate_a = corpus_rate(a_rows, edits_key, units_key)
        rate_b = corpus_rate(b_rows, edits_key, units_key)
        differences = []
        for _ in range(samples):
            indices = [rng.randrange(len(pairs)) for _ in pairs]
            sampled_a = [a_rows[index] for index in indices]
            sampled_b = [b_rows[index] for index in indices]
            differences.append(
                corpus_rate(sampled_b, edits_key, units_key)
                - corpus_rate(sampled_a, edits_key, units_key)
            )
        results[f"{condition_a}_{metric}"] = rate_a
        results[f"{condition_b}_{metric}"] = rate_b
        results[f"delta_{metric}_{condition_b}_minus_{condition_a}"] = rate_b - rate_a
        results[f"delta_{metric}_ci95_low"] = percentile(differences, 0.025)
        results[f"delta_{metric}_ci95_high"] = percentile(differences, 0.975)
        results[f"probability_{condition_b}_better_{metric}"] = sum(
            difference < 0 for difference in differences
        ) / samples
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--condition-a", default="k4")
    parser.add_argument("--condition-b", default="k8")
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 100:
        parser.error("--samples must be at least 100")
    paired = load_pairs(args.predictions, args.condition_a, args.condition_b)
    rows = [
        bootstrap_speaker(pairs, args.condition_a, args.condition_b, args.samples, args.seed + index)
        for index, (_, pairs) in enumerate(sorted(paired.items()))
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} speaker results to {args.output}")


if __name__ == "__main__":
    main()

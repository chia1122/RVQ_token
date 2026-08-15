#!/usr/bin/env python3
"""Build per-utterance WER/CER comparison tables from prediction JSONL."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for ref_index, ref_item in enumerate(reference, start=1):
        current = [ref_index]
        for hyp_index, hyp_item in enumerate(hypothesis, start=1):
            current.append(min(
                current[-1] + 1,
                previous[hyp_index] + 1,
                previous[hyp_index - 1] + (ref_item != hyp_item),
            ))
        previous = current
    return previous[-1]


def score(row: dict) -> dict:
    reference = row["reference"]
    hypothesis = row["hypothesis"]
    reference_words = reference.split()
    hypothesis_words = hypothesis.split()
    reference_characters = list(reference.replace(" ", ""))
    hypothesis_characters = list(hypothesis.replace(" ", ""))
    word_edits = edit_distance(reference_words, hypothesis_words)
    character_edits = edit_distance(reference_characters, hypothesis_characters)
    return {
        "utt_id": row["utt_id"],
        "speaker_id": row["speaker_id"],
        "severity": row["severity"],
        "reference": reference,
        "hypothesis": hypothesis,
        "word_edits": word_edits,
        "reference_words": len(reference_words),
        "wer": word_edits / len(reference_words) if reference_words else 0.0,
        "character_edits": character_edits,
        "reference_characters": len(reference_characters),
        "cer": character_edits / len(reference_characters) if reference_characters else 0.0,
        "length_ratio": (
            len(hypothesis_characters) / len(reference_characters)
            if reference_characters else 0.0
        ),
        "empty_hypothesis": not bool(hypothesis.strip()),
    }


def escape_markdown(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_markdown(path: Path, rows: list[dict], source: Path, limit: int) -> None:
    displayed = sorted(rows, key=lambda row: (row["cer"], row["wer"]), reverse=True)[:limit]
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# WER/CER 逐句文字比對\n\n")
        handle.write(f"來源：`{source}`。CER計算時忽略空格。以下列出CER最高的{len(displayed)}筆；完整結果請查看CSV。\n\n")
        handle.write("| Speaker | Severity | Reference | Hypothesis | WER | CER | Length ratio |\n")
        handle.write("|---|---|---|---|---:|---:|---:|\n")
        for row in displayed:
            values = (
                row["speaker_id"], row["severity"], row["reference"],
                row["hypothesis"] or "(empty)", f"{row['wer']:.3f}",
                f"{row['cer']:.3f}", f"{row['length_ratio']:.3f}",
            )
            handle.write("| " + " | ".join(escape_markdown(value) for value in values) + " |\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--markdown-rows", type=int, default=50)
    args = parser.parse_args()
    if args.markdown_rows < 1:
        parser.error("--markdown-rows must be positive")

    rows = []
    with args.predictions.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            source = json.loads(line)
            required = {"utt_id", "speaker_id", "severity", "reference", "hypothesis"}
            missing = required - set(source)
            if missing:
                raise ValueError(f"Prediction line {line_number} is missing {sorted(missing)}")
            rows.append(score(source))
    if not rows:
        raise ValueError("Prediction file is empty")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "prediction_comparison.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    markdown_path = args.output_dir / "prediction_comparison.md"
    write_markdown(markdown_path, rows, args.predictions, args.markdown_rows)
    print(f"Wrote {len(rows)} rows to {csv_path}")
    print(f"Wrote report table to {markdown_path}")


if __name__ == "__main__":
    main()

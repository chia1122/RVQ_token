#!/usr/bin/env python3
"""Evaluate original and RVQ-prefix reconstructed audio with one ASR model."""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path


RVQ_CONDITION = re.compile(r"^k([1-9][0-9]*)$", re.IGNORECASE)


def rvq_condition_from_row(row: dict) -> str:
    value = row.get("rvq_condition")
    legacy_value = str(row.get("condition", ""))
    if value is None and (legacy_value == "original" or RVQ_CONDITION.fullmatch(legacy_value)):
        value = legacy_value
    if value == "original":
        return value
    match = RVQ_CONDITION.fullmatch(str(value or ""))
    if not match:
        raise ValueError(f"Missing or invalid rvq_condition for {row.get('utt_id', 'row')}")
    return f"k{int(match.group(1))}"


def rvq_condition_order(values) -> list[str]:
    def sort_key(value: str):
        if value == "original":
            return 0, 0, value
        match = RVQ_CONDITION.fullmatch(value)
        if match:
            return 1, int(match.group(1)), value
        return 2, 0, value

    return sorted(set(values), key=sort_key)


def speech_condition(row: dict) -> str:
    value = row.get("condition", row.get("speaker_type"))
    if value not in {"control", "dysarthric"}:
        raise ValueError(f"Invalid speech condition for {row.get('utt_id', 'row')}")
    return value


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).upper()
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'").replace("-", " ")
    text = re.sub(r"[^A-Z' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


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


class Scores:
    def __init__(self):
        self.word_edits = self.words = self.char_edits = self.chars = 0

    def update(self, reference: str, hypothesis: str) -> None:
        ref_words, hyp_words = reference.split(), hypothesis.split()
        ref_chars, hyp_chars = list(reference.replace(" ", "")), list(hypothesis.replace(" ", ""))
        self.word_edits += edit_distance(ref_words, hyp_words)
        self.words += len(ref_words)
        self.char_edits += edit_distance(ref_chars, hyp_chars)
        self.chars += len(ref_chars)

    def row(self) -> dict:
        return {
            "utterances": getattr(self, "utterances", 0),
            "reference_words": self.words,
            "word_edits": self.word_edits,
            "wer": self.word_edits / self.words if self.words else 0.0,
            "reference_characters": self.chars,
            "character_edits": self.char_edits,
            "cer": self.char_edits / self.chars if self.chars else 0.0,
        }


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_items(args: argparse.Namespace) -> list[dict]:
    speakers = {item.strip() for item in args.speakers.split(",") if item.strip()}
    manifest = load_jsonl(args.manifest)
    manifest_by_utt = {row["utt_id"]: row for row in manifest}
    reconstruction_rows = load_jsonl(args.reconstruction_index)
    available_conditions = {rvq_condition_from_row(row) for row in reconstruction_rows}
    if args.conditions.strip().lower() == "auto":
        conditions = {"original", *available_conditions}
    else:
        conditions = {item.strip().lower() for item in args.conditions.split(",") if item.strip()}
        invalid = [value for value in conditions if value != "original" and not RVQ_CONDITION.fullmatch(value)]
        if invalid:
            raise ValueError(f"Invalid evaluation conditions: {sorted(invalid)}")
    items = []
    if "original" in conditions:
        if args.audio_root is None:
            raise ValueError(
                "--audio-root is required when --conditions includes original"
            )
        for row in manifest:
            if args.split != "all" and row["split"] != args.split:
                continue
            if speakers and row["speaker_id"] not in speakers:
                continue
            items.append({
                "rvq_condition": "original", "utt_id": row["utt_id"],
                "audio": args.audio_root / row["audio_path"],
                "speaker_id": row["speaker_id"], "condition": speech_condition(row),
                "severity": row["severity"],
                "split": row["split"], "reference": row["text_norm"],
            })
    for row in reconstruction_rows:
        rvq_condition = rvq_condition_from_row(row)
        if rvq_condition not in conditions:
            continue
        if args.split != "all" and row["split"] != args.split:
            continue
        if speakers and row["speaker_id"] not in speakers:
            continue
        manifest_row = manifest_by_utt[row["utt_id"]]
        items.append({
            "rvq_condition": rvq_condition, "utt_id": row["utt_id"],
            "audio": args.reconstruction_root / row["audio_path"],
            "speaker_id": row["speaker_id"],
            "condition": speech_condition(row) if row.get("condition") in {"control", "dysarthric"} else speech_condition(manifest_row),
            "severity": row["severity"],
            "split": row["split"], "reference": row["text_norm"],
        })
    if not items:
        raise ValueError("No evaluation items matched the requested conditions/split")
    if args.limit_per_condition:
        counts = defaultdict(int)
        limited = []
        for item in items:
            if counts[item["rvq_condition"]] >= args.limit_per_condition:
                continue
            limited.append(item)
            counts[item["rvq_condition"]] += 1
        items = limited
    return items


def write_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    temporary.replace(path)


def prediction_from_hypothesis(item: dict, raw_hypothesis: str) -> dict:
    hypothesis = normalize_text(raw_hypothesis)
    reference = normalize_text(item["reference"])
    reference_words = reference.split()
    ref_chars = list(reference.replace(" ", ""))
    word_edits = edit_distance(reference_words, hypothesis.split())
    char_edits = edit_distance(ref_chars, list(hypothesis.replace(" ", "")))
    return {
        "condition": item["condition"],
        "rvq_condition": item["rvq_condition"], "utt_id": item["utt_id"],
        "speaker_id": item["speaker_id"], "severity": item["severity"],
        "split": item["split"], "reference": reference,
        "hypothesis": hypothesis, "raw_hypothesis": raw_hypothesis,
        "word_edits": word_edits, "reference_words": len(reference_words),
        "wer": word_edits / len(reference_words),
        "character_edits": char_edits, "reference_characters": len(ref_chars),
        "cer": char_edits / len(ref_chars),
    }


def evaluate(args: argparse.Namespace) -> None:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise SystemExit("Install `faster-whisper` before ASR evaluation") from exc

    items = build_items(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    predictions_path = args.output_dir / "predictions.jsonl"
    predictions = [] if args.overwrite or not predictions_path.is_file() else load_jsonl(predictions_path)
    item_metadata = {(item["rvq_condition"], item["utt_id"]): item for item in items}
    normalized_predictions = []
    for row in predictions:
        rvq_condition = rvq_condition_from_row(row)
        metadata = item_metadata.get((rvq_condition, row["utt_id"]))
        if metadata is None:
            continue
        row["rvq_condition"] = rvq_condition
        row["condition"] = metadata["condition"]
        normalized_predictions.append(row)
    predictions = normalized_predictions
    completed = {(row["rvq_condition"], row["utt_id"]) for row in predictions}
    failures = []
    for index, item in enumerate(items, start=1):
        if (item["rvq_condition"], item["utt_id"]) in completed:
            continue
        try:
            segments, _ = model.transcribe(
                str(item["audio"]), language=args.language,
                beam_size=args.beam_size, temperature=0.0,
                condition_on_previous_text=False,
            )
            raw_hypothesis = " ".join(segment.text.strip() for segment in segments).strip()
            predictions.append(prediction_from_hypothesis(item, raw_hypothesis))
            completed.add((item["rvq_condition"], item["utt_id"]))
        except Exception as exc:
            failures.append({
                "rvq_condition": item["rvq_condition"], "utt_id": item["utt_id"],
                "error": f"{type(exc).__name__}: {exc}",
            })
            if not args.skip_errors:
                raise
        if index % args.log_every == 0 or index == len(items):
            write_jsonl(predictions_path, predictions)
            print(f"Processed {index}/{len(items)}; predictions={len(predictions)}; failures={len(failures)}")

    write_jsonl(predictions_path, predictions)
    write_jsonl(args.output_dir / "failures.jsonl", failures)
    if not predictions:
        raise RuntimeError("No successful ASR predictions were produced")

    groups = defaultdict(Scores)
    for row in predictions:
        for group_type, group_value in (
            ("overall", "all"), ("condition", row["condition"]),
            ("speaker", row["speaker_id"]),
            ("severity", row["severity"]),
        ):
            key = (row["rvq_condition"], group_type, group_value)
            groups[key].update(row["reference"], row["hypothesis"])
            groups[key].utterances = getattr(groups[key], "utterances", 0) + 1
    summary_rows = []
    condition_rank = {
        value: index for index, value in enumerate(
            rvq_condition_order(key[0] for key in groups)
        )
    }
    ordered_group_keys = sorted(
        groups, key=lambda key: (condition_rank[key[0]], key[1], key[2])
    )
    for (rvq_condition, group_type, group_value) in ordered_group_keys:
        scores = groups[(rvq_condition, group_type, group_value)]
        summary_rows.append({
            "rvq_condition": rvq_condition, "group_type": group_type,
            "group_value": group_value, **scores.row(),
        })
    with (args.output_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    condition_order = rvq_condition_order(row["rvq_condition"] for row in predictions)
    speaker_severity = {row["speaker_id"]: row["severity"] for row in predictions}
    speaker_conditions = {row["speaker_id"]: row["condition"] for row in predictions}
    for group_type, filename, identity_name in (
        ("condition", "comparison_by_condition.csv", "condition"),
        ("speaker", "comparison_by_speaker.csv", "speaker_id"),
        ("severity", "comparison_by_severity.csv", "severity"),
    ):
        group_values = sorted({
            group_value for _, current_type, group_value in groups
            if current_type == group_type
        })
        comparison = []
        for group_value in group_values:
            comparison_row = {identity_name: group_value}
            if group_type == "speaker":
                comparison_row["severity"] = speaker_severity[group_value]
                comparison_row["condition"] = speaker_conditions[group_value]
            for condition in condition_order:
                scores = groups.get((condition, group_type, group_value))
                comparison_row[f"{condition}_wer"] = scores.row()["wer"] if scores else ""
                comparison_row[f"{condition}_cer"] = scores.row()["cer"] if scores else ""
            comparison.append(comparison_row)
        with (args.output_dir / filename).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
            writer.writeheader()
            writer.writerows(comparison)
    experiment = {
        "asr_backend": "faster-whisper",
        "asr_backend_version": version("faster-whisper"),
        "asr_model": args.model, "device": args.device,
        "compute_type": args.compute_type, "language": args.language,
        "beam_size": args.beam_size, "conditions": condition_order,
        "split": args.split, "items": len(items),
        "predictions": len(predictions), "failures": len(failures),
    }
    (args.output_dir / "experiment.json").write_text(
        json.dumps(experiment, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(experiment, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--audio-root", type=Path,
        help="Required only when --conditions includes original.",
    )
    parser.add_argument("--reconstruction-index", type=Path, required=True)
    parser.add_argument("--reconstruction-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--conditions", default="auto",
        help="Comma-separated original/kN values, or auto to discover reconstruction depths",
    )
    parser.add_argument("--split", choices=("all", "train", "valid", "test"), default="all")
    parser.add_argument("--speakers", default="", help="Optional comma-separated speaker IDs")
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--language", default="en")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--skip-errors", action="store_true")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--limit-per-condition", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.beam_size < 1 or args.log_every < 1:
        parser.error("--beam-size and --log-every must be positive")
    if args.limit_per_condition < 0:
        parser.error("--limit-per-condition cannot be negative")
    return args


if __name__ == "__main__":
    evaluate(parse_args())

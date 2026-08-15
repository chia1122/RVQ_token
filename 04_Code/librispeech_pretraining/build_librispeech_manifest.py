#!/usr/bin/env python3
"""Build a JSONL manifest from official LibriSpeech *.trans.txt files."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).upper()
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'").replace("-", " ")
    text = re.sub(r"[^A-Z' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_subsets(value: str) -> dict[str, str]:
    result = {}
    for item in value.split(","):
        if not item.strip():
            continue
        try:
            subset, split = (part.strip() for part in item.split(":", 1))
        except ValueError as exc:
            raise ValueError("Subsets must use subset:split syntax") from exc
        if split not in {"train", "valid", "test"}:
            raise ValueError(f"Invalid split {split!r} for {subset!r}")
        result[subset] = split
    if not result:
        raise ValueError("No subsets configured")
    return result


def build(args: argparse.Namespace) -> None:
    subsets = parse_subsets(args.subsets)
    rows, failures = [], []
    seen = set()
    for subset, split in subsets.items():
        subset_root = args.audio_root / subset
        if not subset_root.is_dir():
            raise FileNotFoundError(f"LibriSpeech subset not found: {subset_root}")
        transcript_files = sorted(subset_root.rglob("*.trans.txt"))
        if not transcript_files:
            raise FileNotFoundError(f"No *.trans.txt files under {subset_root}")
        for transcript_path in transcript_files:
            with transcript_path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    utt_id, separator, text_raw = line.partition(" ")
                    audio_path = transcript_path.parent / f"{utt_id}.flac"
                    text_norm = normalize_text(text_raw)
                    if not separator or not text_norm or not audio_path.is_file():
                        failures.append({
                            "source": str(transcript_path), "line": line_number,
                            "utt_id": utt_id, "reason": "invalid_transcript_or_missing_audio",
                        })
                        continue
                    if utt_id in seen:
                        raise ValueError(f"Duplicate utterance ID: {utt_id}")
                    seen.add(utt_id)
                    parts = utt_id.split("-")
                    if len(parts) < 3:
                        raise ValueError(f"Unexpected LibriSpeech utterance ID: {utt_id}")
                    rows.append({
                        "utt_id": utt_id,
                        "audio_path": audio_path.relative_to(args.audio_root).as_posix(),
                        "audio_status": "available",
                        "text_raw": text_raw,
                        "text_norm": text_norm,
                        "speaker_id": parts[0],
                        "chapter_id": parts[1],
                        "speaker_type": "control",
                        "severity": "normal",
                        "corpus": "LibriSpeech",
                        "subset": subset,
                        "split": split,
                    })
    rows.sort(key=lambda row: row["utt_id"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "librispeech_all.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    for split in ("train", "valid", "test"):
        with (args.output_dir / f"librispeech_{split}.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                if row["split"] == split:
                    handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    with (args.output_dir / "failures.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in failures:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    counts = Counter(row["split"] for row in rows)
    summary = {
        "audio_root": str(args.audio_root.resolve()), "subsets": subsets,
        "utterances": len(rows), "speakers": len({row["speaker_id"] for row in rows}),
        "train": counts["train"], "valid": counts["valid"], "test": counts["test"],
        "failures": len(failures), "transcript_source": "official *.trans.txt",
    }
    (args.output_dir / "manifest_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--subsets",
        default="train-clean-100:train,dev-clean:valid,test-clean:test",
    )
    args = parser.parse_args()
    try:
        parse_subsets(args.subsets)
    except ValueError as exc:
        parser.error(str(exc))
    return args


if __name__ == "__main__":
    build(parse_args())

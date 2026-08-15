#!/usr/bin/env python3
"""Build validated, speaker-independent TORGO JSONL manifests."""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
import wave
from collections import Counter, defaultdict
from pathlib import Path


SPECIAL_UNINTELLIGIBLE = re.compile(
    r"(?:\[|<)?\s*(?:UNINTELLIGIBLE|UNINTEL|INAUDIBLE|UNKNOWN|XXX)\s*(?:\]|>)?",
    re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).upper()
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'")
    text = text.replace("-", " ")
    text = re.sub(r"[^A-Z' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_speaker_metadata(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"speaker_id", "gender", "speaker_type", "severity", "severity_source"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Speaker metadata must contain: {sorted(required)}")
    result = {}
    for row in rows:
        speaker = row["speaker_id"].strip()
        if not speaker or speaker in result:
            raise ValueError(f"Missing or duplicate speaker_id: {speaker!r}")
        result[speaker] = {key: (value or "").strip() for key, value in row.items()}
    return result


def is_included(metadata: dict[str, str]) -> bool:
    value = metadata.get("include_in_experiment", "true").strip().lower()
    if value not in {"true", "false"}:
        raise ValueError(f"include_in_experiment must be true or false, found {value!r}")
    return value == "true"


def load_splits(path: Path) -> tuple[dict[str, str], dict]:
    config = json.loads(path.read_text(encoding="utf-8"))
    assignment = {}
    for split in ("train", "valid", "test"):
        if split not in config or not isinstance(config[split], list):
            raise ValueError(f"Split config is missing list: {split}")
        for speaker in config[split]:
            if speaker in assignment:
                raise ValueError(f"Speaker appears in multiple splits: {speaker}")
            assignment[speaker] = split
    return assignment, config


def wav_metadata(path: Path) -> tuple[int, int, float]:
    with wave.open(str(path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        num_samples = wav_file.getnframes()
        channels = wav_file.getnchannels()
    if channels != 1:
        raise ValueError(f"Expected mono WAV, found {channels} channels")
    return sample_rate, num_samples, num_samples / sample_rate


def audio_relative_path(raw_path: str) -> Path:
    cleaned = raw_path.strip().replace("\\", "/").lstrip("/")
    if not cleaned or ".." in Path(cleaned).parts:
        raise ValueError(f"Unsafe or empty audio path: {raw_path!r}")
    return Path(cleaned)


def detect_microphone(raw_path: str) -> str:
    match = re.search(r"/wav_([^/]+)/", raw_path.replace("\\", "/"), re.IGNORECASE)
    return match.group(1) if match else "unknown"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_speaker_metadata(args.speaker_metadata)
    speaker_to_split, split_config = load_splits(args.split_config)

    if set(metadata) != set(speaker_to_split):
        missing_split = sorted(set(metadata) - set(speaker_to_split))
        missing_metadata = sorted(set(speaker_to_split) - set(metadata))
        raise ValueError(
            f"Speaker metadata/split mismatch; missing split={missing_split}, "
            f"missing metadata={missing_metadata}"
        )

    manifests: dict[str, list[dict]] = defaultdict(list)
    excluded: list[dict] = []
    seen_utt_ids = set()
    seen_audio_paths = set()

    with args.index.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"session", "audio", "text", "speaker_id"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Input index must contain: {sorted(required)}")

        for line_number, source in enumerate(reader, start=2):
            microphone = detect_microphone(source["audio"])
            if microphone.lower() != args.microphone.lower():
                continue

            speaker = source["speaker_id"].strip()
            raw_text = source["text"].strip()
            base_exclusion = {
                "line_number": line_number,
                "utt_id": source["session"].strip(),
                "audio": source["audio"].strip(),
                "text_raw": raw_text,
            }
            if speaker not in metadata:
                excluded.append({**base_exclusion, "reason": "unknown_speaker"})
                continue
            if not is_included(metadata[speaker]):
                excluded.append({**base_exclusion, "reason": "speaker_excluded_by_protocol"})
                continue
            if not raw_text:
                excluded.append({**base_exclusion, "reason": "empty_transcript"})
                continue
            if SPECIAL_UNINTELLIGIBLE.search(raw_text):
                excluded.append({**base_exclusion, "reason": "unintelligible_transcript"})
                continue
            if re.fullmatch(r"\s*\?+\s*", raw_text):
                excluded.append({**base_exclusion, "reason": "unintelligible_transcript"})
                continue
            if re.search(r"\d", raw_text):
                excluded.append({**base_exclusion, "reason": "numeric_transcript_requires_policy"})
                continue
            normalized = normalize_text(raw_text)
            if not normalized:
                excluded.append({**base_exclusion, "reason": "empty_after_normalization"})
                continue

            utt_id = source["session"].strip()
            relative_audio = audio_relative_path(source["audio"])
            relative_audio_posix = relative_audio.as_posix()
            if utt_id in seen_utt_ids:
                raise ValueError(f"Duplicate utt_id: {utt_id}")
            if relative_audio_posix.lower() in seen_audio_paths:
                raise ValueError(f"Duplicate audio path: {relative_audio_posix}")
            seen_utt_ids.add(utt_id)
            seen_audio_paths.add(relative_audio_posix.lower())

            audio_path = (args.audio_root / relative_audio).resolve()
            if audio_path.exists():
                try:
                    sample_rate, num_samples, duration = wav_metadata(audio_path)
                except (wave.Error, ValueError) as exc:
                    excluded.append({**base_exclusion, "reason": f"invalid_wav:{exc}"})
                    continue
                audio_status = "available"
            elif args.allow_missing_audio:
                sample_rate, num_samples, duration = None, None, None
                audio_status = "missing"
            else:
                excluded.append({**base_exclusion, "reason": "missing_audio"})
                continue

            split = speaker_to_split[speaker]
            session_match = re.search(r"Session\d+", utt_id, re.IGNORECASE)
            row = {
                "utt_id": utt_id,
                "audio_path": relative_audio_posix,
                "audio_status": audio_status,
                "text_raw": raw_text,
                "text_norm": normalized,
                "speaker_id": speaker,
                "session_id": session_match.group(0) if session_match else "unknown",
                "gender": metadata[speaker]["gender"],
                "speaker_type": metadata[speaker]["speaker_type"],
                "severity": metadata[speaker]["severity"] or "unknown",
                "severity_source": metadata[speaker]["severity_source"] or "unknown",
                "microphone": microphone,
                "sample_rate": sample_rate,
                "num_samples": num_samples,
                "duration": duration,
                "corpus": "TORGO",
                "split": split,
            }
            manifests[split].append(row)

    all_rows = manifests["train"] + manifests["valid"] + manifests["test"]
    if not all_rows:
        raise ValueError("No manifest rows were generated")

    actual_speakers = {row["speaker_id"] for row in all_rows}
    expected_speakers = {speaker for speaker, values in metadata.items() if is_included(values)}
    missing_index = sorted(expected_speakers - actual_speakers)
    if missing_index:
        raise ValueError(f"Configured speakers have no selected recordings: {missing_index}")

    split_sets = {name: {row["speaker_id"] for row in rows} for name, rows in manifests.items()}
    if split_sets["train"] & split_sets["valid"] or split_sets["train"] & split_sets["test"] or split_sets["valid"] & split_sets["test"]:
        raise AssertionError("Speaker leakage detected")

    for split in ("train", "valid", "test"):
        manifests[split].sort(key=lambda row: row["utt_id"])
        write_jsonl(output_dir / f"torgo_{split}.jsonl", manifests[split])
    all_rows.sort(key=lambda row: row["utt_id"])
    write_jsonl(output_dir / "torgo_all.jsonl", all_rows)
    write_csv(
        output_dir / "excluded_samples.csv",
        excluded,
        ["line_number", "utt_id", "audio", "text_raw", "reason"],
    )

    stats = []
    for split in ("train", "valid", "test", "all"):
        rows = all_rows if split == "all" else manifests[split]
        durations = [row["duration"] for row in rows if row["duration"] is not None]
        speakers = {row["speaker_id"] for row in rows}
        severity_speakers = Counter()
        type_speakers = Counter()
        for speaker in speakers:
            severity_speakers[metadata[speaker]["severity"] or "unknown"] += 1
            type_speakers[metadata[speaker]["speaker_type"]] += 1
        stats.append({
            "split": split,
            "speakers": len(speakers),
            "utterances": len(rows),
            "hours": round(sum(durations) / 3600, 4) if durations else "",
            "missing_audio": sum(row["audio_status"] == "missing" for row in rows),
            "dysarthric_speakers": type_speakers["dysarthric"],
            "control_speakers": type_speakers["control"],
            "mild_speakers": severity_speakers["mild"],
            "moderate_speakers": severity_speakers["moderate"],
            "moderate_to_severe_speakers": severity_speakers["moderate-to-severe"],
            "severe_speakers": severity_speakers["severe"],
            "unknown_severity_speakers": severity_speakers["unknown"],
        })
    write_csv(output_dir / "dataset_statistics.csv", stats, list(stats[0]))

    audit = {
        "index": str(args.index.resolve()),
        "audio_root": str(args.audio_root.resolve()),
        "microphone": args.microphone,
        "allow_missing_audio": args.allow_missing_audio,
        "normalization": "NFKC; uppercase; apostrophe normalization; hyphen-to-space; keep A-Z/apostrophe/space",
        "split_config": split_config,
        "rows_in_manifest": len(all_rows),
        "excluded_rows": len(excluded),
        "speaker_leakage": False,
    }
    (output_dir / "build_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(all_rows)} rows to {output_dir}")
    print(f"Excluded {len(excluded)} selected-microphone rows")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True, help="Existing TORGO CSV index")
    parser.add_argument("--audio-root", type=Path, required=True, help="Root containing F01/, M01/, etc.")
    parser.add_argument("--speaker-metadata", type=Path, required=True)
    parser.add_argument("--split-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--microphone", default="headMic")
    parser.add_argument("--allow-missing-audio", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())

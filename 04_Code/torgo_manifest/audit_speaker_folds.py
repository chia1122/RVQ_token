#!/usr/bin/env python3
"""Validate speaker folds and audit optional TORGO manifest coverage."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

from build_torgo_manifest import is_included, load_speaker_metadata, speaker_condition


REQUIRED_MANIFEST_FIELDS = {
    "speaker_id",
    "condition",
    "severity",
    "text_norm",
}


def load_fold_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Fold config must be a JSON object")
    if config.get("validation_strategy") != "next_fold_cyclic":
        raise ValueError("validation_strategy must be 'next_fold_cyclic'")
    folds = config.get("folds")
    if not isinstance(folds, list) or len(folds) < 3:
        raise ValueError("Fold config must contain at least three folds")
    return config


def validate_folds(metadata: dict[str, dict[str, str]], config: dict) -> list[dict]:
    folds = config["folds"]
    fold_ids: list[str] = []
    assigned: list[str] = []
    normalized_folds: list[dict] = []

    for fold in folds:
        if not isinstance(fold, dict):
            raise ValueError("Every fold must be a JSON object")
        fold_id = str(fold.get("fold_id", "")).strip()
        speakers = fold.get("speakers")
        if not fold_id or fold_id in fold_ids:
            raise ValueError(f"Missing or duplicate fold_id: {fold_id!r}")
        if not isinstance(speakers, list) or not speakers:
            raise ValueError(f"Fold {fold_id} must contain a non-empty speakers list")
        cleaned = [str(speaker).strip() for speaker in speakers]
        if any(not speaker for speaker in cleaned):
            raise ValueError(f"Fold {fold_id} contains an empty speaker ID")
        fold_ids.append(fold_id)
        assigned.extend(cleaned)
        normalized_folds.append({"fold_id": fold_id, "speakers": cleaned})

    duplicates = sorted(
        speaker for speaker, count in Counter(assigned).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"Speakers appear in multiple folds: {duplicates}")

    known = set(metadata)
    assigned_set = set(assigned)
    included = {
        speaker for speaker, values in metadata.items() if is_included(values)
    }
    unknown = sorted(assigned_set - known)
    missing = sorted(included - assigned_set)
    inactive = sorted(assigned_set - included)
    if unknown or missing or inactive:
        raise ValueError(
            "Fold/metadata mismatch; "
            f"unknown={unknown}, missing_included={missing}, "
            f"assigned_but_excluded={inactive}"
        )

    for fold in normalized_folds:
        conditions = {
            speaker_condition(metadata[speaker]) for speaker in fold["speakers"]
        }
        if conditions != {"control", "dysarthric"}:
            raise ValueError(
                f"Fold {fold['fold_id']} must contain control and dysarthric "
                f"speakers; found {sorted(conditions)}"
            )

    required = config.get("required_speaker_metadata", {})
    if not isinstance(required, dict):
        raise ValueError("required_speaker_metadata must be a JSON object")
    for speaker, expected in required.items():
        if speaker not in metadata:
            raise ValueError(f"Required speaker is missing from metadata: {speaker}")
        if not isinstance(expected, dict):
            raise ValueError(f"Required metadata for {speaker} must be an object")
        for field, expected_value in expected.items():
            actual = metadata[speaker].get(field, "")
            if actual.strip().lower() != str(expected_value).strip().lower():
                raise ValueError(
                    f"Required metadata mismatch for {speaker}.{field}: "
                    f"expected {expected_value!r}, found {actual!r}"
                )

    return normalized_folds


def build_rotations(folds: list[dict]) -> list[dict]:
    all_speakers = {
        speaker for fold in folds for speaker in fold["speakers"]
    }
    rotations: list[dict] = []
    for index, test_fold in enumerate(folds):
        valid_fold = folds[(index + 1) % len(folds)]
        test_speakers = set(test_fold["speakers"])
        valid_speakers = set(valid_fold["speakers"])
        train_speakers = all_speakers - test_speakers - valid_speakers
        if (
            test_speakers & valid_speakers
            or test_speakers & train_speakers
            or valid_speakers & train_speakers
        ):
            raise AssertionError("Speaker leakage detected while building rotations")
        if train_speakers | valid_speakers | test_speakers != all_speakers:
            raise AssertionError("Rotation does not cover every included speaker")
        rotations.append({
            "rotation_id": f"rotation_{index + 1:02d}_test_{test_fold['fold_id'].lower()}",
            "train_folds": [
                fold["fold_id"]
                for fold in folds
                if fold["fold_id"] not in {
                    test_fold["fold_id"], valid_fold["fold_id"]
                }
            ],
            "valid_fold": valid_fold["fold_id"],
            "test_fold": test_fold["fold_id"],
            "train": sorted(train_speakers),
            "valid": sorted(valid_speakers),
            "test": sorted(test_speakers),
        })

    test_counts = Counter(
        speaker for rotation in rotations for speaker in rotation["test"]
    )
    valid_counts = Counter(
        speaker for rotation in rotations for speaker in rotation["valid"]
    )
    if set(test_counts.values()) != {1} or set(valid_counts.values()) != {1}:
        raise AssertionError("Every speaker must be test and validation exactly once")
    return rotations


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on manifest line {line_number}") from exc
            missing = REQUIRED_MANIFEST_FIELDS - set(row)
            if missing:
                raise ValueError(
                    f"Manifest line {line_number} is missing fields: {sorted(missing)}"
                )
            rows.append(row)
    if not rows:
        raise ValueError("Manifest contains no rows")
    return rows


def validate_manifest(
    rows: list[dict], metadata: dict[str, dict[str, str]]
) -> None:
    included = {
        speaker for speaker, values in metadata.items() if is_included(values)
    }
    actual = {str(row["speaker_id"]) for row in rows}
    missing = sorted(included - actual)
    unknown = sorted(actual - included)
    if missing or unknown:
        raise ValueError(
            f"Manifest speaker mismatch; missing_included={missing}, "
            f"unknown_or_excluded={unknown}"
        )
    for row in rows:
        speaker = str(row["speaker_id"])
        expected_condition = speaker_condition(metadata[speaker])
        expected_severity = metadata[speaker]["severity"] or "unknown"
        if row["condition"] != expected_condition:
            raise ValueError(
                f"Manifest condition mismatch for {speaker}: "
                f"expected {expected_condition!r}, found {row['condition']!r}"
            )
        if row["severity"] != expected_severity:
            raise ValueError(
                f"Manifest severity mismatch for {speaker}: "
                f"expected {expected_severity!r}, found {row['severity']!r}"
            )


def _speaker_counter(
    speakers: Iterable[str], metadata: dict[str, dict[str, str]], field: str
) -> dict[str, int]:
    return dict(sorted(Counter(metadata[speaker][field] for speaker in speakers).items()))


def audit_manifest(
    rows: list[dict],
    rotations: list[dict],
    metadata: dict[str, dict[str, str]],
) -> tuple[list[dict], list[dict]]:
    statistics: list[dict] = []
    overlaps: list[dict] = []

    for rotation in rotations:
        rows_by_role: dict[str, list[dict]] = {}
        texts_by_role: dict[str, set[str]] = {}
        for role in ("train", "valid", "test"):
            speakers = rotation[role]
            speaker_set = set(speakers)
            role_rows = [row for row in rows if row["speaker_id"] in speaker_set]
            if not role_rows:
                raise ValueError(
                    f"{rotation['rotation_id']} has no manifest rows for role={role}"
                )
            rows_by_role[role] = role_rows
            texts_by_role[role] = {str(row["text_norm"]) for row in role_rows}
            durations = [
                float(row["duration"])
                for row in role_rows
                if row.get("duration") is not None
            ]
            statistics.append({
                "rotation_id": rotation["rotation_id"],
                "role": role,
                "folds": ",".join(
                    rotation["train_folds"]
                    if role == "train"
                    else [rotation[f"{role}_fold"]]
                ),
                "speakers": ",".join(speakers),
                "speaker_count": len(speakers),
                "utterances": len(role_rows),
                "hours": round(sum(durations) / 3600, 6) if durations else "",
                "unique_texts": len(texts_by_role[role]),
                "missing_audio": sum(
                    row.get("audio_status") == "missing" for row in role_rows
                ),
                "condition_speakers": json.dumps(
                    _speaker_counter(speakers, metadata, "speaker_type"),
                    sort_keys=True,
                ),
                "severity_speakers": json.dumps(
                    _speaker_counter(speakers, metadata, "severity"),
                    sort_keys=True,
                ),
                "gender_speakers": json.dumps(
                    _speaker_counter(speakers, metadata, "gender"),
                    sort_keys=True,
                ),
            })

        train_texts = texts_by_role["train"]
        valid_texts = texts_by_role["valid"]
        test_texts = texts_by_role["test"]
        test_seen = sum(
            str(row["text_norm"]) in train_texts for row in rows_by_role["test"]
        )
        test_total = len(rows_by_role["test"])
        overlaps.append({
            "rotation_id": rotation["rotation_id"],
            "train_unique_texts": len(train_texts),
            "valid_unique_texts": len(valid_texts),
            "test_unique_texts": len(test_texts),
            "train_valid_shared_unique": len(train_texts & valid_texts),
            "train_test_shared_unique": len(train_texts & test_texts),
            "valid_test_shared_unique": len(valid_texts & test_texts),
            "test_utterances": test_total,
            "test_seen_prompt_utterances": test_seen,
            "test_unseen_prompt_utterances": test_total - test_seen,
            "test_seen_prompt_ratio": test_seen / test_total,
        })
    return statistics, overlaps


def load_exclusion_counts(path: Path | None) -> dict[str, int] | None:
    if path is None:
        return None
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if rows and "reason" not in rows[0]:
        raise ValueError("Excluded-samples CSV must contain a reason column")
    return dict(sorted(Counter(row["reason"] for row in rows).items()))


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare_output_dir(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def run_audit(args: argparse.Namespace) -> dict:
    metadata = load_speaker_metadata(args.speaker_metadata)
    config = load_fold_config(args.fold_config)
    folds = validate_folds(metadata, config)
    rotations = build_rotations(folds)
    manifest_rows: list[dict] | None = None
    if args.manifest is not None:
        manifest_rows = load_jsonl(args.manifest)
        validate_manifest(manifest_rows, metadata)
    exclusion_counts = load_exclusion_counts(args.excluded_samples)
    output_dir = prepare_output_dir(args.output_dir)

    generated_dir = output_dir / "generated_splits"
    generated_dir.mkdir()
    for rotation in rotations:
        write_json(
            generated_dir / f"{rotation['rotation_id']}.json",
            {
                "description": (
                    f"{config.get('protocol_id', 'speaker_folds')} "
                    f"{rotation['rotation_id']}"
                ),
                "train": rotation["train"],
                "valid": rotation["valid"],
                "test": rotation["test"],
            },
        )

    statistics: list[dict] = []
    overlaps: list[dict] = []
    if manifest_rows is not None:
        statistics, overlaps = audit_manifest(manifest_rows, rotations, metadata)
        write_csv(output_dir / "fold_statistics.csv", statistics)
        write_csv(output_dir / "prompt_overlap.csv", overlaps)

    pending_citation_speakers = sorted(
        speaker
        for speaker, values in metadata.items()
        if is_included(values) and "todo" in values.get("severity_source", "").lower()
    )
    included_speakers = sorted(
        speaker for speaker, values in metadata.items() if is_included(values)
    )
    audit = {
        "protocol_id": config.get("protocol_id", "unknown"),
        "status": "valid",
        "validation_strategy": config["validation_strategy"],
        "prompt_overlap_policy": config.get("prompt_overlap_policy", "unspecified"),
        "manifest_audited": args.manifest is not None,
        "included_speakers": included_speakers,
        "included_speaker_count": len(included_speakers),
        "fold_count": len(folds),
        "rotation_count": len(rotations),
        "test_counts_by_speaker": dict(sorted(Counter(
            speaker for rotation in rotations for speaker in rotation["test"]
        ).items())),
        "valid_counts_by_speaker": dict(sorted(Counter(
            speaker for rotation in rotations for speaker in rotation["valid"]
        ).items())),
        "condition_speakers": _speaker_counter(
            included_speakers, metadata, "speaker_type"
        ),
        "severity_speakers": _speaker_counter(
            included_speakers, metadata, "severity"
        ),
        "gender_speakers": _speaker_counter(
            included_speakers, metadata, "gender"
        ),
        "pending_citation_detail_speakers": pending_citation_speakers,
        "exclusions_by_reason": exclusion_counts,
        "warnings": [
            "Full manifest coverage and prompt overlap were not audited."
        ] if args.manifest is None else [],
    }
    write_json(output_dir / "fold_audit.json", audit)
    print(json.dumps({
        "status": audit["status"],
        "protocol_id": audit["protocol_id"],
        "folds": audit["fold_count"],
        "rotations": audit["rotation_count"],
        "included_speakers": audit["included_speaker_count"],
        "manifest_audited": audit["manifest_audited"],
    }, indent=2, sort_keys=True))
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speaker-metadata", type=Path, required=True)
    parser.add_argument("--fold-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--excluded-samples", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        run_audit(parse_args())
    except (FileExistsError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None

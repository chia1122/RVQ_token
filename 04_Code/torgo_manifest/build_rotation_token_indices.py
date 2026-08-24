#!/usr/bin/env python3
"""Build speaker-rotation token indexes that share one master token store."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


REQUIRED_INDEX_FIELDS = {
    "utt_id",
    "token_path",
    "num_frames",
    "num_codebooks",
    "codebook_size",
    "speaker_id",
    "condition",
    "severity",
    "split",
    "text_norm",
}
SPLIT_ROLES = ("train", "valid", "test")
VALID_CONDITIONS = {"control", "dysarthric"}


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path.name} line {line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected an object in {path.name} line {line_number}"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"Token index is empty: {path}")
    return rows


def resolve_token_path(token_root: Path, token_path: object) -> Path:
    if not isinstance(token_path, str) or not token_path:
        raise ValueError("token_path must be a non-empty relative path")
    relative = Path(token_path)
    if relative.is_absolute():
        raise ValueError(f"Token path must be relative: {token_path}")
    resolved = (token_root / relative).resolve()
    try:
        resolved.relative_to(token_root)
    except ValueError as exc:
        raise ValueError(f"Token path escapes token root: {token_path}") from exc
    return resolved


def validate_master_index(rows: list[dict], token_root: Path) -> dict:
    token_root = token_root.resolve()
    utt_ids: set[str] = set()
    token_paths: set[str] = set()
    dimensions: set[tuple[int, int]] = set()
    speakers: set[str] = set()

    for line_number, row in enumerate(rows, start=1):
        missing = REQUIRED_INDEX_FIELDS - set(row)
        if missing:
            raise ValueError(
                f"Master token index line {line_number} is missing {sorted(missing)}"
            )
        utt_id = row["utt_id"]
        if not isinstance(utt_id, str) or not utt_id:
            raise ValueError(f"Invalid utt_id on line {line_number}")
        if utt_id in utt_ids:
            raise ValueError(f"Duplicate utt_id in master token index: {utt_id}")
        utt_ids.add(utt_id)

        token_path = row["token_path"]
        if token_path in token_paths:
            raise ValueError(f"Duplicate token_path in master token index: {token_path}")
        token_paths.add(token_path)
        resolved_token = resolve_token_path(token_root, token_path)
        if not resolved_token.is_file():
            raise FileNotFoundError(f"Token file not found: {token_path}")

        condition = row["condition"]
        if condition not in VALID_CONDITIONS:
            raise ValueError(
                f"Invalid condition for {utt_id}: {condition!r}; "
                f"expected one of {sorted(VALID_CONDITIONS)}"
            )
        speaker_id = row["speaker_id"]
        if not isinstance(speaker_id, str) or not speaker_id:
            raise ValueError(f"Invalid speaker_id for {utt_id}")
        speakers.add(speaker_id)
        try:
            dimensions.add((int(row["codebook_size"]), int(row["num_codebooks"])))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid token dimensions for {utt_id}") from exc

    if len(dimensions) != 1:
        raise ValueError(f"Inconsistent token dimensions: {sorted(dimensions)}")
    codebook_size, num_codebooks = next(iter(dimensions))
    if codebook_size < 1 or num_codebooks < 1:
        raise ValueError("Token dimensions must be positive")
    return {
        "utterance_count": len(rows),
        "speakers": speakers,
        "codebook_size": codebook_size,
        "num_codebooks": num_codebooks,
    }


def load_rotation_configs(directory: Path) -> list[tuple[str, dict]]:
    paths = sorted(directory.glob("rotation_*.json"))
    if not paths:
        raise ValueError(f"No rotation_*.json files found in {directory}")
    rotations: list[tuple[str, dict]] = []
    for path in paths:
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid rotation JSON: {path.name}") from exc
        if not isinstance(config, dict):
            raise ValueError(f"Rotation config must be an object: {path.name}")
        rotations.append((path.stem, config))
    return rotations


def speaker_split_map(
    rotation_id: str, config: dict, expected_speakers: set[str]
) -> dict[str, str]:
    assignment: dict[str, str] = {}
    for role in SPLIT_ROLES:
        speakers = config.get(role)
        if not isinstance(speakers, list) or not speakers:
            raise ValueError(
                f"{rotation_id} must define a non-empty {role} speaker list"
            )
        for speaker in speakers:
            if not isinstance(speaker, str) or not speaker:
                raise ValueError(f"{rotation_id} contains an invalid {role} speaker")
            if speaker in assignment:
                raise ValueError(
                    f"{rotation_id} assigns speaker {speaker} to multiple splits"
                )
            assignment[speaker] = role
    assigned = set(assignment)
    if assigned != expected_speakers:
        missing = sorted(expected_speakers - assigned)
        unknown = sorted(assigned - expected_speakers)
        raise ValueError(
            f"{rotation_id} speaker coverage mismatch: "
            f"missing={missing}, unknown={unknown}"
        )
    return assignment


def build_rotation_rows(
    master_rows: list[dict], assignment: dict[str, str]
) -> list[dict]:
    output_rows = []
    for row in master_rows:
        output_row = dict(row)
        output_row["split"] = assignment[row["speaker_id"]]
        output_rows.append(output_row)
    return output_rows


def summarize_rotation(rotation_id: str, rows: list[dict]) -> dict:
    speakers_by_role = {
        role: {row["speaker_id"] for row in rows if row["split"] == role}
        for role in SPLIT_ROLES
    }
    return {
        "rotation_id": rotation_id,
        "status": "valid",
        "missing_token_files": 0,
        "metadata_mismatches": 0,
        "utterances": {
            role: sum(row["split"] == role for row in rows)
            for role in SPLIT_ROLES
        },
        "speakers": {
            role: len(speakers_by_role[role]) for role in SPLIT_ROLES
        },
    }


def condition_speaker_counts(rows: list[dict]) -> dict[str, int]:
    values = {
        (row["condition"], row["speaker_id"])
        for row in rows
    }
    return dict(sorted(Counter(condition for condition, _ in values).items()))


def severity_speaker_counts(rows: list[dict]) -> dict[str, int]:
    values = {
        (row["severity"], row["speaker_id"])
        for row in rows
    }
    return dict(sorted(Counter(severity for severity, _ in values).items()))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def prepare_output_dir(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise FileExistsError(f"Output path is not a directory: {path}")
        if any(path.iterdir()):
            raise FileExistsError(
                f"Refusing to overwrite non-empty output directory: {path}"
            )


def build_indices(args: argparse.Namespace) -> dict:
    master_rows = load_jsonl(args.master_token_index)
    master = validate_master_index(master_rows, args.master_token_root)
    rotations = load_rotation_configs(args.generated_splits_dir)

    built: list[tuple[str, list[dict]]] = []
    for rotation_id, config in rotations:
        assignment = speaker_split_map(
            rotation_id, config, master["speakers"]
        )
        built.append((rotation_id, build_rotation_rows(master_rows, assignment)))

    prepare_output_dir(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rotation_summaries = []
    for rotation_id, rows in built:
        rotation_dir = args.output_dir / rotation_id
        rotation_dir.mkdir()
        write_jsonl(rotation_dir / "tokens.jsonl", rows)
        summary = summarize_rotation(rotation_id, rows)
        summary["condition_speakers"] = condition_speaker_counts(rows)
        summary["severity_speakers"] = severity_speaker_counts(rows)
        rotation_summaries.append(summary)

    audit = {
        "status": "valid",
        "master_index": args.master_token_index.name,
        "rotation_count": len(rotation_summaries),
        "utterance_count": master["utterance_count"],
        "speaker_count": len(master["speakers"]),
        "num_codebooks": master["num_codebooks"],
        "codebook_size": master["codebook_size"],
        "token_files_shared": True,
        "rotations": rotation_summaries,
    }
    (args.output_dir / "rotation_index_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-token-index", type=Path, required=True)
    parser.add_argument("--master-token-root", type=Path, required=True)
    parser.add_argument("--generated-splits-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = build_indices(args)
    print(json.dumps({
        "status": audit["status"],
        "rotations": audit["rotation_count"],
        "utterances_per_rotation": audit["utterance_count"],
        "speakers": audit["speaker_count"],
        "num_codebooks": audit["num_codebooks"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

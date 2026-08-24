#!/usr/bin/env python3
"""Aggregate completed speaker-fold RVQ sweeps without rerunning training."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


ROTATION_PATTERN = re.compile(r"rotation_(\d+)_test_(.+)")
LONG_FIELDS = (
    "protocol", "rotation", "test_fold", "codec", "representation_mode",
    "rvq_mode", "condition", "input_depth", "active_rvq_layers",
    "effective_fusion", "parameter_count", "trainable_parameter_count",
    "active_embedding_parameter_count", "non_embedding_parameter_count",
    "depth", "seed",
    "group_type", "group_value", "metric", "value",
)
SUMMARY_REPRESENTATION_FIELDS = ("representation_mode", "rvq_mode", "condition")
RATE_METRICS = (
    "wer", "cer", "substitution_rate", "deletion_rate", "insertion_rate",
    "empty_hypothesis_ratio", "ctc_blank_frame_ratio",
)


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read valid JSON: {path}") from exc


def read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    except OSError as exc:
        raise ValueError(f"Cannot read CSV: {path}") from exc


def discover_rotations(root: Path, expected_rotations: int = 7) -> list[tuple[int, str, Path]]:
    discovered: dict[int, tuple[str, Path]] = {}
    unexpected = []
    for path in root.glob("rotation_*_test_*"):
        if not path.is_dir():
            continue
        match = ROTATION_PATTERN.fullmatch(path.name)
        if not match:
            continue
        number = int(match.group(1))
        test_fold = match.group(2)
        if number == 0:
            continue
        if number < 1 or number > expected_rotations:
            unexpected.append(path.name)
            continue
        if number in discovered:
            raise ValueError(f"Duplicate rotation number {number}: {path}")
        discovered[number] = (test_fold, path)
    expected = set(range(1, expected_rotations + 1))
    missing = sorted(expected - set(discovered))
    if missing:
        raise ValueError(f"Missing formal rotations: {missing}")
    if unexpected:
        raise ValueError(f"Unexpected formal rotation directories: {sorted(unexpected)}")
    test_folds = [value[0] for value in discovered.values()]
    if len(test_folds) != len(set(test_folds)):
        raise ValueError(f"Duplicate test-fold labels: {sorted(test_folds)}")
    return [
        (number, discovered[number][0], discovered[number][1])
        for number in sorted(discovered)
    ]


def option(arguments: list[str], name: str) -> str | None:
    for index, argument in enumerate(arguments):
        if argument == name:
            if index + 1 >= len(arguments):
                raise ValueError(f"Missing value after {name}")
            return arguments[index + 1]
        if argument.startswith(name + "="):
            return argument.split("=", 1)[1]
    return None


def validate_rotation_configs(
    rotations: list[tuple[int, str, Path]], expected_selection: str,
) -> dict:
    reference = None
    for _, _, path in rotations:
        config_path = path / "sweep_config.json"
        config = read_json(config_path)
        comparable = {
            "codec": config.get("codec"),
            "depths": config.get("depths"),
            "seeds": config.get("seeds"),
            "train_args": config.get("train_args"),
            "representation_mode": config.get("representation_mode", "discrete_learned"),
            "rvq_mode": config.get("rvq_mode", "cumulative"),
        }
        if not isinstance(comparable["depths"], list) or not comparable["depths"]:
            raise ValueError(f"Invalid depths in {config_path}")
        if not isinstance(comparable["seeds"], list) or not comparable["seeds"]:
            raise ValueError(f"Invalid seeds in {config_path}")
        if not isinstance(comparable["train_args"], list):
            raise ValueError(f"Invalid train_args in {config_path}")
        selection = option(comparable["train_args"], "--selection-metric")
        if selection != expected_selection:
            raise ValueError(
                f"Expected selection metric {expected_selection!r}, got {selection!r} "
                f"in {config_path}"
            )
        if reference is None:
            reference = comparable
        elif comparable != reference:
            differing = [key for key in comparable if comparable[key] != reference[key]]
            raise ValueError(f"Sweep configuration differs in {path.name}: {differing}")
    assert reference is not None
    return reference


def validate_run_rows(rotation_path: Path, config: dict) -> list[dict[str, str]]:
    rows = read_csv(rotation_path / "sweep_runs.csv")
    expected = {
        (int(depth), int(seed))
        for depth in config["depths"] for seed in config["seeds"]
    }
    observed = []
    for row in rows:
        try:
            observed.append((int(row["depth"]), int(row["seed"])))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid run row in {rotation_path / 'sweep_runs.csv'}") from exc
        if row.get("status") != "valid":
            raise ValueError(
                f"Non-valid run in {rotation_path.name}: depth={row.get('depth')} "
                f"seed={row.get('seed')} status={row.get('status')}"
            )
    if len(observed) != len(set(observed)):
        raise ValueError(f"Duplicate depth/seed run rows in {rotation_path.name}")
    if set(observed) != expected:
        missing = sorted(expected - set(observed))
        extra = sorted(set(observed) - expected)
        raise ValueError(
            f"Incomplete runs in {rotation_path.name}: missing={missing}, extra={extra}"
        )
    return rows


def integer_from_ratio(value: float, denominator: int, label: str) -> int:
    raw = value * denominator
    rounded = round(raw)
    if not math.isclose(raw, rounded, rel_tol=1e-9, abs_tol=1e-6):
        raise ValueError(f"Cannot reconstruct integer numerator for {label}: {raw}")
    return int(rounded)


def read_prediction_metadata(path: Path) -> dict[str, tuple[str, str]]:
    speakers: dict[str, tuple[str, str]] = {}
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                try:
                    speaker = str(row["speaker_id"])
                    value = (str(row["condition"]), str(row["severity"]))
                except KeyError as exc:
                    raise ValueError(
                        f"Missing prediction metadata on line {line_number}: {path}"
                    ) from exc
                previous = speakers.setdefault(speaker, value)
                if previous != value:
                    raise ValueError(f"Inconsistent metadata for speaker {speaker} in {path}")
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read predictions: {path}") from exc
    if not speakers:
        raise ValueError(f"No prediction metadata found: {path}")
    return speakers


def read_rotation_long(
    path: Path, protocol: str, rotation: int, test_fold: str, config: dict,
) -> list[dict]:
    rows = read_csv(path / "trajectory_long.csv")
    required = {"codec", "depth", "seed", "group_type", "group_value", "metric", "value"}
    output = []
    seen = set()
    expected_runs = {
        (int(depth), int(seed))
        for depth in config["depths"] for seed in config["seeds"]
    }
    observed_runs = set()
    for row in rows:
        missing = required - set(row)
        if missing:
            raise ValueError(f"Missing long-format fields {sorted(missing)} in {path}")
        try:
            depth, seed, value = int(row["depth"]), int(row["seed"]), float(row["value"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid long-format row in {path}") from exc
        if not math.isfinite(value):
            raise ValueError(f"Non-finite metric value in {path}")
        if row["codec"] != config["codec"]:
            raise ValueError(f"Codec mismatch in {path / 'trajectory_long.csv'}")
        key = (depth, seed, row["group_type"], row["group_value"], row["metric"])
        if key in seen:
            raise ValueError(f"Duplicate long-format key {key} in {path}")
        seen.add(key)
        observed_runs.add((depth, seed))
        rvq_mode = row.get("rvq_mode") or config.get("rvq_mode", "cumulative")
        condition = row.get("condition") or (
            f"individual_q{depth}" if rvq_mode == "individual"
            else "cumulative_q1" if depth == 1 else f"cumulative_q1_{depth}"
        )
        output.append({
            "protocol": protocol, "rotation": rotation, "test_fold": test_fold,
            "codec": row["codec"],
            "representation_mode": row.get("representation_mode")
            or config.get("representation_mode", "discrete_learned"),
            "rvq_mode": rvq_mode, "condition": condition,
            "input_depth": int(row.get("input_depth") or depth),
            "active_rvq_layers": row.get("active_rvq_layers") or (
                str(depth) if rvq_mode == "individual"
                else ",".join(str(layer) for layer in range(1, depth + 1))
            ),
            "effective_fusion": row.get("effective_fusion") or (
                "single_active_layer" if depth == 1 or rvq_mode == "individual"
                else "sqrt_normalized_sum"
            ),
            "parameter_count": row.get("parameter_count", ""),
            "trainable_parameter_count": row.get("trainable_parameter_count", ""),
            "active_embedding_parameter_count": row.get(
                "active_embedding_parameter_count", ""
            ),
            "non_embedding_parameter_count": row.get(
                "non_embedding_parameter_count", ""
            ),
            "depth": depth, "seed": seed,
            "group_type": row["group_type"], "group_value": row["group_value"],
            "metric": row["metric"], "value": value,
        })
    if observed_runs != expected_runs:
        raise ValueError(
            f"Long-format run coverage differs in {path.name}: "
            f"missing={sorted(expected_runs - observed_runs)}"
        )
    return output


def collect_results(
    root: Path, protocol: str, expected_rotations: int = 7,
    expected_selection: str = "cer",
) -> tuple[list[dict], list[dict], dict[str, tuple[str, str]], dict]:
    rotations = discover_rotations(root, expected_rotations)
    config = validate_rotation_configs(rotations, expected_selection)
    combined_long = []
    group_records = []
    speaker_metadata: dict[str, tuple[str, str]] = {}
    run_count = 0
    for rotation, test_fold, rotation_path in rotations:
        rows = validate_run_rows(rotation_path, config)
        if not (rotation_path / "trajectory_summary.csv").is_file():
            raise ValueError(f"Missing trajectory summary: {rotation_path}")
        combined_long.extend(read_rotation_long(
            rotation_path, protocol, rotation, test_fold, config
        ))
        for row in rows:
            depth, seed = int(row["depth"]), int(row["seed"])
            rvq_mode = config.get("rvq_mode", "cumulative")
            condition = (
                f"individual_q{depth}" if rvq_mode == "individual"
                else "cumulative_q1" if depth == 1 else f"cumulative_q1_{depth}"
            )
            run_name = condition if rvq_mode == "individual" else f"k{depth}"
            run_root = rotation_path / str(config["codec"]) / run_name / f"seed_{seed}"
            result_path = run_root / "results.json"
            result = read_json(result_path)
            groups = result.get("test", {}).get("groups")
            if not isinstance(groups, list) or not groups:
                raise ValueError(f"Missing test groups: {result_path}")
            seen_groups = set()
            for group in groups:
                try:
                    group_type = str(group["group_type"])
                    group_value = str(group["group_value"])
                except KeyError as exc:
                    raise ValueError(f"Invalid test group: {result_path}") from exc
                group_key = (group_type, group_value)
                if group_key in seen_groups:
                    raise ValueError(f"Duplicate group {group_key} in {result_path}")
                seen_groups.add(group_key)
                record = {
                    "protocol": protocol, "rotation": rotation,
                    "test_fold": test_fold, "codec": config["codec"],
                    "depth": depth, "seed": seed,
                    "representation_mode": result.get(
                        "representation_mode", config.get("representation_mode", "discrete_learned")
                    ),
                    "rvq_mode": result.get("rvq_mode", rvq_mode),
                    "condition": result.get("condition", condition),
                    "group_type": group_type, "group_value": group_value,
                    **group,
                }
                group_records.append(record)
            metadata = read_prediction_metadata(run_root / "test_predictions.jsonl")
            for speaker, value in metadata.items():
                previous = speaker_metadata.setdefault(speaker, value)
                if previous != value:
                    raise ValueError(f"Inconsistent cross-run metadata for speaker {speaker}")
            run_count += 1
    expected_runs = expected_rotations * len(config["depths"]) * len(config["seeds"])
    if run_count != expected_runs:
        raise ValueError(f"Expected {expected_runs} runs, collected {run_count}")
    audit = {
        "protocol": protocol,
        "trajectory_root": str(root.resolve()),
        "codec": config["codec"],
        "depths": config["depths"],
        "seeds": config["seeds"],
        "selection_metric": expected_selection,
        "representation_mode": config.get("representation_mode", "discrete_learned"),
        "rvq_mode": config.get("rvq_mode", "cumulative"),
        "rotations": expected_rotations,
        "runs": run_count,
        "speakers": len(speaker_metadata),
        "status": "valid",
    }
    return combined_long, group_records, speaker_metadata, audit


def summarize_rows(rows: list[dict], key_fields: tuple[str, ...]) -> list[dict]:
    grouped: dict[tuple, list[float]] = defaultdict(list)
    rotations: dict[tuple, set[int]] = defaultdict(set)
    reported_rotation_counts: dict[tuple, set[int]] = defaultdict(set)
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        grouped[key].append(float(row["value"]))
        if "rotation" in row:
            rotations[key].add(int(row["rotation"]))
        elif row.get("n_rotations") not in (None, ""):
            reported_rotation_counts[key].add(int(row["n_rotations"]))
    output = []
    for key in sorted(grouped):
        values = grouped[key]
        result = {field: value for field, value in zip(key_fields, key)}
        if len(reported_rotation_counts[key]) > 1:
            raise ValueError(f"Inconsistent n_rotations for summary key {key}")
        n_rotations = (
            len(rotations[key]) if rotations[key]
            else next(iter(reported_rotation_counts[key]), "")
        )
        result.update({
            "mean": statistics.fmean(values),
            "sd": statistics.stdev(values) if len(values) > 1 else "",
            "n_valid": len(values),
            "n_rotations": n_rotations,
        })
        output.append(result)
    return output


def pooled_micro_rows(group_records: list[dict]) -> list[dict]:
    totals: dict[tuple, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    rotations: dict[tuple, set[int]] = defaultdict(set)
    required = (
        "utterances", "reference_words", "reference_characters",
        "substitutions", "deletions", "insertions", "cer",
        "empty_hypothesis_ratio",
    )
    for group in group_records:
        missing = [field for field in required if field not in group]
        if missing:
            raise ValueError(
                f"Cannot compute pooled micro for {group['group_type']}/"
                f"{group['group_value']}; missing {missing}"
            )
        key = (
            group["protocol"], group["codec"],
            group["representation_mode"], group["rvq_mode"], group["condition"],
            int(group["depth"]), int(group["seed"]),
            group["group_type"], group["group_value"],
        )
        utterances = int(group["utterances"])
        reference_characters = int(group["reference_characters"])
        totals[key]["utterances"] += utterances
        totals[key]["reference_words"] += int(group["reference_words"])
        totals[key]["reference_characters"] += reference_characters
        for metric in ("substitutions", "deletions", "insertions"):
            totals[key][metric] += int(group[metric])
        totals[key]["character_edits"] += integer_from_ratio(
            float(group["cer"]), reference_characters, "CER"
        )
        totals[key]["empty_hypotheses"] += integer_from_ratio(
            float(group["empty_hypothesis_ratio"]), utterances, "empty hypothesis ratio"
        )
        rotations[key].add(int(group["rotation"]))
    output = []
    for key in sorted(totals):
        total = totals[key]
        word_edits = total["substitutions"] + total["deletions"] + total["insertions"]
        values = {
            "wer": (word_edits, total["reference_words"]),
            "cer": (total["character_edits"], total["reference_characters"]),
            "substitutions": (total["substitutions"], None),
            "deletions": (total["deletions"], None),
            "insertions": (total["insertions"], None),
            "substitution_rate": (total["substitutions"], total["reference_words"]),
            "deletion_rate": (total["deletions"], total["reference_words"]),
            "insertion_rate": (total["insertions"], total["reference_words"]),
            "empty_hypothesis_ratio": (total["empty_hypotheses"], total["utterances"]),
        }
        for metric, (numerator, denominator) in values.items():
            value = numerator if denominator is None else numerator / denominator if denominator else 0.0
            output.append({
                "protocol": key[0], "codec": key[1],
                "representation_mode": key[2], "rvq_mode": key[3], "condition": key[4],
                "depth": key[5], "seed": key[6],
                "group_type": key[7], "group_value": key[8], "metric": metric,
                "value": value, "numerator": numerator,
                "denominator": "" if denominator is None else denominator,
                "n_rotations": len(rotations[key]),
            })
    return output


def speaker_macro_rows(
    group_records: list[dict], speaker_metadata: dict[str, tuple[str, str]],
) -> list[dict]:
    speakers: dict[tuple, list[dict]] = defaultdict(list)
    for group in group_records:
        if group["group_type"] != "speaker":
            continue
        speaker = group["group_value"]
        if speaker not in speaker_metadata:
            raise ValueError(f"Missing metadata for speaker {speaker}")
        key = (
            group["protocol"], group["codec"], group["representation_mode"],
            group["rvq_mode"], group["condition"],
            int(group["depth"]), int(group["seed"]),
        )
        speakers[key].append(group)
    output = []
    for key in sorted(speakers):
        records = speakers[key]
        partitions = {
            ("speaker_macro", "all"): records,
        }
        for condition in sorted({speaker_metadata[row["group_value"]][0] for row in records}):
            partitions[("condition_speaker_macro", condition)] = [
                row for row in records if speaker_metadata[row["group_value"]][0] == condition
            ]
        for severity in sorted({speaker_metadata[row["group_value"]][1] for row in records}):
            partitions[("severity_speaker_macro", severity)] = [
                row for row in records if speaker_metadata[row["group_value"]][1] == severity
            ]
        for (group_type, group_value), partition in partitions.items():
            n_rotations = len({int(row["rotation"]) for row in partition})
            for metric in RATE_METRICS:
                values = [float(row[metric]) for row in partition if metric in row]
                if len(values) != len(partition):
                    continue
                output.append({
                    "protocol": key[0], "codec": key[1],
                    "representation_mode": key[2], "rvq_mode": key[3], "condition": key[4],
                    "depth": key[5], "seed": key[6],
                    "group_type": group_type, "group_value": group_value,
                    "metric": metric, "value": statistics.fmean(values),
                    "n_speakers": len(values),
                    "n_rotations": n_rotations,
                })
    return output


def write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare_output_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def aggregate(args: argparse.Namespace) -> dict:
    combined, groups, metadata, audit = collect_results(
        args.trajectory_root, args.protocol_id, args.expected_rotations,
        args.expected_selection,
    )
    run_summary = summarize_rows(
        combined,
        ("protocol", "codec", *SUMMARY_REPRESENTATION_FIELDS, "depth",
         "group_type", "group_value", "metric"),
    )
    pooled = pooled_micro_rows(groups)
    pooled_summary = summarize_rows(
        pooled,
        ("protocol", "codec", *SUMMARY_REPRESENTATION_FIELDS, "depth",
         "group_type", "group_value", "metric"),
    )
    speaker_macro = speaker_macro_rows(groups, metadata)
    speaker_macro_summary = summarize_rows(
        speaker_macro,
        ("protocol", "codec", *SUMMARY_REPRESENTATION_FIELDS, "depth",
         "group_type", "group_value", "metric"),
    )
    prepare_output_dir(args.output_dir)
    write_csv(args.output_dir / "trajectory_long.csv", combined, LONG_FIELDS)
    summary_fields = (
        "protocol", "codec", *SUMMARY_REPRESENTATION_FIELDS, "depth",
        "group_type", "group_value", "metric",
        "mean", "sd", "n_valid", "n_rotations",
    )
    write_csv(args.output_dir / "trajectory_run_summary.csv", run_summary, summary_fields)
    write_csv(
        args.output_dir / "trajectory_pooled_micro_by_seed.csv", pooled,
        (
            "protocol", "codec", *SUMMARY_REPRESENTATION_FIELDS,
            "depth", "seed", "group_type", "group_value",
            "metric", "value", "numerator", "denominator", "n_rotations",
        ),
    )
    write_csv(
        args.output_dir / "trajectory_pooled_micro_summary.csv", pooled_summary,
        summary_fields,
    )
    write_csv(
        args.output_dir / "trajectory_speaker_macro_by_seed.csv", speaker_macro,
        (
            "protocol", "codec", *SUMMARY_REPRESENTATION_FIELDS,
            "depth", "seed", "group_type", "group_value",
            "metric", "value", "n_speakers", "n_rotations",
        ),
    )
    write_csv(
        args.output_dir / "trajectory_speaker_macro_summary.csv", speaker_macro_summary,
        summary_fields,
    )
    audit.update({
        "long_rows": len(combined),
        "run_summary_rows": len(run_summary),
        "pooled_micro_rows": len(pooled),
        "speaker_macro_rows": len(speaker_macro),
        "notes": [
            "Run summaries are fold/run-macro means, not pooled micro metrics.",
            "Pooled micro metrics sum available numerators and denominators per seed.",
            "CTC blank-frame ratio is excluded from pooled micro because valid-frame "
            "denominators are not stored in results.json.",
            "WER and CER are ASR performance metrics, not clinical intelligibility.",
        ],
    })
    (args.output_dir / "aggregation_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    return audit


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--expected-rotations", type=int, default=7)
    parser.add_argument("--expected-selection", default="cer")
    args = parser.parse_args(argv)
    if args.expected_rotations < 1:
        parser.error("--expected-rotations must be positive")
    return args


def main() -> None:
    audit = aggregate(parse_args())
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

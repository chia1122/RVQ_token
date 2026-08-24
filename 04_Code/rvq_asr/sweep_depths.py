#!/usr/bin/env python3
"""Run and aggregate capacity-matched RVQ CTC depth trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


METRICS = (
    "wer", "cer", "substitutions", "deletions", "insertions",
    "substitution_rate", "deletion_rate", "insertion_rate",
    "empty_hypothesis_ratio", "ctc_blank_frame_ratio",
)
REPRESENTATION_FIELDS = (
    "representation_mode", "rvq_mode", "condition", "input_depth",
    "active_rvq_layers", "effective_fusion", "parameter_count",
    "trainable_parameter_count", "active_embedding_parameter_count",
    "non_embedding_parameter_count",
)
FORBIDDEN_TRAIN_ARGS = {
    "--token-index", "--token-root", "--output-dir", "--num-rvq-layers", "--seed",
    "--active-rvq-layers", "--layer-fusion",
}


@dataclass(frozen=True)
class RunSpec:
    codec: str
    depth: int
    seed: int
    output_dir: Path


def read_index_dimensions(path: Path) -> tuple[int, int]:
    dimensions = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                dimensions.add((int(row["codebook_size"]), int(row["num_codebooks"])))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid token dimensions on line {line_number}") from exc
    if not dimensions:
        raise ValueError("Token index is empty")
    if len(dimensions) != 1:
        raise ValueError(f"Inconsistent token dimensions: {sorted(dimensions)}")
    return next(iter(dimensions))


def parse_integer_list(value: str, name: str, minimum: int | None = None) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"{name} must be comma-separated integers") from exc
    if not values or len(values) != len(set(values)):
        raise ValueError(f"{name} must contain unique integers")
    if minimum is not None and min(values) < minimum:
        raise ValueError(f"{name} values must be at least {minimum}")
    return values


def resolve_depths(value: str, num_codebooks: int) -> list[int]:
    depths = (
        list(range(1, num_codebooks + 1))
        if value.strip().lower() == "auto"
        else sorted(parse_integer_list(value, "depths", minimum=1))
    )
    unavailable = [depth for depth in depths if depth > num_codebooks]
    if unavailable:
        raise ValueError(
            f"Requested depths {unavailable}, but token index has {num_codebooks} codebooks"
        )
    return depths


def validate_codec_name(codec: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", codec):
        raise ValueError("codec must contain only letters, numbers, dot, underscore, or hyphen")
    return codec


def plan_runs(
    output_root: Path, codec: str, depths: list[int], seeds: list[int], resume: bool,
) -> list[RunSpec]:
    codec = validate_codec_name(codec)
    runs = [
        RunSpec(codec, depth, seed, output_root / codec / f"k{depth}" / f"seed_{seed}")
        for depth in depths for seed in seeds
    ]
    destinations = [run.output_dir.resolve() for run in runs]
    if len(destinations) != len(set(destinations)):
        raise ValueError("Duplicate sweep output directories")
    collisions = [run.output_dir for run in runs if run.output_dir.exists()]
    if collisions and not resume:
        preview = ", ".join(str(path) for path in collisions[:3])
        raise FileExistsError(
            f"Refusing to reuse {len(collisions)} existing run directories: {preview}"
        )
    return runs


def validate_train_args(train_args: list[str]) -> None:
    for argument in train_args:
        option = argument.split("=", 1)[0]
        if option in FORBIDDEN_TRAIN_ARGS:
            raise ValueError(f"Sweep controls {option}; do not pass it after --")


def build_train_command(args: argparse.Namespace, run: RunSpec) -> list[str]:
    return [
        sys.executable, "-m", "rvq_asr.train_probe",
        "--token-index", str(args.token_index),
        "--token-root", str(args.token_root),
        "--output-dir", str(run.output_dir),
        "--num-rvq-layers", str(run.depth),
        "--seed", str(run.seed),
        *args.train_args,
    ]


def legacy_groups(test_metrics: dict) -> list[dict]:
    groups = [{
        "group_type": "overall", "group_value": "all",
        **{metric: test_metrics[metric] for metric in METRICS if metric in test_metrics},
    }]
    for group_type in ("condition", "severity", "speaker"):
        wer_values = test_metrics.get(f"wer_by_{group_type}", {})
        cer_values = test_metrics.get(f"cer_by_{group_type}", {})
        for group_value in sorted(set(wer_values) | set(cer_values)):
            groups.append({
                "group_type": group_type, "group_value": group_value,
                "wer": wer_values.get(group_value), "cer": cer_values.get(group_value),
            })
    return groups


def collect_long_rows(runs: list[RunSpec]) -> tuple[list[dict], list[dict]]:
    long_rows, statuses = [], []
    for run in runs:
        result_path = run.output_dir / "results.json"
        if not result_path.is_file():
            statuses.append({
                "codec": run.codec, "depth": run.depth, "seed": run.seed,
                "status": "missing", "result_path": str(result_path),
            })
            continue
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            test_metrics = result["test"]
            groups = test_metrics.get("groups") or legacy_groups(test_metrics)
            representation = {}
            if result.get("representation_mode"):
                parameter_counts = result.get("parameter_counts", {})
                representation = {
                    "representation_mode": result["representation_mode"],
                    "rvq_mode": result.get("rvq_mode", ""),
                    "condition": result.get("condition", ""),
                    "input_depth": result.get("num_rvq_layers", run.depth),
                    "active_rvq_layers": ",".join(
                        str(layer) for layer in result.get("active_rvq_layers", [])
                    ),
                    "effective_fusion": result.get("effective_fusion", ""),
                    "parameter_count": parameter_counts.get("total", ""),
                    "trainable_parameter_count": parameter_counts.get("trainable", ""),
                    "active_embedding_parameter_count": parameter_counts.get(
                        "active_embedding", ""
                    ),
                    "non_embedding_parameter_count": parameter_counts.get(
                        "non_embedding", ""
                    ),
                }
            added = 0
            for group in groups:
                for metric in METRICS:
                    value = group.get(metric)
                    if isinstance(value, (int, float)) and math.isfinite(float(value)):
                        long_rows.append({
                            "codec": run.codec, "depth": run.depth, "seed": run.seed,
                            **representation,
                            "group_type": group["group_type"],
                            "group_value": group["group_value"],
                            "metric": metric, "value": value,
                        })
                        added += 1
            statuses.append({
                "codec": run.codec, "depth": run.depth, "seed": run.seed,
                **representation,
                "status": "valid" if added else "no_metrics",
                "result_path": str(result_path),
            })
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            statuses.append({
                "codec": run.codec, "depth": run.depth, "seed": run.seed,
                "status": f"invalid:{type(exc).__name__}",
                "result_path": str(result_path),
            })
    return long_rows, statuses


def summarize_long_rows(long_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[float]] = {}
    for row in long_rows:
        representation = tuple(row.get(field, "") for field in REPRESENTATION_FIELDS)
        key = (
            row["codec"], row["depth"], *representation, row["group_type"],
            row["group_value"], row["metric"],
        )
        grouped.setdefault(key, []).append(float(row["value"]))
    summary = []
    for key, values in sorted(grouped.items()):
        representation_values = key[2:2 + len(REPRESENTATION_FIELDS)]
        representation = (
            {field: value for field, value in zip(REPRESENTATION_FIELDS, representation_values)}
            if any(value != "" for value in representation_values) else {}
        )
        summary.append({
            "codec": key[0], "depth": key[1],
            **representation,
            "group_type": key[2 + len(REPRESENTATION_FIELDS)],
            "group_value": key[3 + len(REPRESENTATION_FIELDS)],
            "metric": key[4 + len(REPRESENTATION_FIELDS)],
            "mean": statistics.fmean(values),
            "sd": statistics.stdev(values) if len(values) > 1 else "",
            "n_valid": len(values),
        })
    return summary


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_trajectory_outputs(output_root: Path, runs: list[RunSpec]) -> tuple[int, int]:
    long_rows, statuses = collect_long_rows(runs)
    summary_rows = summarize_long_rows(long_rows)
    include_representation = any(
        row.get("representation_mode") for row in long_rows
    )
    representation_fields = list(REPRESENTATION_FIELDS) if include_representation else []
    write_csv(
        output_root / "trajectory_long.csv", long_rows,
        ["codec", "depth", "seed", *representation_fields,
         "group_type", "group_value", "metric", "value"],
    )
    write_csv(
        output_root / "trajectory_summary.csv", summary_rows,
        ["codec", "depth", *representation_fields,
         "group_type", "group_value", "metric", "mean", "sd", "n_valid"],
    )
    write_csv(
        output_root / "sweep_runs.csv", statuses,
        ["codec", "depth", "seed", *representation_fields, "status", "result_path"],
    )
    return len(long_rows), len(summary_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-index", type=Path, required=True)
    parser.add_argument("--token-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--codec", required=True)
    parser.add_argument("--depths", default="auto", help="auto or comma-separated depths")
    parser.add_argument("--seeds", default="1337", help="Comma-separated integer seeds")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("train_args", nargs=argparse.REMAINDER, help="Arguments passed after -- to train_probe")
    args = parser.parse_args()
    if args.train_args[:1] == ["--"]:
        args.train_args = args.train_args[1:]
    validate_train_args(args.train_args)
    return args


def main(args: argparse.Namespace) -> None:
    _, num_codebooks = read_index_dimensions(args.token_index)
    depths = resolve_depths(args.depths, num_codebooks)
    seeds = parse_integer_list(args.seeds, "seeds")
    runs = plan_runs(args.output_root, args.codec, depths, seeds, args.resume or args.aggregate_only)
    commands = [(run, build_train_command(args, run)) for run in runs]
    if args.dry_run:
        print(json.dumps([{"output_dir": str(run.output_dir), "command": command} for run, command in commands], indent=2))
        return
    args.output_root.mkdir(parents=True, exist_ok=True)
    config = {
        "codec": args.codec, "token_index": str(args.token_index.resolve()),
        "token_root": str(args.token_root.resolve()), "depths": depths,
        "seeds": seeds, "train_args": args.train_args,
    }
    config_path = args.output_root / "sweep_config.json"
    if config_path.is_file():
        existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        if existing_config != config:
            raise FileExistsError(
                f"Existing sweep_config.json does not match this sweep: {config_path}"
            )
        if not (args.resume or args.aggregate_only):
            raise FileExistsError(
                f"Sweep config already exists; use --resume after verifying it: {config_path}"
            )
    config_path.write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    if not args.aggregate_only:
        environment = os.environ.copy()
        code_root = str(Path(__file__).resolve().parents[1])
        environment["PYTHONPATH"] = os.pathsep.join(
            value for value in (code_root, environment.get("PYTHONPATH", "")) if value
        )
        for run, command in commands:
            if args.resume and (run.output_dir / "results.json").is_file():
                print(f"Skipping completed run: {run.output_dir}")
                continue
            run.output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Running codec={run.codec} depth={run.depth} seed={run.seed}", flush=True)
            completed = subprocess.run(command, env=environment, check=False)
            if completed.returncode and not args.continue_on_error:
                raise SystemExit(completed.returncode)
    long_count, summary_count = write_trajectory_outputs(args.output_root, runs)
    print(f"Wrote {long_count} long rows and {summary_count} summary rows")


if __name__ == "__main__":
    main(parse_args())

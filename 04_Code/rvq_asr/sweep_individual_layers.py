#!/usr/bin/env python3
"""Run matched individual RVQ-layer sweeps from completed cumulative configs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .aggregate_rotations import discover_rotations, option
from .sweep_depths import (
    RunSpec,
    parse_integer_list,
    read_index_dimensions,
    validate_codec_name,
    validate_train_args,
    write_trajectory_outputs,
)


FORBIDDEN_REFERENCE_ARGS = {
    "--representation-mode", "--rvq-mode", "--condition-name",
}


@dataclass(frozen=True)
class RotationPlan:
    rotation: int
    test_fold: str
    reference_dir: Path
    output_dir: Path
    token_index: Path
    token_root: Path
    codec: str
    layers: tuple[int, ...]
    seeds: tuple[int, ...]
    train_args: tuple[str, ...]
    reference_hash: str
    runs: tuple[RunSpec, ...]


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_values(value: str, available: list[int], name: str) -> list[int]:
    if value.strip().lower() == "auto":
        return list(available)
    selected = sorted(parse_integer_list(value, name, minimum=1))
    unavailable = sorted(set(selected) - set(available))
    if unavailable:
        raise ValueError(f"Requested {name} {unavailable}, available values are {available}")
    return selected


def validate_reference_train_args(arguments: list[str], expected_selection: str) -> None:
    validate_train_args(arguments)
    for argument in arguments:
        if argument.split("=", 1)[0] in FORBIDDEN_REFERENCE_ARGS:
            raise ValueError(f"Reference config cannot control {argument.split('=', 1)[0]}")
    selection = option(arguments, "--selection-metric")
    if selection != expected_selection:
        raise ValueError(
            f"Expected reference selection metric {expected_selection!r}, got {selection!r}"
        )


def build_plans(args: argparse.Namespace) -> list[RotationPlan]:
    rotations = discover_rotations(args.reference_trajectory_root, args.expected_rotations)
    requested_rotations = select_values(
        args.rotations, [number for number, _, _ in rotations], "rotations"
    )
    plans = []
    common = None
    destinations = []
    for rotation, test_fold, reference_dir in rotations:
        if rotation not in requested_rotations:
            continue
        config_path = reference_dir / "sweep_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        codec = validate_codec_name(str(config["codec"]))
        available_layers = [int(value) for value in config["depths"]]
        available_seeds = [int(value) for value in config["seeds"]]
        layers = select_values(args.layers, available_layers, "layers")
        seeds = select_values(args.seeds, available_seeds, "seeds")
        train_args = list(config["train_args"])
        validate_reference_train_args(train_args, args.expected_selection)
        comparable = (codec, available_layers, available_seeds, train_args)
        if common is None:
            common = comparable
        elif comparable != common:
            raise ValueError(f"Reference sweep configuration differs in {reference_dir.name}")
        token_index = Path(config["token_index"])
        token_root = Path(config["token_root"])
        _, num_codebooks = read_index_dimensions(token_index)
        if max(layers) > num_codebooks:
            raise ValueError(
                f"Requested Q{max(layers)}, but {token_index} has {num_codebooks} codebooks"
            )
        output_dir = args.output_root / reference_dir.name
        runs = tuple(
            RunSpec(
                codec, layer, seed,
                output_dir / codec / f"individual_q{layer}" / f"seed_{seed}",
            )
            for layer in layers for seed in seeds
        )
        destinations.extend(run.output_dir.resolve() for run in runs)
        plans.append(RotationPlan(
            rotation, test_fold, reference_dir, output_dir, token_index, token_root,
            codec, tuple(layers), tuple(seeds), tuple(train_args), file_hash(config_path), runs,
        ))
    if len(destinations) != len(set(destinations)):
        raise ValueError("Duplicate individual-layer output directories")
    collisions = [path for path in destinations if path.exists()]
    if collisions and not (args.resume or args.aggregate_only):
        preview = ", ".join(str(path) for path in collisions[:3])
        raise FileExistsError(
            f"Refusing to reuse {len(collisions)} existing run directories: {preview}"
        )
    return plans


def build_command(plan: RotationPlan, run: RunSpec) -> list[str]:
    layer = run.depth
    return [
        sys.executable, "-m", "rvq_asr.train_probe",
        "--token-index", str(plan.token_index),
        "--token-root", str(plan.token_root),
        "--output-dir", str(run.output_dir),
        "--num-rvq-layers", str(layer),
        "--active-rvq-layers", str(layer),
        "--representation-mode", "discrete_learned",
        "--rvq-mode", "individual",
        "--condition-name", f"individual_q{layer}",
        "--seed", str(run.seed),
        *plan.train_args,
    ]


def individual_config(plan: RotationPlan, protocol_id: str) -> dict:
    return {
        "protocol_id": protocol_id,
        "codec": plan.codec,
        "representation_mode": "discrete_learned",
        "rvq_mode": "individual",
        "conditions": [f"individual_q{layer}" for layer in plan.layers],
        "layers": list(plan.layers),
        "depths": list(plan.layers),
        "seeds": list(plan.seeds),
        "token_index": str(plan.token_index.resolve()),
        "token_root": str(plan.token_root.resolve()),
        "train_args": list(plan.train_args),
        "reference_sweep_config": str((plan.reference_dir / "sweep_config.json").resolve()),
        "reference_sweep_config_sha256": plan.reference_hash,
    }


def write_config(path: Path, config: dict, resume: bool) -> None:
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != config:
            raise FileExistsError(f"Existing individual sweep config differs: {path}")
        if not resume:
            raise FileExistsError(f"Sweep config already exists; use --resume: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-trajectory-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--expected-rotations", type=int, default=7)
    parser.add_argument("--expected-selection", default="cer")
    parser.add_argument("--rotations", default="auto")
    parser.add_argument("--layers", default="auto")
    parser.add_argument("--seeds", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args(argv)


def main(args: argparse.Namespace) -> None:
    plans = build_plans(args)
    commands = [
        (plan, run, build_command(plan, run))
        for plan in plans for run in plan.runs
    ]
    if args.dry_run:
        print(json.dumps([
            {
                "rotation": plan.rotation,
                "condition": f"individual_q{run.depth}",
                "layer": run.depth,
                "seed": run.seed,
                "output_dir": str(run.output_dir),
                "command": command,
            }
            for plan, run, command in commands
        ], indent=2))
        return
    environment = os.environ.copy()
    code_root = str(Path(__file__).resolve().parents[1])
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (code_root, environment.get("PYTHONPATH", "")) if value
    )
    for plan in plans:
        write_config(
            plan.output_dir / "sweep_config.json",
            individual_config(plan, args.protocol_id),
            args.resume or args.aggregate_only,
        )
        if not args.aggregate_only:
            for run in plan.runs:
                if args.resume and (run.output_dir / "results.json").is_file():
                    print(f"Skipping completed run: {run.output_dir}")
                    continue
                run.output_dir.mkdir(parents=True, exist_ok=True)
                print(
                    f"Running rotation={plan.rotation} individual_q{run.depth} "
                    f"seed={run.seed}", flush=True,
                )
                completed = subprocess.run(
                    build_command(plan, run), env=environment, check=False
                )
                if completed.returncode and not args.continue_on_error:
                    raise SystemExit(completed.returncode)
        long_count, summary_count = write_trajectory_outputs(plan.output_dir, list(plan.runs))
        print(
            f"{plan.output_dir.name}: wrote {long_count} long rows and "
            f"{summary_count} summary rows"
        )


if __name__ == "__main__":
    main(parse_args())

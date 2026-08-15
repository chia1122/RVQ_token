#!/usr/bin/env python3
"""Generate progress-report figures from RVQ ASR results.json files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


RUNS = {
    "AvgPool r=4": "encodec_k4_baseline",
    "AvgPool r=2": "encodec_k4_reduction2",
    "Conv1d r=4": "encodec_k4_conv",
}


def load_runs(runs_root: Path) -> dict[str, dict]:
    loaded = {}
    for label, directory in RUNS.items():
        path = runs_root / directory / "results.json"
        if path.is_file():
            loaded[label] = json.loads(path.read_text(encoding="utf-8"))
        else:
            print(f"Skipping missing run: {path}")
    if not loaded:
        raise SystemExit(f"No results.json files found under {runs_root}")
    return loaded


def metric(history_row: dict, split: str, name: str):
    values = history_row.get(split)
    return values.get(name) if values else None


def plot_curves(plt, runs: dict[str, dict], output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for label, result in runs.items():
        history = result["history"]
        epochs = [row["epoch"] for row in history]
        for axis, name, title in zip(
            axes,
            ("loss", "cer", "wer"),
            ("CTC loss", "Character error rate", "Word error rate"),
        ):
            valid = [metric(row, "valid", name) for row in history]
            axis.plot(epochs, valid, label=f"{label} valid")
            train = [metric(row, "train", name) for row in history]
            if all(value is not None for value in train):
                axis.plot(epochs, train, linestyle="--", label=f"{label} train")
            axis.set_title(title)
            axis.set_xlabel("Epoch")
            axis.grid(alpha=0.25)
    axes[0].set_ylabel("Metric value")
    handles, labels = axes[-1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    figure.tight_layout(rect=(0, 0.12, 1, 1))
    figure.savefig(output_dir / "training_curves.png", dpi=180)
    plt.close(figure)


def plot_comparison(plt, runs: dict[str, dict], output_dir: Path) -> None:
    labels = list(runs)
    train_cer = []
    valid_cer = []
    test_cer = []
    for result in runs.values():
        final = result["history"][-1]
        train_cer.append(metric(final, "train", "cer") or float("nan"))
        valid_cer.append(metric(final, "valid", "cer"))
        test_cer.append(result["test"]["cer"])

    positions = list(range(len(labels)))
    width = 0.25
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar([x - width for x in positions], train_cer, width, label="Train")
    axis.bar(positions, valid_cer, width, label="Validation")
    axis.bar([x + width for x in positions], test_cer, width, label="Test")
    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.set_ylabel("CER (lower is better)")
    axis.set_title("K=4 probe architecture comparison")
    axis.set_ylim(0, max(1.05, max(test_cer) * 1.1))
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "architecture_comparison.png", dpi=180)
    plt.close(figure)


def plot_diagnostics(plt, runs: dict[str, dict], output_dir: Path) -> None:
    labels = list(runs)
    length_ratio = [result["test"]["hypothesis_reference_length_ratio"] for result in runs.values()]
    blank_ratio = [result["test"]["ctc_blank_frame_ratio"] for result in runs.values()]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(labels, length_ratio)
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_title("Test output/reference length")
    axes[0].set_ylim(0, 1.05)
    axes[1].bar(labels, blank_ratio)
    axes[1].set_title("Test CTC blank-frame ratio")
    axes[1].set_ylim(0, 1.05)
    for axis in axes:
        axis.tick_params(axis="x", rotation=15)
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "sequence_diagnostics.png", dpi=180)
    plt.close(figure)


def write_summary(runs: dict[str, dict], output_dir: Path) -> None:
    fields = [
        "run", "best_epoch", "final_train_wer", "final_train_cer",
        "final_valid_wer", "final_valid_cer", "test_wer", "test_cer",
        "test_length_ratio", "test_empty_ratio", "test_blank_ratio",
    ]
    with (output_dir / "experiment_summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for label, result in runs.items():
            final = result["history"][-1]
            test = result["test"]
            writer.writerow({
                "run": label,
                "best_epoch": result["best_epoch"],
                "final_train_wer": metric(final, "train", "wer"),
                "final_train_cer": metric(final, "train", "cer"),
                "final_valid_wer": metric(final, "valid", "wer"),
                "final_valid_cer": metric(final, "valid", "cer"),
                "test_wer": test["wer"],
                "test_cer": test["cer"],
                "test_length_ratio": test["hypothesis_reference_length_ratio"],
                "test_empty_ratio": test["empty_hypothesis_ratio"],
                "test_blank_ratio": test["ctc_blank_frame_ratio"],
            })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("04_Code/rvq_asr/runs"))
    parser.add_argument("--output-dir", type=Path, default=Path("04_Code/rvq_asr/reports/figures"))
    args = parser.parse_args()
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("Install matplotlib to generate report figures") from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.runs_root)
    plot_curves(plt, runs, args.output_dir)
    plot_comparison(plt, runs, args.output_dir)
    plot_diagnostics(plt, runs, args.output_dir)
    write_summary(runs, args.output_dir)
    print(f"Generated report artifacts in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path

from rvq_asr.aggregate_rotations import (
    aggregate,
    discover_rotations,
    prepare_output_dir,
)


DEPTHS = list(range(1, 9))
SEEDS = [1337, 2026, 3407]


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def group_row(group_type, group_value, speakers, word_error):
    utterances = len(speakers)
    reference_words = utterances * 10
    reference_characters = utterances * 20
    substitutions = utterances * word_error
    deletions = utterances
    insertions = 0
    return {
        "group_type": group_type,
        "group_value": group_value,
        "utterances": utterances,
        "reference_words": reference_words,
        "reference_characters": reference_characters,
        "wer": (substitutions + deletions + insertions) / reference_words,
        "cer": (utterances * 2) / reference_characters,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "substitution_rate": substitutions / reference_words,
        "deletion_rate": deletions / reference_words,
        "insertion_rate": 0.0,
        "empty_hypothesis_ratio": 0.0,
        "ctc_blank_frame_ratio": 0.8,
    }


def build_fixture(root: Path) -> None:
    metadata = {}
    for rotation in range(1, 8):
        speakers = [
            (f"D{rotation}", "dysarthric", "severe"),
            (f"C{rotation}", "control", "control"),
        ]
        if rotation == 7:
            speakers.insert(1, ("D8", "dysarthric", "mild"))
        for speaker, condition, severity in speakers:
            metadata[speaker] = (condition, severity)
        rotation_root = root / f"rotation_{rotation:02d}_test_{chr(96 + rotation)}"
        config = {
            "codec": "speechtokenizer",
            "depths": DEPTHS,
            "seeds": SEEDS,
            "train_args": ["--selection-metric", "cer", "--batch-size", "8"],
        }
        write_json(rotation_root / "sweep_config.json", config)
        run_rows = []
        long_rows = []
        for depth in DEPTHS:
            for seed in SEEDS:
                run_root = rotation_root / "speechtokenizer" / f"k{depth}" / f"seed_{seed}"
                word_error = 1 + (depth + seed) % 2
                dysarthric = [row for row in speakers if row[1] == "dysarthric"]
                control = [row for row in speakers if row[1] == "control"]
                groups = [
                    group_row("overall", "all", speakers, word_error),
                    group_row("condition", "dysarthric", dysarthric, word_error),
                    group_row("condition", "control", control, word_error),
                ]
                for speaker, condition, severity in speakers:
                    groups.append(group_row("speaker", speaker, [(speaker, condition, severity)], word_error))
                for severity in sorted({row[2] for row in speakers}):
                    selected = [row for row in speakers if row[2] == severity]
                    groups.append(group_row("severity", severity, selected, word_error))
                write_json(run_root / "results.json", {"test": {"groups": groups}})
                for group in groups:
                    for metric in (
                        "wer", "cer", "substitutions", "deletions", "insertions",
                        "substitution_rate", "deletion_rate", "insertion_rate",
                        "empty_hypothesis_ratio", "ctc_blank_frame_ratio",
                    ):
                        long_rows.append({
                            "codec": "speechtokenizer", "depth": depth, "seed": seed,
                            "group_type": group["group_type"],
                            "group_value": group["group_value"],
                            "metric": metric, "value": group[metric],
                        })
                with (run_root / "test_predictions.jsonl").open(
                    "w", encoding="utf-8", newline="\n"
                ) as handle:
                    for speaker, condition, severity in speakers:
                        handle.write(json.dumps({
                            "utt_id": f"{speaker}-utt", "speaker_id": speaker,
                            "condition": condition, "severity": severity,
                            "reference": "TEST", "hypothesis": "TEST",
                        }) + "\n")
                run_rows.append({
                    "codec": "speechtokenizer", "depth": depth, "seed": seed,
                    "status": "valid", "result_path": str(run_root / "results.json"),
                })
        with (rotation_root / "sweep_runs.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("codec", "depth", "seed", "status", "result_path"),
            )
            writer.writeheader()
            writer.writerows(run_rows)
        with (rotation_root / "trajectory_long.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=(
                "codec", "depth", "seed", "group_type", "group_value", "metric", "value",
            ))
            writer.writeheader()
            writer.writerows(long_rows)
        (rotation_root / "trajectory_summary.csv").write_text(
            "codec,depth,group_type,group_value,metric,mean,sd,n_valid\n",
            encoding="utf-8",
        )


class AggregateRotationsTest(unittest.TestCase):
    def test_complete_168_run_aggregation_and_rotation_zero_exclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "trajectory"
            build_fixture(root)
            (root / "rotation_00_test_smoke").mkdir(parents=True)
            output = Path(directory) / "combined"
            audit = aggregate(argparse.Namespace(
                trajectory_root=root,
                output_dir=output,
                protocol_id="synthetic_v1",
                expected_rotations=7,
                expected_selection="cer",
            ))
            self.assertEqual(audit["runs"], 168)
            self.assertEqual(audit["speakers"], 15)
            self.assertEqual(audit["status"], "valid")
            with (output / "trajectory_long.csv").open(
                encoding="utf-8-sig", newline=""
            ) as handle:
                long_rows = list(csv.DictReader(handle))
            self.assertEqual({int(row["rotation"]) for row in long_rows}, set(range(1, 8)))
            with (output / "trajectory_pooled_micro_by_seed.csv").open(
                encoding="utf-8-sig", newline=""
            ) as handle:
                pooled = list(csv.DictReader(handle))
            overall = next(row for row in pooled if (
                row["depth"] == "1" and row["seed"] == "1337"
                and row["group_type"] == "overall" and row["metric"] == "wer"
            ))
            self.assertEqual(overall["n_rotations"], "7")
            with (output / "trajectory_speaker_macro_by_seed.csv").open(
                encoding="utf-8-sig", newline=""
            ) as handle:
                speaker_macro = list(csv.DictReader(handle))
            macro = next(row for row in speaker_macro if (
                row["depth"] == "1" and row["seed"] == "1337"
                and row["group_type"] == "speaker_macro" and row["metric"] == "cer"
            ))
            self.assertEqual(macro["n_speakers"], "15")
            self.assertTrue((output / "aggregation_audit.json").is_file())

    def test_missing_and_unexpected_rotations_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for rotation in range(1, 7):
                (root / f"rotation_{rotation:02d}_test_x").mkdir()
            with self.assertRaisesRegex(ValueError, "Missing formal rotations"):
                discover_rotations(root)
            (root / "rotation_07_test_x").mkdir()
            (root / "rotation_08_test_x").mkdir()
            with self.assertRaisesRegex(ValueError, "Unexpected formal rotation"):
                discover_rotations(root)

    def test_configuration_difference_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "trajectory"
            build_fixture(root)
            config_path = root / "rotation_03_test_c" / "sweep_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["train_args"][-1] = "16"
            write_json(config_path, config)
            with self.assertRaisesRegex(ValueError, "configuration differs"):
                aggregate(argparse.Namespace(
                    trajectory_root=root,
                    output_dir=Path(directory) / "combined",
                    protocol_id="synthetic_v1",
                    expected_rotations=7,
                    expected_selection="cer",
                ))

    def test_nonempty_output_directory_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "combined"
            output.mkdir()
            (output / "keep.txt").write_text("user data", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                prepare_output_dir(output)
            self.assertEqual((output / "keep.txt").read_text(encoding="utf-8"), "user data")


if __name__ == "__main__":
    unittest.main()

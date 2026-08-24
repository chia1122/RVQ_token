import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path

from rvq_asr.aggregate_rotations import aggregate
from rvq_asr.compare_representations import compare
from rvq_asr.sweep_individual_layers import build_command, build_plans
from rvq_asr.test_aggregate_rotations import build_fixture


def args(reference, output, **overrides):
    values = {
        "reference_trajectory_root": reference,
        "output_root": output,
        "protocol_id": "individual_synthetic_v1",
        "expected_rotations": 7,
        "expected_selection": "cer",
        "rotations": "auto",
        "layers": "auto",
        "seeds": "auto",
        "dry_run": True,
        "resume": False,
        "aggregate_only": False,
        "continue_on_error": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def convert_fixture_to_individual(root: Path) -> None:
    for rotation in range(1, 8):
        rotation_root = root / f"rotation_{rotation:02d}_test_{chr(96 + rotation)}"
        config_path = rotation_root / "sweep_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config.update({
            "protocol_id": "individual_synthetic_v1",
            "representation_mode": "discrete_learned",
            "rvq_mode": "individual",
            "layers": config["depths"],
            "conditions": [f"individual_q{depth}" for depth in config["depths"]],
        })
        config_path.write_text(json.dumps(config), encoding="utf-8")
        codec_root = rotation_root / "speechtokenizer"
        for depth in range(1, 9):
            (codec_root / f"k{depth}").rename(codec_root / f"individual_q{depth}")


class IndividualLayerSweepTest(unittest.TestCase):
    def test_reference_driven_plan_has_168_isolated_individual_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "cumulative"
            build_fixture(reference)
            plans = build_plans(args(reference, Path(directory) / "individual"))
            runs = [(plan, run) for plan in plans for run in plan.runs]
            self.assertEqual(len(plans), 7)
            self.assertEqual(len(runs), 168)
            self.assertEqual(len({run.output_dir for _, run in runs}), 168)
            for plan, run in runs:
                command = build_command(plan, run)
                layer = str(run.depth)
                self.assertEqual(command[command.index("--num-rvq-layers") + 1], layer)
                self.assertEqual(command[command.index("--active-rvq-layers") + 1], layer)
                self.assertEqual(
                    command[command.index("--condition-name") + 1],
                    f"individual_q{layer}",
                )

    def test_subset_and_reference_mismatch_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "cumulative"
            build_fixture(reference)
            plans = build_plans(args(
                reference, Path(directory) / "individual",
                rotations="1", layers="2,8", seeds="1337",
            ))
            self.assertEqual(len(plans), 1)
            self.assertEqual([(run.depth, run.seed) for run in plans[0].runs], [(2, 1337), (8, 1337)])
            plans[0].runs[0].output_dir.mkdir(parents=True)
            with self.assertRaisesRegex(FileExistsError, "Refusing to reuse"):
                build_plans(args(reference, Path(directory) / "individual"))
            config_path = reference / "rotation_03_test_c" / "sweep_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["train_args"][-1] = "16"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "configuration differs"):
                build_plans(args(reference, Path(directory) / "other"))

    def test_requested_layer_cannot_exceed_available_codebooks(self):
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "cumulative"
            build_fixture(reference)
            config_path = reference / "rotation_01_test_a" / "sweep_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            token_index = Path(config["token_index"])
            rows = [
                json.loads(line)
                for line in token_index.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            for row in rows:
                row["num_codebooks"] = 4
            token_index.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "has 4 codebooks"):
                build_plans(args(reference, Path(directory) / "individual"))

    def test_individual_aggregation_and_paired_comparison_preserve_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            cumulative_root = Path(directory) / "cumulative"
            individual_root = Path(directory) / "individual"
            build_fixture(cumulative_root)
            build_fixture(individual_root)
            convert_fixture_to_individual(individual_root)
            cumulative_output = Path(directory) / "cumulative_aggregate"
            individual_output = Path(directory) / "individual_aggregate"
            aggregate(argparse.Namespace(
                trajectory_root=cumulative_root, output_dir=cumulative_output,
                protocol_id="cumulative_synthetic_v1", expected_rotations=7,
                expected_selection="cer",
            ))
            audit = aggregate(argparse.Namespace(
                trajectory_root=individual_root, output_dir=individual_output,
                protocol_id="individual_synthetic_v1", expected_rotations=7,
                expected_selection="cer",
            ))
            self.assertEqual(audit["rvq_mode"], "individual")
            with (individual_output / "trajectory_long.csv").open(
                encoding="utf-8-sig", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({row["rvq_mode"] for row in rows}, {"individual"})
            self.assertEqual(
                {row["condition"] for row in rows},
                {f"individual_q{depth}" for depth in range(1, 9)},
            )
            paired = compare(cumulative_output, individual_output)
            self.assertTrue(paired)
            self.assertEqual(
                {float(row["delta_individual_minus_cumulative"]) for row in paired},
                {0.0},
            )


if __name__ == "__main__":
    unittest.main()

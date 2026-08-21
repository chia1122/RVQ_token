import json
import tempfile
import unittest
from pathlib import Path

from rvq_asr.sweep_depths import (
    collect_long_rows,
    plan_runs,
    resolve_depths,
    summarize_long_rows,
    validate_train_args,
)


class SweepDepthsTest(unittest.TestCase):
    def test_resolve_all_depths_and_reject_excess(self):
        self.assertEqual(resolve_depths("auto", 4), [1, 2, 3, 4])
        self.assertEqual(resolve_depths("3,1,2", 4), [1, 2, 3])
        with self.assertRaises(ValueError):
            resolve_depths("1,5", 4)
        with self.assertRaises(ValueError):
            validate_train_args(["--active-rvq-layers", "1"])
        with self.assertRaises(ValueError):
            validate_train_args(["--layer-fusion=learned"])

    def test_depth_seed_outputs_are_isolated_and_collisions_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = plan_runs(root, "encodec", [1, 2], [7, 9], resume=False)
            self.assertEqual(len({run.output_dir for run in runs}), 4)
            runs[0].output_dir.mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                plan_runs(root, "encodec", [1, 2], [7, 9], resume=False)
            self.assertEqual(len(plan_runs(root, "encodec", [1, 2], [7, 9], resume=True)), 4)

    def test_long_and_summary_aggregation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = plan_runs(root, "dac", [1], [1, 2], resume=True)
            for run, wer in zip(runs, (0.2, 0.4)):
                run.output_dir.mkdir(parents=True)
                result = {"test": {"groups": [{
                    "group_type": "condition", "group_value": "dysarthric",
                    "wer": wer, "cer": wer / 2, "substitutions": 2,
                    "deletions": 1, "insertions": 0,
                    "substitution_rate": 0.2, "deletion_rate": 0.1,
                    "insertion_rate": 0.0, "empty_hypothesis_ratio": 0.0,
                    "ctc_blank_frame_ratio": 0.8,
                }]}}
                (run.output_dir / "results.json").write_text(json.dumps(result), encoding="utf-8")
            long_rows, statuses = collect_long_rows(runs)
            self.assertTrue(all(row["status"] == "valid" for row in statuses))
            wer_rows = [row for row in long_rows if row["metric"] == "wer"]
            self.assertEqual({row["seed"] for row in wer_rows}, {1, 2})
            summary = summarize_long_rows(long_rows)
            wer_summary = next(row for row in summary if row["metric"] == "wer")
            self.assertAlmostEqual(wer_summary["mean"], 0.3)
            self.assertEqual(wer_summary["n_valid"], 2)
            self.assertGreater(wer_summary["sd"], 0)


if __name__ == "__main__":
    unittest.main()

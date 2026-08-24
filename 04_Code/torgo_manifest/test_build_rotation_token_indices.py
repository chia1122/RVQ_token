import argparse
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from build_rotation_token_indices import build_indices


class RotationTokenIndexTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.token_root = self.root / "master_tokens"
        self.token_root.mkdir()
        self.master_index = self.token_root / "tokens.jsonl"
        self.splits = self.root / "generated_splits"
        self.splits.mkdir()
        self.output = self.root / "rotation_indices"
        self.rows = []
        speaker_metadata = {
            "D1": ("dysarthric", "mild"),
            "D2": ("dysarthric", "severe"),
            "C1": ("control", "control"),
            "C2": ("control", "control"),
        }
        for index, (speaker, (condition, severity)) in enumerate(
            speaker_metadata.items(), start=1
        ):
            relative = Path("master") / speaker / f"u{index}.pt"
            token_file = self.token_root / relative
            token_file.parent.mkdir(parents=True, exist_ok=True)
            token_file.write_bytes(b"synthetic token placeholder")
            self.rows.append({
                "utt_id": f"u{index}",
                "token_path": relative.as_posix(),
                "num_frames": 20,
                "num_codebooks": 8,
                "codebook_size": 1024,
                "speaker_id": speaker,
                "condition": condition,
                "severity": severity,
                "split": "train",
                "text_norm": f"TEXT {index}",
            })
        self.write_jsonl(self.master_index, self.rows)
        self.write_rotation(
            "rotation_01_test_a",
            train=["D2", "C2"], valid=["D1"], test=["C1"],
        )
        self.write_rotation(
            "rotation_02_test_b",
            train=["D1", "C1"], valid=["C2"], test=["D2"],
        )

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def write_jsonl(path, rows):
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

    def write_rotation(self, name, **roles):
        (self.splits / f"{name}.json").write_text(
            json.dumps(roles), encoding="utf-8"
        )

    def args(self, output=None):
        return argparse.Namespace(
            master_token_index=self.master_index,
            master_token_root=self.token_root,
            generated_splits_dir=self.splits,
            output_dir=output or self.output,
        )

    @staticmethod
    def read_jsonl(path):
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]

    def test_builds_dynamic_rotation_indexes_sharing_master_tokens(self):
        audit = build_indices(self.args())

        self.assertEqual(audit["rotation_count"], 2)
        self.assertEqual(audit["utterance_count"], 4)
        self.assertTrue(audit["token_files_shared"])
        self.assertTrue(all(
            rotation["missing_token_files"] == 0
            and rotation["metadata_mismatches"] == 0
            for rotation in audit["rotations"]
        ))
        first = self.read_jsonl(
            self.output / "rotation_01_test_a" / "tokens.jsonl"
        )
        second = self.read_jsonl(
            self.output / "rotation_02_test_b" / "tokens.jsonl"
        )
        self.assertEqual(
            {row["token_path"] for row in first},
            {row["token_path"] for row in second},
        )
        first_by_speaker = {row["speaker_id"]: row for row in first}
        self.assertEqual(first_by_speaker["D1"]["split"], "valid")
        self.assertEqual(first_by_speaker["C1"]["split"], "test")
        self.assertEqual(first_by_speaker["D1"]["condition"], "dysarthric")
        self.assertEqual(first_by_speaker["D1"]["severity"], "mild")
        self.assertEqual(first_by_speaker["D1"]["text_norm"], "TEXT 1")
        audit_text = (self.output / "rotation_index_audit.json").read_text(
            encoding="utf-8"
        )
        self.assertNotIn(str(self.root.resolve()), audit_text)

    def test_refuses_nonempty_output_directory(self):
        self.output.mkdir()
        (self.output / "keep.txt").write_text("user data", encoding="utf-8")
        with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
            build_indices(self.args())
        self.assertEqual(
            (self.output / "keep.txt").read_text(encoding="utf-8"), "user data"
        )

    def test_cli_smoke(self):
        script = Path(__file__).with_name("build_rotation_token_indices.py")
        completed = subprocess.run([
            sys.executable,
            str(script),
            "--master-token-index", str(self.master_index),
            "--master-token-root", str(self.token_root),
            "--generated-splits-dir", str(self.splits),
            "--output-dir", str(self.output),
        ], check=True, capture_output=True, text=True)
        result = json.loads(completed.stdout)
        self.assertEqual(result["status"], "valid")
        self.assertEqual(result["rotations"], 2)
        self.assertEqual(result["utterances_per_rotation"], 4)
        self.assertTrue((self.output / "rotation_index_audit.json").is_file())

    def test_rejects_incomplete_or_overlapping_speaker_assignments(self):
        path = self.splits / "rotation_01_test_a.json"
        path.write_text(json.dumps({
            "train": ["D2", "C2"], "valid": ["D1"], "test": ["D1"]
        }), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "multiple splits"):
            build_indices(self.args())
        self.assertFalse(self.output.exists())

        path.write_text(json.dumps({
            "train": ["D2"], "valid": ["D1"], "test": ["C1"]
        }), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            build_indices(self.args())
        self.assertFalse(self.output.exists())

    def test_rejects_duplicate_ids_paths_and_missing_token_files(self):
        cases = []

        duplicate_id = [dict(row) for row in self.rows]
        duplicate_id[1]["utt_id"] = duplicate_id[0]["utt_id"]
        cases.append((duplicate_id, "Duplicate utt_id"))

        duplicate_path = [dict(row) for row in self.rows]
        duplicate_path[1]["token_path"] = duplicate_path[0]["token_path"]
        cases.append((duplicate_path, "Duplicate token_path"))

        missing_file = [dict(row) for row in self.rows]
        missing_file[1]["token_path"] = "master/D2/missing.pt"
        cases.append((missing_file, "Token file not found"))

        escaped = [dict(row) for row in self.rows]
        escaped[1]["token_path"] = "../outside.pt"
        cases.append((escaped, "escapes token root"))

        for rows, message in cases:
            with self.subTest(message=message):
                self.write_jsonl(self.master_index, rows)
                with self.assertRaisesRegex((ValueError, FileNotFoundError), message):
                    build_indices(self.args())
                self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()

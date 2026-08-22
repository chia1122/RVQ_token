import argparse
import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

from audit_speaker_folds import (
    audit_manifest,
    build_rotations,
    load_exclusion_counts,
    load_fold_config,
    load_jsonl,
    run_audit,
    validate_folds,
    validate_manifest,
)
from build_torgo_manifest import load_speaker_metadata


MODULE_DIR = Path(__file__).resolve().parent
CANONICAL_METADATA_PATH = MODULE_DIR / "config" / "speaker_metadata.csv"
METADATA_PATH = MODULE_DIR / "config" / "speaker_metadata_including_mild_v1.csv"
FOLD_CONFIG_PATH = MODULE_DIR / "config" / "speaker_folds_including_mild_v1.json"


class SpeakerFoldConfigTest(unittest.TestCase):
    def setUp(self):
        self.metadata = load_speaker_metadata(METADATA_PATH)
        self.config = load_fold_config(FOLD_CONFIG_PATH)

    def test_protocol_config_has_complete_disjoint_rotations(self):
        folds = validate_folds(self.metadata, self.config)
        rotations = build_rotations(folds)

        self.assertEqual(len(folds), 7)
        self.assertEqual(len(rotations), 7)
        self.assertEqual(
            {speaker for fold in folds for speaker in fold["speakers"]},
            set(self.metadata),
        )
        self.assertTrue(all(len(rotation["train_folds"]) == 5 for rotation in rotations))

        test_counts = {
            speaker: sum(speaker in rotation["test"] for rotation in rotations)
            for speaker in self.metadata
        }
        valid_counts = {
            speaker: sum(speaker in rotation["valid"] for rotation in rotations)
            for speaker in self.metadata
        }
        self.assertEqual(set(test_counts.values()), {1})
        self.assertEqual(set(valid_counts.values()), {1})

    def test_versioned_metadata_changes_only_mild_protocol_fields(self):
        canonical = load_speaker_metadata(CANONICAL_METADATA_PATH)
        self.assertEqual(set(canonical), set(self.metadata))
        for speaker in canonical:
            if speaker not in {"F04", "M03"}:
                self.assertEqual(canonical[speaker], self.metadata[speaker])
                continue
            self.assertEqual(canonical[speaker]["severity"], "mild")
            self.assertEqual(self.metadata[speaker]["severity"], "mild")
            self.assertEqual(canonical[speaker]["speaker_type"], "dysarthric")
            self.assertEqual(self.metadata[speaker]["speaker_type"], "dysarthric")
            self.assertEqual(canonical[speaker]["include_in_experiment"], "false")
            self.assertEqual(self.metadata[speaker]["include_in_experiment"], "true")
            differing = {
                field
                for field in canonical[speaker]
                if canonical[speaker][field] != self.metadata[speaker][field]
            }
            self.assertEqual(
                differing, {"severity_source", "include_in_experiment"}
            )

    def test_required_mild_speakers_are_asserted(self):
        invalid = copy.deepcopy(self.metadata)
        invalid["F04"]["include_in_experiment"] = "false"
        with self.assertRaisesRegex(ValueError, "assigned_but_excluded"):
            validate_folds(invalid, self.config)

        invalid = copy.deepcopy(self.metadata)
        invalid["M03"]["severity"] = "severe"
        with self.assertRaisesRegex(ValueError, "Required metadata mismatch"):
            validate_folds(invalid, self.config)

    def test_duplicate_missing_and_unknown_speakers_are_rejected(self):
        duplicate = copy.deepcopy(self.config)
        duplicate["folds"][1]["speakers"].append("F01")
        with self.assertRaisesRegex(ValueError, "multiple folds"):
            validate_folds(self.metadata, duplicate)

        missing = copy.deepcopy(self.config)
        missing["folds"][0]["speakers"].remove("F01")
        with self.assertRaisesRegex(ValueError, "missing_included"):
            validate_folds(self.metadata, missing)

        unknown = copy.deepcopy(self.config)
        unknown["folds"][0]["speakers"][0] = "UNKNOWN"
        with self.assertRaisesRegex(ValueError, "unknown"):
            validate_folds(self.metadata, unknown)

    def test_every_fold_requires_both_conditions(self):
        invalid = copy.deepcopy(self.config)
        invalid["folds"][0]["speakers"].remove("MC01")
        invalid["folds"][1]["speakers"].append("MC01")
        with self.assertRaisesRegex(ValueError, "must contain control and dysarthric"):
            validate_folds(self.metadata, invalid)


class SpeakerFoldManifestAuditTest(unittest.TestCase):
    def setUp(self):
        self.metadata = load_speaker_metadata(METADATA_PATH)
        self.config = load_fold_config(FOLD_CONFIG_PATH)
        self.folds = validate_folds(self.metadata, self.config)
        self.rotations = build_rotations(self.folds)

    def synthetic_rows(self):
        rows = []
        for speaker, values in self.metadata.items():
            for index, text in enumerate(("COMMON PROMPT", f"UNIQUE {speaker}")):
                rows.append({
                    "utt_id": f"{speaker}-{index}",
                    "speaker_id": speaker,
                    "condition": values["speaker_type"],
                    "severity": values["severity"],
                    "text_norm": text,
                    "duration": 1.0,
                    "audio_status": "available",
                })
        return rows

    def write_manifest(self, path, rows):
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

    def test_manifest_audit_counts_prompt_overlap(self):
        rows = self.synthetic_rows()
        validate_manifest(rows, self.metadata)
        statistics, overlaps = audit_manifest(
            rows, self.rotations, self.metadata
        )

        self.assertEqual(len(statistics), 21)
        self.assertEqual(len(overlaps), 7)
        self.assertTrue(all(row["test_seen_prompt_ratio"] == 0.5 for row in overlaps))
        self.assertTrue(all(row["test_unseen_prompt_utterances"] > 0 for row in overlaps))
        self.assertTrue(all(row["hours"] for row in statistics))

    def test_manifest_missing_speaker_and_label_mismatch_fail(self):
        rows = [
            row for row in self.synthetic_rows() if row["speaker_id"] != "F04"
        ]
        with self.assertRaisesRegex(ValueError, r"missing_included=\['F04'\]"):
            validate_manifest(rows, self.metadata)

        rows = self.synthetic_rows()
        rows[0]["condition"] = "control"
        with self.assertRaisesRegex(ValueError, "condition mismatch"):
            validate_manifest(rows, self.metadata)

    def test_jsonl_loader_rejects_missing_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            path.write_text('{"speaker_id": "F01"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing fields"):
                load_jsonl(path)

    def test_cli_outputs_are_isolated_and_builder_compatible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            output = root / "audit"
            self.write_manifest(manifest, self.synthetic_rows())
            args = argparse.Namespace(
                speaker_metadata=METADATA_PATH,
                fold_config=FOLD_CONFIG_PATH,
                manifest=manifest,
                excluded_samples=None,
                output_dir=output,
            )

            audit = run_audit(args)
            self.assertTrue(audit["manifest_audited"])
            self.assertEqual(audit["included_speaker_count"], 15)
            self.assertTrue((output / "fold_statistics.csv").exists())
            self.assertTrue((output / "prompt_overlap.csv").exists())

            generated = sorted((output / "generated_splits").glob("*.json"))
            self.assertEqual(len(generated), 7)
            for path in generated:
                config = json.loads(path.read_text(encoding="utf-8"))
                assigned = config["train"] + config["valid"] + config["test"]
                self.assertEqual(len(assigned), 15)
                self.assertEqual(len(set(assigned)), 15)
                self.assertEqual(set(assigned), set(self.metadata))

            audit_text = (output / "fold_audit.json").read_text(encoding="utf-8")
            self.assertNotIn(str(root.resolve()), audit_text)
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                run_audit(args)

    def test_cli_validates_manifest_before_creating_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            output = root / "audit"
            rows = [
                row for row in self.synthetic_rows() if row["speaker_id"] != "F04"
            ]
            self.write_manifest(manifest, rows)
            args = argparse.Namespace(
                speaker_metadata=METADATA_PATH,
                fold_config=FOLD_CONFIG_PATH,
                manifest=manifest,
                excluded_samples=None,
                output_dir=output,
            )
            with self.assertRaisesRegex(ValueError, r"missing_included=\['F04'\]"):
                run_audit(args)
            self.assertFalse(output.exists())

    def test_exclusion_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "excluded.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["reason"])
                writer.writeheader()
                writer.writerows([
                    {"reason": "missing_audio"},
                    {"reason": "missing_audio"},
                    {"reason": "empty_transcript"},
                ])
            self.assertEqual(
                load_exclusion_counts(path),
                {"empty_transcript": 1, "missing_audio": 2},
            )


if __name__ == "__main__":
    unittest.main()

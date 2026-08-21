import unittest
import json
import tempfile
from argparse import Namespace
from pathlib import Path

from evaluate_with_faster_whisper import (
    Scores,
    build_items,
    normalize_text,
    prediction_from_hypothesis,
    rvq_condition_order,
)
from reconstruct_encodec_prefixes import (
    DEFAULT_LAYERS,
    parse_layers,
    safe_name,
    validate_requested_layers,
)
from reconstruct_dac_prefixes import prefix_latent_from_codes
from paired_bootstrap import load_pairs, percentile


class ReconstructionHelpersTest(unittest.TestCase):
    def test_parse_layers(self):
        self.assertEqual(parse_layers("8,1,4,4"), [1, 4, 8])
        self.assertEqual(parse_layers(DEFAULT_LAYERS), list(range(1, 9)))
        with self.assertRaises(ValueError):
            parse_layers("0,1")

    def test_requested_depth_cannot_exceed_num_codebooks(self):
        class Codes:
            ndim = 2
            shape = (2, 3)

        payload = {"num_codebooks": 3, "codes": Codes()}
        self.assertEqual(validate_requested_layers(payload, [1, 2, 3], "fixture"), 3)
        with self.assertRaises(ValueError):
            validate_requested_layers(payload, [4], "fixture")

    def test_dynamic_rvq_condition_order(self):
        self.assertEqual(
            rvq_condition_order(["k10", "k3", "original", "k1"]),
            ["original", "k1", "k3", "k10"],
        )

    def test_build_items_preserves_speech_condition(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            reconstruction = root / "reconstruction.jsonl"
            manifest_row = {
                "utt_id": "u1", "audio_path": "u1.wav", "speaker_id": "F01",
                "condition": "dysarthric", "severity": "severe", "split": "test",
                "text_norm": "HELLO",
            }
            reconstruction_row = {
                "utt_id": "u1", "audio_path": "k3/u1.wav", "speaker_id": "F01",
                "condition": "dysarthric", "rvq_condition": "k3",
                "severity": "severe", "split": "test", "text_norm": "HELLO",
            }
            manifest.write_text(json.dumps(manifest_row) + "\n", encoding="utf-8")
            reconstruction.write_text(json.dumps(reconstruction_row) + "\n", encoding="utf-8")
            args = Namespace(
                conditions="auto", speakers="", manifest=manifest,
                reconstruction_index=reconstruction, audio_root=root,
                reconstruction_root=root, split="test", limit_per_condition=0,
            )
            items = build_items(args)
            self.assertEqual(
                {(item["rvq_condition"], item["condition"]) for item in items},
                {("original", "dysarthric"), ("k3", "dysarthric")},
            )
            prediction = prediction_from_hypothesis(
                next(item for item in items if item["rvq_condition"] == "k3"),
                "hello",
            )
            self.assertEqual(prediction["condition"], "dysarthric")
            self.assertEqual(prediction["rvq_condition"], "k3")

    def test_safe_name(self):
        self.assertEqual(safe_name("F01/a:1"), "F01_a_1")

    def test_normalize_and_scores(self):
        self.assertEqual(normalize_text("Don't-stop!"), "DON'T STOP")
        scores = Scores()
        scores.update("A B", "A C")
        self.assertEqual(scores.row()["wer"], 0.5)

    def test_dac_from_codes_tuple(self):
        class Quantizer:
            def from_codes(self, codes):
                return "latent", "projected", codes

        class Model:
            quantizer = Quantizer()

        self.assertEqual(prefix_latent_from_codes(Model(), "codes"), "latent")

    def test_percentile(self):
        self.assertEqual(percentile([0.0, 10.0], 0.5), 5.0)

    def test_bootstrap_reads_new_rvq_condition_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            rows = [
                {"rvq_condition": depth, "condition": "control", "utt_id": "u1",
                 "speaker_id": "FC01", "severity": "control"}
                for depth in ("k1", "k2")
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            pairs = load_pairs(path, "k1", "k2")
            self.assertEqual(len(pairs["FC01"]), 1)


if __name__ == "__main__":
    unittest.main()

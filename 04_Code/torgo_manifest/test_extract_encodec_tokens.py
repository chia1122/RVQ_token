import json
import tempfile
import unittest
from pathlib import Path

from extract_encodec_tokens import load_manifest, resolve_audio, token_relative_path


class TokenExtractorHelpersTest(unittest.TestCase):
    def test_resolve_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(resolve_audio(root, "F01/a.wav"), (root / "F01/a.wav").resolve())
            with self.assertRaises(ValueError):
                resolve_audio(root, "../outside.wav")

    def test_token_relative_path(self):
        row = {"split": "train", "speaker_id": "F01", "utt_id": "F01/a:1"}
        self.assertEqual(token_relative_path(row).as_posix(), "train/F01/F01_a_1.pt")

    def test_load_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            row = {
                "utt_id": "u1",
                "audio_path": "F01/a.wav",
                "speaker_id": "F01",
                "condition": "dysarthric",
                "severity": "severe",
                "split": "train",
            }
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            self.assertEqual(load_manifest(path), [row])

    def test_load_manifest_accepts_legacy_speaker_type(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            row = {
                "utt_id": "u1",
                "audio_path": "FC01/a.wav",
                "speaker_id": "FC01",
                "speaker_type": "control",
                "severity": "control",
                "split": "train",
            }
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            loaded = load_manifest(path)
            self.assertEqual(loaded[0]["condition"], "control")


if __name__ == "__main__":
    unittest.main()

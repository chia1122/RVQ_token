import tempfile
import unittest
import wave
from pathlib import Path

from build_torgo_manifest import (
    audio_relative_path,
    is_included,
    normalize_text,
    speaker_condition,
    wav_metadata,
)


class ManifestHelpersTest(unittest.TestCase):
    def test_normalize_text(self):
        self.assertEqual(normalize_text("  Don't—ask me!  "), "DON'T ASK ME")
        self.assertEqual(normalize_text("well-known"), "WELL KNOWN")
        self.assertEqual(normalize_text("What?"), "WHAT")

    def test_audio_relative_path(self):
        self.assertEqual(audio_relative_path("/F01/Session1/a.wav").as_posix(), "F01/Session1/a.wav")
        with self.assertRaises(ValueError):
            audio_relative_path("../outside.wav")

    def test_inclusion_flag(self):
        self.assertTrue(is_included({}))
        self.assertFalse(is_included({"include_in_experiment": "false"}))
        with self.assertRaises(ValueError):
            is_included({"include_in_experiment": "maybe"})

    def test_speaker_condition(self):
        self.assertEqual(speaker_condition({"speaker_type": "control"}), "control")
        self.assertEqual(
            speaker_condition({"speaker_type": "dysarthric"}), "dysarthric"
        )
        with self.assertRaises(ValueError):
            speaker_condition({"speaker_type": "unknown"})

    def test_wav_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.wav"
            with wave.open(str(path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(b"\x00\x00" * 8000)
            self.assertEqual(wav_metadata(path), (16000, 8000, 0.5))


if __name__ == "__main__":
    unittest.main()

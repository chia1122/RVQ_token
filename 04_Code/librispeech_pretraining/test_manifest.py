import unittest

from build_librispeech_manifest import normalize_text, parse_subsets


class LibriSpeechManifestTest(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(normalize_text("Don't-stop."), "DON'T STOP")

    def test_parse_subsets(self):
        self.assertEqual(
            parse_subsets("train-clean-100:train,dev-clean:valid"),
            {"train-clean-100": "train", "dev-clean": "valid"},
        )
        with self.assertRaises(ValueError):
            parse_subsets("train-clean-100:development")


if __name__ == "__main__":
    unittest.main()

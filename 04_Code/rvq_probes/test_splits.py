import unittest

from rvq_probes.splits import validate_speaker_disjoint


class SplitTest(unittest.TestCase):
    def row(self, speaker, split, condition="control"):
        return {
            "utt_id": f"{speaker}-{split}", "token_path": "x.pt",
            "speaker_id": speaker, "condition": condition,
            "severity": "control" if condition == "control" else "severe",
            "split": split,
        }

    def test_disjoint_split_passes(self):
        result = validate_speaker_disjoint([
            self.row("A", "train"), self.row("B", "valid"),
            self.row("C", "test", "dysarthric"),
        ])
        self.assertEqual(result["test"]["speakers"], ["C"])

    def test_leakage_fails_loudly(self):
        with self.assertRaisesRegex(ValueError, "Speaker leakage"):
            validate_speaker_disjoint([
                self.row("A", "train"), self.row("B", "valid"),
                self.row("A", "test"),
            ])


if __name__ == "__main__":
    unittest.main()

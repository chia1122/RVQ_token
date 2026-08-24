import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
    from rvq_asr.data import CTCBatchCollator, RVQTokenDataset
    from rvq_asr.text import CharacterTokenizer
except ModuleNotFoundError:
    HAS_TORCH = False
else:
    HAS_TORCH = True


@unittest.skipUnless(HAS_TORCH, "PyTorch is required for RVQTokenDataset tests")
class RVQTokenDatasetTest(unittest.TestCase):
    def test_rotation_index_controls_split_and_metadata_for_shared_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relative_token = Path("master") / "F04" / "u1.pt"
            token_path = root / relative_token
            token_path.parent.mkdir(parents=True)
            torch.save({
                "utt_id": "u1",
                "codes": torch.zeros((20, 8), dtype=torch.int16),
                "split": "train",
                "condition": "control",
                "severity": "control",
            }, token_path)
            index = root / "rotation_tokens.jsonl"
            row = {
                "utt_id": "u1",
                "token_path": relative_token.as_posix(),
                "num_frames": 20,
                "num_codebooks": 8,
                "codebook_size": 1024,
                "speaker_id": "F04",
                "condition": "dysarthric",
                "severity": "mild",
                "split": "test",
                "text_norm": "A",
            }
            index.write_text(json.dumps(row) + "\n", encoding="utf-8")

            dataset = RVQTokenDataset(
                index, root, "test", 8, CharacterTokenizer()
            )
            sample = dataset[0]
            self.assertEqual(tuple(sample["codes"].shape), (20, 8))
            self.assertEqual(sample["speaker_id"], "F04")
            self.assertEqual(sample["condition"], "dysarthric")
            self.assertEqual(sample["severity"], "mild")

            batch = CTCBatchCollator(1024)([sample])
            self.assertEqual(batch["speaker_ids"], ["F04"])
            self.assertEqual(batch["conditions"], ["dysarthric"])
            self.assertEqual(batch["severities"], ["mild"])

            with self.assertRaisesRegex(ValueError, "No rows found"):
                RVQTokenDataset(index, root, "train", 8, CharacterTokenizer())


if __name__ == "__main__":
    unittest.main()

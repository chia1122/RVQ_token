import tempfile
import unittest
from pathlib import Path

from rvq_probes.phonemes import PhonemeTokenizer, load_cmudict, transcript_to_phonemes


class PhonemeTest(unittest.TestCase):
    def test_first_pronunciation_stress_removal_and_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cmudict"
            path.write_text("WORD 1 W ER1 D\nWORD 2 W AO2 R D\n", encoding="ascii")
            lexicon = load_cmudict(path)
        phones, audit = transcript_to_phonemes("WORD", lexicon)
        self.assertEqual(phones, ["W", "ER", "D"])
        self.assertEqual(audit[0]["alternative_pronunciation_count"], 1)
        self.assertEqual(audit[0]["source"], "lexicon")

    def test_oov_uses_g2p_and_ctc_decode(self):
        phones, audit = transcript_to_phonemes(
            "OOV", {}, lambda word: ["OW1", " ", "V", "IY2"]
        )
        self.assertEqual(phones, ["OW", "V", "IY"])
        self.assertEqual(audit[0]["source"], "g2p")
        tokenizer = PhonemeTokenizer(["<blank>", "IY", "OW", "V"])
        ids = tokenizer.encode(phones)
        self.assertEqual(tokenizer.decode_ctc([ids[0], ids[0], 0, ids[1], ids[2]]), phones)


if __name__ == "__main__":
    unittest.main()

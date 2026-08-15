import unittest

from evaluate_with_faster_whisper import Scores, normalize_text
from reconstruct_encodec_prefixes import parse_layers, safe_name
from reconstruct_dac_prefixes import prefix_latent_from_codes
from paired_bootstrap import percentile


class ReconstructionHelpersTest(unittest.TestCase):
    def test_parse_layers(self):
        self.assertEqual(parse_layers("8,1,4,4"), [1, 4, 8])
        with self.assertRaises(ValueError):
            parse_layers("0,1")

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


if __name__ == "__main__":
    unittest.main()

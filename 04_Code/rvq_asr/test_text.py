import unittest

from rvq_asr.text import (
    CharacterTokenizer,
    ErrorRate,
    edit_counts,
    edit_distance,
    prediction_row,
)


class TextTest(unittest.TestCase):
    def test_ctc_decode_removes_blanks_and_repeats(self):
        tokenizer = CharacterTokenizer()
        h = tokenizer.token_to_id["H"]
        i = tokenizer.token_to_id["I"]
        self.assertEqual(tokenizer.decode_ctc([h, h, 0, i, i]), "HI")

    def test_edit_distance(self):
        self.assertEqual(edit_distance("A B C".split(), "A D C".split()), 1)

    def test_edit_counts(self):
        self.assertEqual(edit_counts("A B C".split(), "A D C E".split()), (1, 0, 1))
        self.assertEqual(edit_counts("A B C".split(), "A C".split()), (0, 1, 0))

    def test_prediction_preserves_condition(self):
        row = prediction_row(
            "u1", "F01", "dysarthric", "severe", "A B", "A C"
        )
        self.assertEqual(row["condition"], "dysarthric")
        self.assertEqual(row["substitutions"], 1)

    def test_error_rate(self):
        metric = ErrorRate()
        metric.update_words("A B C", "A C")
        self.assertAlmostEqual(metric.value, 1 / 3)
        self.assertEqual(metric.deletions, 1)

    def test_character_error_rate_ignores_spaces(self):
        metric = ErrorRate()
        metric.update_characters("AB CD", "AB D")
        self.assertAlmostEqual(metric.value, 1 / 4)


if __name__ == "__main__":
    unittest.main()

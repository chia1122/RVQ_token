import unittest

from rvq_asr.text import CharacterTokenizer, ErrorRate, edit_distance


class TextTest(unittest.TestCase):
    def test_ctc_decode_removes_blanks_and_repeats(self):
        tokenizer = CharacterTokenizer()
        h = tokenizer.token_to_id["H"]
        i = tokenizer.token_to_id["I"]
        self.assertEqual(tokenizer.decode_ctc([h, h, 0, i, i]), "HI")

    def test_edit_distance(self):
        self.assertEqual(edit_distance("A B C".split(), "A D C".split()), 1)

    def test_error_rate(self):
        metric = ErrorRate()
        metric.update_words("A B C", "A C")
        self.assertAlmostEqual(metric.value, 1 / 3)

    def test_character_error_rate_ignores_spaces(self):
        metric = ErrorRate()
        metric.update_characters("AB CD", "AB D")
        self.assertAlmostEqual(metric.value, 1 / 4)


if __name__ == "__main__":
    unittest.main()

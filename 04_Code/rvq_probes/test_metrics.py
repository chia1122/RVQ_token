import unittest

from rvq_probes.metrics import binary_counts, phoneme_error_metrics


class MetricTest(unittest.TestCase):
    def test_phoneme_error_counts(self):
        result = phoneme_error_metrics(["A", "B", "C"], ["A", "D", "C", "E"])
        self.assertEqual(result["substitutions"], 1)
        self.assertEqual(result["insertions"], 1)
        self.assertAlmostEqual(result["per"], 2 / 3)

    def test_binary_counts(self):
        result = binary_counts([1, 0, 1, 0], [1, 1, 0, 0])
        self.assertEqual((result["true_positive"], result["false_positive"]), (1, 1))
        self.assertAlmostEqual(result["f1"], 0.5)


if __name__ == "__main__":
    unittest.main()

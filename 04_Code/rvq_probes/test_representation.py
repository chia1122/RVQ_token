import tempfile
import unittest
from pathlib import Path

import torch

from rvq_probes.representation import (
    FrozenCodebook, load_rvq_representation, select_individual_codes,
)


class RepresentationTest(unittest.TestCase):
    def test_individual_selection_is_not_cumulative(self):
        codes = torch.tensor([[1, 5], [2, 6], [3, 7]])
        self.assertEqual(select_individual_codes(codes, 1, 2).tolist(), [1, 2, 3])
        self.assertEqual(select_individual_codes(codes, 2, 2).tolist(), [5, 6, 7])

    def test_token_ids_map_to_selected_frozen_table(self):
        q1 = FrozenCodebook(torch.arange(12).reshape(4, 3).float(), 1, 4)
        q2 = FrozenCodebook((100 + torch.arange(12)).reshape(4, 3).float(), 2, 4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokens.pt"
            torch.save({"codes": torch.tensor([[0, 1], [2, 3]])}, path)
            first = load_rvq_representation(path, 1, q1, 2)
            second = load_rvq_representation(path, 2, q2, 2)
        self.assertTrue(torch.equal(first, q1.table[[0, 2]]))
        self.assertTrue(torch.equal(second, q2.table[[1, 3]]))
        self.assertFalse(torch.equal(first, second))

    def test_invalid_range_empty_and_shape_fail(self):
        table = FrozenCodebook(torch.zeros(4, 3), 1, 4)
        with self.assertRaisesRegex(ValueError, "outside"):
            table(torch.tensor([4]))
        with self.assertRaisesRegex(ValueError, "Empty"):
            select_individual_codes(torch.empty(0, 2), 1, 2)
        with self.assertRaisesRegex(ValueError, "Expected codes"):
            select_individual_codes(torch.zeros(2, 2, 2), 1, 2)


if __name__ == "__main__":
    unittest.main()

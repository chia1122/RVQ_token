import json
import tempfile
import unittest
from pathlib import Path

from rvq_probes.boundaries import (
    boundaries_to_frames,
    internal_phone_boundary_times,
    match_boundary_frames,
    read_mfa_phone_intervals,
    timestamp_to_nearest_frame,
)


class BoundaryTests(unittest.TestCase):
    def test_nearest_frame_is_half_up_and_clamped(self):
        self.assertEqual(timestamp_to_nearest_frame(0.01, 0.02, 10), 1)
        self.assertEqual(timestamp_to_nearest_frame(9.0, 0.02, 10), 9)

    def test_colliding_boundaries_become_one_training_target(self):
        self.assertEqual(boundaries_to_frames([0.020, 0.021], 0.02, 10), ([1], 1))

    def test_internal_boundaries_skip_silence_gaps(self):
        intervals = [
            {"start": 0.1, "end": 0.2, "phone": "S"},
            {"start": 0.2, "end": 0.3, "phone": "T"},
            {"start": 0.4, "end": 0.5, "phone": "K"},
        ]
        self.assertEqual(internal_phone_boundary_times(intervals), [0.2])

    def test_tolerant_matching_is_one_to_one(self):
        result = match_boundary_frames([10, 12], [11], tolerance=1)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 0, 1))
        exact = match_boundary_frames([10], [11], tolerance=0)
        self.assertEqual(exact["f1"], 0.0)

    def test_real_mfa_schema_parser(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x.json"
            path.write_text(json.dumps({"tiers": {"phones": {"entries": [[0.1, 0.2, "S"]]}}}))
            self.assertEqual(read_mfa_phone_intervals(path)[0]["phone"], "S")


if __name__ == "__main__":
    unittest.main()

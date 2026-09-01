#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rvq_probes.boundaries import (
    boundaries_to_frames,
    internal_phone_boundary_times,
    read_mfa_phone_intervals,
)
from rvq_probes.splits import load_index


def main(args):
    rows = load_index(args.token_index)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "boundary_targets.jsonl"
    missing, collisions, boundaries = [], 0, 0
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            path = args.alignment_dir / row["speaker_id"] / f'{row["utt_id"]}.json'
            if not path.exists():
                missing.append(row["utt_id"])
                continue
            intervals = read_mfa_phone_intervals(path)
            times = internal_phone_boundary_times(intervals)
            frames, collapsed = boundaries_to_frames(times, args.frame_duration, int(row["num_frames"]))
            collisions += collapsed
            boundaries += len(frames)
            handle.write(json.dumps({
                "utt_id": row["utt_id"], "speaker_id": row["speaker_id"],
                "num_frames": int(row["num_frames"]), "boundary_frames": frames,
                "boundary_times_seconds": times, "phone_intervals": intervals,
                "collapsed_frame_collisions": collapsed,
            }, sort_keys=True) + "\n")
    if missing and not args.allow_missing:
        raise RuntimeError(f"Missing {len(missing)} alignments; first: {missing[:5]}")
    summary = {
        "utterances_in_index": len(rows), "utterances_written": len(rows) - len(missing),
        "missing_alignments": len(missing),
        "missing_utt_ids": missing if len(missing) <= 100 else missing[:100],
        "missing_utt_ids_truncated": len(missing) > 100,
        "boundary_frames": boundaries, "collapsed_frame_collisions": collisions,
        "frame_duration_seconds": args.frame_duration,
        "mapping": "nearest_single_frame_round_half_up",
        "boundary_definition": "junction_between_adjacent_non_silence_phone_intervals",
        "utterance_edges_excluded": True, "silence_intervals_retained_for_audit": True,
    }
    (args.output_dir / "preparation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Convert MFA JSON alignments to codec-frame boundaries")
    parser.add_argument("--token-index", type=Path, required=True)
    parser.add_argument("--alignment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-duration", type=float, default=0.02)
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())

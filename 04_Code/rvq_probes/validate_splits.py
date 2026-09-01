#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from rvq_probes.splits import load_index, validate_speaker_disjoint

def main(args):
    paths = sorted(args.rotations_root.glob("rotation_*/tokens.jsonl")) if args.rotations_root else [args.token_index]
    if not paths:
        raise SystemExit("No token indices found")
    reports = [{"token_index": str(path), "splits": validate_speaker_disjoint(load_index(path))} for path in paths]
    print(json.dumps(reports, indent=2))

def parse_args():
    parser = argparse.ArgumentParser(description="Fail loudly on speaker leakage")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--token-index", type=Path)
    group.add_argument("--rotations-root", type=Path)
    return parser.parse_args()

if __name__ == "__main__":
    main(parse_args())

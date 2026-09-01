from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


SPLITS = ("train", "valid", "test")


def load_index(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = {"utt_id", "token_path", "speaker_id", "condition", "severity", "split"}
            missing = required - row.keys()
            if missing:
                raise ValueError(f"Index line {line_number} missing {sorted(missing)}")
            if row["split"] not in SPLITS:
                raise ValueError(f"Invalid split for {row['utt_id']}: {row['split']}")
            rows.append(row)
    if not rows:
        raise ValueError("Empty token index")
    return rows


def validate_speaker_disjoint(rows: list[dict]) -> dict:
    speakers = {
        split: {row["speaker_id"] for row in rows if row["split"] == split}
        for split in SPLITS
    }
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        overlap = speakers[left] & speakers[right]
        if overlap:
            raise ValueError(f"Speaker leakage between {left}/{right}: {sorted(overlap)}")
    return {
        split: {
            "speakers": sorted(speakers[split]),
            "speaker_count": len(speakers[split]),
            "utterances": sum(row["split"] == split for row in rows),
            "condition_utterances": dict(sorted(Counter(
                row["condition"] for row in rows if row["split"] == split
            ).items())),
            "severity_utterances": dict(sorted(Counter(
                row["severity"] for row in rows if row["split"] == split
            ).items())),
        }
        for split in SPLITS
    }

#!/usr/bin/env python3
"""Compare paired DAC and SpeechTokenizer ASR predictions by speaker/condition."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from evaluate_with_faster_whisper import Scores, rvq_condition_from_row, rvq_condition_order


def load(path: Path) -> dict[tuple[str, str], dict]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return {(rvq_condition_from_row(row), row["utt_id"]): row for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dac-predictions", type=Path, required=True)
    parser.add_argument("--speech-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dac, speech = load(args.dac_predictions), load(args.speech_predictions)
    paired_keys = sorted(set(dac) & set(speech))
    if not paired_keys:
        raise ValueError("No paired condition/utterance predictions")
    groups = defaultdict(Scores)
    counts = defaultdict(int)
    metadata = {}
    for key in paired_keys:
        rvq_condition = key[0]
        for codec, source in (("dac", dac), ("speechtokenizer", speech)):
            row = source[key]
            group = (codec, rvq_condition, row["speaker_id"])
            groups[group].update(row["reference"], row["hypothesis"])
            counts[group] += 1
            metadata[row["speaker_id"]] = row["severity"]
    output_rows = []
    condition_rank = {
        value: index for index, value in enumerate(
            rvq_condition_order(condition for _, condition, _ in groups)
        )
    }
    for (codec, condition, speaker), scores in sorted(
        groups.items(), key=lambda item: (item[0][0], condition_rank[item[0][1]], item[0][2])
    ):
        output_rows.append({
            "codec": codec, "rvq_condition": condition, "speaker_id": speaker,
            "severity": metadata[speaker], "utterances": counts[(codec, condition, speaker)],
            **scores.row(),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote {len(output_rows)} rows from {len(paired_keys)} paired predictions to {args.output}")


if __name__ == "__main__":
    main()

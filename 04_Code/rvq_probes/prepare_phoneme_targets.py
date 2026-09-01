#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from rvq_probes.phonemes import load_cmudict, transcript_to_phonemes
from rvq_probes.splits import load_index


def main(args):
    rows = load_index(args.token_index)
    lexicon = load_cmudict(args.lexicon)
    g2p = None
    if args.use_g2p:
        from g2p_en import G2p
        g2p = G2p()
    outputs, failures = [], []
    vocabulary, sources = set(), Counter()
    for row in rows:
        try:
            phones, audit = transcript_to_phonemes(row["text_norm"], lexicon, g2p)
            vocabulary.update(phones)
            sources.update(item["source"] for item in audit)
            outputs.append({
                "utt_id": row["utt_id"], "text_norm": row["text_norm"],
                "phonemes": phones, "word_pronunciations": audit,
                "oov_words": [item["word"] for item in audit if item["source"] == "g2p"],
            })
        except Exception as exc:
            failures.append({"utt_id": row["utt_id"], "error": f"{type(exc).__name__}: {exc}"})
            if not args.skip_errors:
                raise
    if failures and not args.skip_errors:
        raise RuntimeError(f"{len(failures)} phoneme target failures")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, values in (("phoneme_targets.jsonl", outputs), ("failures.jsonl", failures)):
        with (args.output_dir / name).open("w", encoding="utf-8") as handle:
            for value in values:
                handle.write(json.dumps(value, sort_keys=True) + "\n")
    vocabulary_list = ["<blank>"] + sorted(vocabulary)
    (args.output_dir / "phoneme_vocabulary.json").write_text(
        json.dumps(vocabulary_list, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "token_index": str(args.token_index.resolve()), "lexicon": str(args.lexicon.resolve()),
        "lexicon_policy": "first_pronunciation", "stress_removed": True,
        "ctc_silence_excluded": True, "g2p_for_oov": args.use_g2p,
        "utterances": len(outputs), "failures": len(failures),
        "vocabulary_size_including_blank": len(vocabulary_list),
        "word_source_counts": dict(sorted(sources.items())),
    }
    (args.output_dir / "preprocessing_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-index", type=Path, required=True)
    parser.add_argument("--lexicon", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--use-g2p", action="store_true")
    parser.add_argument("--skip-errors", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())

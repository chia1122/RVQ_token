#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from rvq_probes.splits import load_index


def load_first_pronunciations(path: Path):
    pronunciations = {}
    counts = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.strip().split()
            if len(fields) < 2:
                continue
            word = fields[0].lower()
            start = 1
            while start < len(fields):
                try:
                    float(fields[start])
                    start += 1
                except ValueError:
                    break
            phones = fields[start:]
            if not phones:
                continue
            counts[word] = counts.get(word, 0) + 1
            pronunciations.setdefault(word, phones)
    return pronunciations, counts


def main(args):
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory exists: {args.output_dir}")
    rows = load_index(args.token_index)
    if args.limit_per_split:
        rows = [row for split in ("train", "valid", "test")
                for row in [r for r in rows if r["split"] == split][:args.limit_per_split]]
    if args.limit:
        rows = rows[:args.limit]
    first, counts = load_first_pronunciations(args.base_dictionary)
    from g2p_en import G2p
    g2p = G2p()
    vocabulary = sorted({word.lower() for row in rows for word in row["text_norm"].split()})
    selected, audits, oov = {}, [], []
    for word in vocabulary:
        if word in first:
            phones = first[word]
            source = "english_us_arpa_dictionary"
            alternatives = counts[word] - 1
        else:
            phones = [value for value in g2p(word.upper()) if value.strip() and value != " "]
            phones = [value.upper() for value in phones if re.fullmatch(r"[A-Za-z]+[012]?", value)]
            if not phones:
                raise ValueError(f"G2P produced no phones for {word}")
            source = "g2p_en"
            alternatives = 0
            oov.append(word)
        selected[word] = phones
        audits.append({
            "word": word, "source": source, "selected_pronunciation": phones,
            "pronunciation_count": counts.get(word, 0),
            "alternative_pronunciation_count": alternatives,
        })
    corpus = args.output_dir / "corpus"
    corpus.mkdir(parents=True)
    mapping = []
    for row in rows:
        speaker_dir = corpus / row["speaker_id"]
        speaker_dir.mkdir(exist_ok=True)
        stem = row["utt_id"]
        wav_link = speaker_dir / f"{stem}.wav"
        parts = stem.split("-")
        source = args.audio_root / parts[0] / parts[1] / "wav_headMic" / f"{parts[-1]}.wav"
        if not source.is_file():
            raise FileNotFoundError(source)
        os.symlink(source.resolve(), wav_link)
        (speaker_dir / f"{stem}.lab").write_text(row["text_norm"] + "\n", encoding="utf-8")
        mapping.append({
            "utt_id": stem, "speaker_id": row["speaker_id"],
            "source_audio": str(source.resolve()), "mfa_audio": str(wav_link),
        })
    dictionary = args.output_dir / "first_pronunciation.dict"
    with dictionary.open("w", encoding="utf-8") as handle:
        for word in vocabulary:
            handle.write(word + "\t" + " ".join(selected[word]) + "\n")
    for name, values in (("pronunciation_audit.jsonl", audits), ("corpus_mapping.jsonl", mapping)):
        with (args.output_dir / name).open("w", encoding="utf-8") as handle:
            for value in values:
                handle.write(json.dumps(value, sort_keys=True) + "\n")
    summary = {
        "utterances": len(rows), "speakers": len({row["speaker_id"] for row in rows}),
        "vocabulary": len(vocabulary), "oov_words": len(oov), "oov_list": oov,
        "dictionary_policy": "first_listed_pronunciation",
        "base_dictionary": str(args.base_dictionary.resolve()),
        "context_aware_selection": False,
    }
    (args.output_dir / "preparation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-index", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--base-dictionary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--limit-per-split", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())

#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def normalize_chars(text):
    return text.replace(" ", "")


def align(ref, hyp):
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    op = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        op[i][0] = "D"
    for j in range(1, m + 1):
        dp[0][j] = j
        op[0][j] = "I"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                op[i][j] = "M"
            else:
                candidates = [
                    (dp[i - 1][j - 1] + 1, "S"),
                    (dp[i - 1][j] + 1, "D"),
                    (dp[i][j - 1] + 1, "I"),
                ]
                dp[i][j], op[i][j] = min(candidates, key=lambda x: x[0])

    counts = Counter()
    i, j = n, m
    while i > 0 or j > 0:
        operation = op[i][j]
        if operation == "M":
            i -= 1
            j -= 1
        elif operation == "S":
            counts["S"] += 1
            i -= 1
            j -= 1
        elif operation == "D":
            counts["D"] += 1
            i -= 1
        elif operation == "I":
            counts["I"] += 1
            j -= 1
        else:
            break
    return counts


def new_stats():
    return {
        "utterances": 0,
        "reference_chars": 0,
        "errors": Counter(),
        "cer_buckets": Counter(),
    }


def update_stats(stats, ref, hyp):
    counts = align(ref, hyp)
    stats["utterances"] += 1
    stats["reference_chars"] += len(ref)
    stats["errors"].update(counts)

    edits = counts["S"] + counts["D"] + counts["I"]
    cer = edits / len(ref) if ref else 0.0

    if cer == 0:
        stats["cer_buckets"]["exact"] += 1
    if cer < 0.05:
        stats["cer_buckets"]["cer<5%"] += 1
    if cer < 0.10:
        stats["cer_buckets"]["cer<10%"] += 1
    if cer > 0.50:
        stats["cer_buckets"]["cer>50%"] += 1


def summarize(stats):
    s = stats["errors"]["S"]
    d = stats["errors"]["D"]
    i = stats["errors"]["I"]
    total = s + d + i
    ref = stats["reference_chars"]
    utts = stats["utterances"]
    buckets = stats["cer_buckets"]

    return {
        "utterances": utts,
        "reference_chars": ref,
        "substitutions": s,
        "deletions": d,
        "insertions": i,
        "total_errors": total,
        "cer": total / ref if ref else 0.0,
        "substitution_share": s / total if total else 0.0,
        "deletion_share": d / total if total else 0.0,
        "insertion_share": i / total if total else 0.0,
        "exact": buckets["exact"],
        "exact_ratio": buckets["exact"] / utts if utts else 0.0,
        "cer_lt_5": buckets["cer<5%"],
        "cer_lt_5_ratio": buckets["cer<5%"] / utts if utts else 0.0,
        "cer_lt_10": buckets["cer<10%"],
        "cer_lt_10_ratio": buckets["cer<10%"] / utts if utts else 0.0,
        "cer_gt_50": buckets["cer>50%"],
        "cer_gt_50_ratio": buckets["cer>50%"] / utts if utts else 0.0,
    }


def print_section(title, stats):
    x = summarize(stats)
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    print(f"Utterances        : {x['utterances']:,}")
    print(f"Reference chars   : {x['reference_chars']:,}")
    print(f"CER               : {x['cer']:.4f} ({x['cer']:.2%})")
    print(f"Substitutions (S) : {x['substitutions']:,} ({x['substitution_share']:.2%})")
    print(f"Deletions (D)     : {x['deletions']:,} ({x['deletion_share']:.2%})")
    print(f"Insertions (I)    : {x['insertions']:,} ({x['insertion_share']:.2%})")
    print(f"Exact             : {x['exact']:,} ({x['exact_ratio']:.2%})")
    print(f"CER < 5%          : {x['cer_lt_5']:,} ({x['cer_lt_5_ratio']:.2%})")
    print(f"CER < 10%         : {x['cer_lt_10']:,} ({x['cer_lt_10_ratio']:.2%})")
    print(f"CER > 50%         : {x['cer_gt_50']:,} ({x['cer_gt_50_ratio']:.2%})")


def analyze(path, save_json=None):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    overall = new_stats()
    by_severity = defaultdict(new_stats)
    by_speaker = defaultdict(new_stats)

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "reference" not in row or "hypothesis" not in row:
                raise KeyError(f"Missing reference/hypothesis at line {line_number}")

            ref = normalize_chars(row["reference"])
            hyp = normalize_chars(row["hypothesis"])
            severity = str(row.get("severity", "unknown"))
            speaker = str(row.get("speaker_id", "unknown"))

            update_stats(overall, ref, hyp)
            update_stats(by_severity[severity], ref, hyp)
            update_stats(by_speaker[speaker], ref, hyp)

    print("#" * 72)
    print("Severity-aware Character Error Analysis")
    print(f"Input file: {path}")
    print("#" * 72)

    print_section("OVERALL", overall)

    print("\n" + "#" * 72)
    print("BY SEVERITY")
    print("#" * 72)
    for severity in sorted(by_severity):
        print_section(f"SEVERITY: {severity}", by_severity[severity])

    print("\n" + "#" * 72)
    print("BY SPEAKER")
    print("#" * 72)
    for speaker, stats in sorted(
        by_speaker.items(),
        key=lambda kv: summarize(kv[1])["cer"],
        reverse=True,
    ):
        print_section(f"SPEAKER: {speaker}", stats)

    if save_json:
        out = {
            "input_file": str(path),
            "overall": summarize(overall),
            "by_severity": {k: summarize(v) for k, v in sorted(by_severity.items())},
            "by_speaker": {k: summarize(v) for k, v in sorted(by_speaker.items())},
        }
        save_json = Path(save_json)
        save_json.parent.mkdir(parents=True, exist_ok=True)
        save_json.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(f"\nSaved JSON summary to: {save_json}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Severity-aware character error analysis for test_predictions.jsonl"
    )
    parser.add_argument("path", type=Path, help="Path to test_predictions.jsonl")
    parser.add_argument("--save-json", type=Path, help="Optional output JSON summary path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    analyze(args.path, args.save_json)

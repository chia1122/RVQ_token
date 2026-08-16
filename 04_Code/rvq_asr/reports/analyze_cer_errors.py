import argparse
import json
from collections import Counter
from pathlib import Path

"""
python rvq_asr/reports/analyze_cer_errors.py \
  rvq_asr/runs/librispeech_speechtokenizer_q1_seed1337/test_predictions.jsonl
"""



def normalize_chars(text):
    """Remove spaces to match character-level CER evaluation."""
    return text.replace(" ", "")


def align(ref, hyp):
    """
    Character-level Levenshtein alignment.

    Returns counts of:
        S = substitution
        D = deletion
        I = insertion
    """
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

                dp[i][j], op[i][j] = min(
                    candidates,
                    key=lambda x: x[0]
                )

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


def analyze(path):

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    total = Counter()

    reference_chars = 0
    utterances = 0

    cer_buckets = Counter()

    with path.open("r", encoding="utf-8") as f:

        for line_number, line in enumerate(f, start=1):

            if not line.strip():
                continue

            row = json.loads(line)

            if "reference" not in row:
                raise KeyError(
                    f"Missing 'reference' at line {line_number}"
                )

            if "hypothesis" not in row:
                raise KeyError(
                    f"Missing 'hypothesis' at line {line_number}"
                )

            ref = normalize_chars(row["reference"])
            hyp = normalize_chars(row["hypothesis"])

            counts = align(ref, hyp)

            total.update(counts)

            reference_chars += len(ref)
            utterances += 1

            edits = (
                counts["S"]
                + counts["D"]
                + counts["I"]
            )

            cer = edits / len(ref) if ref else 0.0

            if cer == 0:
                cer_buckets["exact"] += 1

            if cer < 0.05:
                cer_buckets["cer<5%"] += 1

            if cer < 0.10:
                cer_buckets["cer<10%"] += 1

            if cer > 0.50:
                cer_buckets["cer>50%"] += 1

    if utterances == 0:
        raise ValueError("No prediction rows found.")

    S = total["S"]
    D = total["D"]
    I = total["I"]

    errors = S + D + I

    cer = (
        errors / reference_chars
        if reference_chars
        else 0.0
    )

    print()
    print("=" * 60)
    print("Character Error Analysis")
    print("=" * 60)

    print(f"Input file        : {path}")
    print(f"Utterances        : {utterances}")
    print(f"Reference chars   : {reference_chars:,}")

    print()
    print("-" * 60)
    print("Edit Operations")
    print("-" * 60)

    print(f"Substitutions (S) : {S:,}")
    print(f"Deletions (D)     : {D:,}")
    print(f"Insertions (I)    : {I:,}")
    print(f"Total errors      : {errors:,}")

    print()
    print(f"CER               : {cer:.4f} ({cer:.2%})")

    if errors > 0:

        print()
        print("-" * 60)
        print("Error Composition")
        print("-" * 60)

        print(
            f"Substitution      : "
            f"{S / errors:.2%}"
        )

        print(
            f"Deletion          : "
            f"{D / errors:.2%}"
        )

        print(
            f"Insertion         : "
            f"{I / errors:.2%}"
        )

    print()
    print("-" * 60)
    print("Utterance-level CER")
    print("-" * 60)

    print(
        f"Exact             : "
        f"{cer_buckets['exact']:,} "
        f"({cer_buckets['exact'] / utterances:.2%})"
    )

    print(
        f"CER < 5%          : "
        f"{cer_buckets['cer<5%']:,} "
        f"({cer_buckets['cer<5%'] / utterances:.2%})"
    )

    print(
        f"CER < 10%         : "
        f"{cer_buckets['cer<10%']:,} "
        f"({cer_buckets['cer<10%'] / utterances:.2%})"
    )

    print(
        f"CER > 50%         : "
        f"{cer_buckets['cer>50%']:,} "
        f"({cer_buckets['cer>50%'] / utterances:.2%})"
    )

    print("=" * 60)
    print()


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Analyze character-level substitution, deletion, "
            "and insertion errors from test_predictions.jsonl."
        )
    )

    parser.add_argument(
        "path",
        type=Path,
        help="Path to test_predictions.jsonl",
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    analyze(args.path)
from __future__ import annotations

from rvq_asr.text import edit_counts


def phoneme_error_metrics(reference: list[str], hypothesis: list[str]) -> dict:
    substitutions, deletions, insertions = edit_counts(reference, hypothesis)
    count = len(reference)
    return {
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "reference_phonemes": count,
        "per": (substitutions + deletions + insertions) / count if count else 0.0,
    }


def binary_counts(reference: list[int], prediction: list[int]) -> dict:
    if len(reference) != len(prediction):
        raise ValueError("Binary metric lengths differ")
    tp = sum(r == 1 and p == 1 for r, p in zip(reference, prediction))
    fp = sum(r == 0 and p == 1 for r, p in zip(reference, prediction))
    fn = sum(r == 1 and p == 0 for r, p in zip(reference, prediction))
    tn = sum(r == 0 and p == 0 for r, p in zip(reference, prediction))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "true_positive": tp, "false_positive": fp, "false_negative": fn,
        "true_negative": tn, "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }

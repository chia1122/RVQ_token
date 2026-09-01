"""MFA phone alignment conversion and boundary metrics."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Sequence


SILENCE_PHONES = {"", "sil", "sp", "spn", "<eps>"}


def timestamp_to_nearest_frame(timestamp: float, frame_duration: float, length: int) -> int:
    """Map seconds to one representation frame using deterministic half-up rounding."""
    if not math.isfinite(timestamp) or timestamp < 0:
        raise ValueError(f"invalid boundary timestamp: {timestamp}")
    if not math.isfinite(frame_duration) or frame_duration <= 0:
        raise ValueError(f"invalid frame duration: {frame_duration}")
    if length <= 0:
        raise ValueError(f"invalid representation length: {length}")
    frame = int(math.floor(timestamp / frame_duration + 0.5))
    return min(max(frame, 0), length - 1)


def read_mfa_phone_intervals(path: str | Path) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = payload.get("tiers", {}).get("phones", {}).get("entries")
    if not isinstance(entries, list):
        raise ValueError(f"missing phones interval tier: {path}")
    intervals = []
    for entry in entries:
        if not isinstance(entry, list) or len(entry) != 3:
            raise ValueError(f"invalid phone interval in {path}: {entry!r}")
        start, end, phone = entry
        start, end = float(start), float(end)
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start <= end):
            raise ValueError(f"invalid phone interval in {path}: {entry!r}")
        intervals.append({"start": start, "end": end, "phone": str(phone)})
    return intervals


def internal_phone_boundary_times(intervals: Sequence[dict]) -> list[float]:
    """Return boundaries between adjacent non-silence phones, excluding utterance edges."""
    result = []
    for left, right in zip(intervals, intervals[1:]):
        if left["phone"].lower() in SILENCE_PHONES or right["phone"].lower() in SILENCE_PHONES:
            continue
        # A gap denotes silence even if MFA does not emit an explicit silence interval.
        if not math.isclose(float(left["end"]), float(right["start"]), abs_tol=1e-4):
            continue
        result.append((float(left["end"]) + float(right["start"])) / 2.0)
    return result


def boundaries_to_frames(times: Iterable[float], frame_duration: float, length: int) -> tuple[list[int], int]:
    mapped = [timestamp_to_nearest_frame(t, frame_duration, length) for t in times]
    unique = sorted(set(mapped))
    return unique, len(mapped) - len(unique)


def match_boundary_frames(reference: Sequence[int], predicted: Sequence[int], tolerance: int) -> dict[str, int | float]:
    """Maximum one-to-one matching within tolerance, prioritizing smaller distance."""
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    ref = sorted(set(int(x) for x in reference))
    pred = sorted(set(int(x) for x in predicted))
    candidates = sorted(
        (abs(p - r), p, r) for p in pred for r in ref if abs(p - r) <= tolerance
    )
    used_pred: set[int] = set()
    used_ref: set[int] = set()
    for _, p, r in candidates:
        if p not in used_pred and r not in used_ref:
            used_pred.add(p)
            used_ref.add(r)
    tp = len(used_pred)
    fp = len(pred) - tp
    fn = len(ref) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}

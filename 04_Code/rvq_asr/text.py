from __future__ import annotations

from dataclasses import dataclass


class CharacterTokenizer:
    """Fixed English character vocabulary matching TORGO normalization."""

    symbols = ["<blank>", "<unk>", " ", "'"] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    def __init__(self):
        self.token_to_id = {token: index for index, token in enumerate(self.symbols)}
        self.id_to_token = dict(enumerate(self.symbols))
        self.blank_id = self.token_to_id["<blank>"]
        self.unk_id = self.token_to_id["<unk>"]

    def __len__(self) -> int:
        return len(self.symbols)

    def encode(self, text: str) -> list[int]:
        return [self.token_to_id.get(character, self.unk_id) for character in text]

    def decode_ctc(self, token_ids: list[int]) -> str:
        output = []
        previous = None
        for token_id in token_ids:
            if token_id != previous and token_id != self.blank_id:
                output.append(self.id_to_token.get(token_id, "<unk>"))
            previous = token_id
        return "".join(output).strip()


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for ref_index, ref_item in enumerate(reference, start=1):
        current = [ref_index]
        for hyp_index, hyp_item in enumerate(hypothesis, start=1):
            current.append(min(
                current[-1] + 1,
                previous[hyp_index] + 1,
                previous[hyp_index - 1] + (ref_item != hyp_item),
            ))
        previous = current
    return previous[-1]


def edit_counts(reference: list[str], hypothesis: list[str]) -> tuple[int, int, int]:
    """Return deterministic substitution, deletion, and insertion counts."""
    previous = [(index, 0, 0, index) for index in range(len(hypothesis) + 1)]
    for ref_index, ref_item in enumerate(reference, start=1):
        current = [(ref_index, 0, ref_index, 0)]
        for hyp_index, hyp_item in enumerate(hypothesis, start=1):
            if ref_item == hyp_item:
                current.append(previous[hyp_index - 1])
                continue
            diagonal = previous[hyp_index - 1]
            deletion = previous[hyp_index]
            insertion = current[-1]
            candidates = (
                (diagonal[0] + 1, diagonal[1] + 1, diagonal[2], diagonal[3]),
                (deletion[0] + 1, deletion[1], deletion[2] + 1, deletion[3]),
                (insertion[0] + 1, insertion[1], insertion[2], insertion[3] + 1),
            )
            current.append(min(candidates, key=lambda value: (value[0], value[2] + value[3], value[1])))
        previous = current
    _, substitutions, deletions, insertions = previous[-1]
    return substitutions, deletions, insertions


def prediction_row(
    utt_id: str, speaker_id: str, condition: str, severity: str,
    reference: str, hypothesis: str,
) -> dict:
    substitutions, deletions, insertions = edit_counts(
        reference.split(), hypothesis.split()
    )
    return {
        "utt_id": utt_id, "speaker_id": speaker_id, "condition": condition,
        "severity": severity, "reference": reference, "hypothesis": hypothesis,
        "substitutions": substitutions, "deletions": deletions,
        "insertions": insertions,
    }


@dataclass
class ErrorRate:
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    reference_units: int = 0

    @property
    def edits(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    def update(self, reference: list[str], hypothesis: list[str]) -> None:
        substitutions, deletions, insertions = edit_counts(reference, hypothesis)
        self.substitutions += substitutions
        self.deletions += deletions
        self.insertions += insertions
        self.reference_units += len(reference)

    def update_words(self, reference: str, hypothesis: str) -> None:
        reference_words = reference.split()
        self.update(reference_words, hypothesis.split())

    def update_characters(self, reference: str, hypothesis: str, ignore_spaces: bool = True) -> None:
        if ignore_spaces:
            reference = reference.replace(" ", "")
            hypothesis = hypothesis.replace(" ", "")
        self.update(list(reference), list(hypothesis))

    @property
    def value(self) -> float:
        return self.edits / self.reference_units if self.reference_units else 0.0

    def counts_and_rates(self) -> dict[str, float | int]:
        denominator = self.reference_units
        return {
            "substitutions": self.substitutions,
            "deletions": self.deletions,
            "insertions": self.insertions,
            "substitution_rate": self.substitutions / denominator if denominator else 0.0,
            "deletion_rate": self.deletions / denominator if denominator else 0.0,
            "insertion_rate": self.insertions / denominator if denominator else 0.0,
        }

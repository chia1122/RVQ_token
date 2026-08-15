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


@dataclass
class ErrorRate:
    edits: int = 0
    reference_units: int = 0

    def update_words(self, reference: str, hypothesis: str) -> None:
        reference_words = reference.split()
        self.edits += edit_distance(reference_words, hypothesis.split())
        self.reference_units += len(reference_words)

    def update_characters(self, reference: str, hypothesis: str, ignore_spaces: bool = True) -> None:
        if ignore_spaces:
            reference = reference.replace(" ", "")
            hypothesis = hypothesis.replace(" ", "")
        self.edits += edit_distance(list(reference), list(hypothesis))
        self.reference_units += len(reference)

    @property
    def value(self) -> float:
        return self.edits / self.reference_units if self.reference_units else 0.0

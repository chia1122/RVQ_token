from __future__ import annotations

import re
from pathlib import Path

STRESS = re.compile(r"[012]$")
SILENCE = {"SIL", "SP", "SPN", "<SIL>", "NOI"}


def normalize_phone(phone: str) -> str:
    return STRESS.sub("", phone.strip().upper())


def load_cmudict(path: Path) -> dict[str, list[list[str]]]:
    lexicon: dict[str, list[list[str]]] = {}
    with path.open(encoding="latin-1") as handle:
        for line in handle:
            if not line.strip() or line.startswith(";;;"):
                continue
            fields = line.split()
            if len(fields) < 3 or not fields[1].isdigit():
                continue
            word = re.sub(r"\(\d+\)$", "", fields[0].upper())
            phones = [normalize_phone(value) for value in fields[2:]]
            phones = [value for value in phones if value and value not in SILENCE]
            if phones and phones not in lexicon.setdefault(word, []):
                lexicon[word].append(phones)
    if not lexicon:
        raise ValueError(f"No pronunciations loaded from {path}")
    return lexicon


def transcript_to_phonemes(text: str, lexicon: dict[str, list[list[str]]], g2p=None) -> tuple[list[str], list[dict]]:
    phones: list[str] = []
    audit: list[dict] = []
    for word in text.upper().split():
        pronunciations = lexicon.get(word)
        if pronunciations:
            selected = pronunciations[0]
            source = "lexicon"
            alternative_count = len(pronunciations) - 1
        else:
            if g2p is None:
                raise ValueError(f"OOV without G2P: {word}")
            selected = [
                normalize_phone(value) for value in g2p(word)
                if value.strip() and value != " "
            ]
            selected = [value for value in selected if value not in SILENCE]
            if not selected:
                raise ValueError(f"G2P produced no phonemes for {word}")
            source = "g2p"
            alternative_count = 0
        phones.extend(selected)
        audit.append({
            "word": word, "source": source, "selected_pronunciation": selected,
            "pronunciation_count": len(pronunciations) if pronunciations else 0,
            "alternative_pronunciation_count": alternative_count,
        })
    if not phones:
        raise ValueError("Transcript produced an empty phoneme sequence")
    return phones, audit


class PhonemeTokenizer:
    def __init__(self, symbols: list[str]):
        if not symbols or symbols[0] != "<blank>":
            raise ValueError("Phoneme vocabulary must start with <blank>")
        self.symbols = symbols
        self.token_to_id = {value: index for index, value in enumerate(symbols)}
        if len(self.token_to_id) != len(symbols):
            raise ValueError("Duplicate phoneme symbol")
        self.blank_id = 0

    def __len__(self):
        return len(self.symbols)

    def encode(self, phones: list[str]) -> list[int]:
        try:
            return [self.token_to_id[value] for value in phones]
        except KeyError as exc:
            raise ValueError(f"Unknown phoneme: {exc.args[0]}") from exc

    def decode_ctc(self, ids: list[int]) -> list[str]:
        result = []
        previous = None
        for value in ids:
            if value != previous and value != self.blank_id:
                if not 0 <= value < len(self.symbols):
                    raise ValueError(f"Invalid phoneme ID: {value}")
                result.append(self.symbols[value])
            previous = value
        return result

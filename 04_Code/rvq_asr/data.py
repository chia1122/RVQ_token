from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .text import CharacterTokenizer


def load_token_file(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions before weights_only was introduced.
        return torch.load(path, map_location="cpu")


class RVQTokenDataset(Dataset):
    def __init__(
        self,
        token_index: Path,
        token_root: Path,
        split: str,
        num_rvq_layers: int,
        tokenizer: CharacterTokenizer,
    ):
        self.token_root = token_root.resolve()
        self.num_rvq_layers = num_rvq_layers
        self.tokenizer = tokenizer
        self.rows = []
        with token_index.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("split") != split:
                    continue
                required = {"utt_id", "token_path", "num_codebooks", "speaker_id", "severity", "text_norm"}
                missing = required - set(row)
                if missing:
                    raise ValueError(f"Token index line {line_number} is missing {sorted(missing)}")
                if int(row["num_codebooks"]) < num_rvq_layers:
                    raise ValueError(
                        f"{row['utt_id']} has {row['num_codebooks']} codebooks, "
                        f"cannot use first {num_rvq_layers}"
                    )
                self.rows.append(row)
        if not self.rows:
            raise ValueError(f"No rows found for split={split!r}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        token_path = (self.token_root / row["token_path"]).resolve()
        try:
            token_path.relative_to(self.token_root)
        except ValueError as exc:
            raise ValueError(f"Token path escapes token root: {row['token_path']}") from exc
        payload = load_token_file(token_path)
        codes = payload["codes"]
        if codes.ndim != 2 or codes.shape[1] < self.num_rvq_layers:
            raise ValueError(f"Invalid [T,N] codes in {token_path}: {tuple(codes.shape)}")
        codes = codes[:, : self.num_rvq_layers].to(torch.long)
        targets = torch.tensor(self.tokenizer.encode(row["text_norm"]), dtype=torch.long)
        if targets.numel() == 0 or codes.shape[0] < targets.numel():
            raise ValueError(
                f"CTC length invalid for {row['utt_id']}: frames={codes.shape[0]}, targets={targets.numel()}"
            )
        return {
            "codes": codes,
            "targets": targets,
            "utt_id": row["utt_id"],
            "speaker_id": row["speaker_id"],
            "condition": row.get("condition", "unknown"),
            "severity": row["severity"],
            "text": row["text_norm"],
        }


class CTCBatchCollator:
    def __init__(self, codebook_size: int):
        self.code_padding_id = codebook_size

    def __call__(self, samples: list[dict]) -> dict:
        codes = pad_sequence(
            [sample["codes"] for sample in samples],
            batch_first=True,
            padding_value=self.code_padding_id,
        )
        input_lengths = torch.tensor([sample["codes"].shape[0] for sample in samples], dtype=torch.long)
        target_lengths = torch.tensor([sample["targets"].numel() for sample in samples], dtype=torch.long)
        targets = torch.cat([sample["targets"] for sample in samples])
        return {
            "codes": codes,
            "input_lengths": input_lengths,
            "targets": targets,
            "target_lengths": target_lengths,
            "utt_ids": [sample["utt_id"] for sample in samples],
            "speaker_ids": [sample["speaker_id"] for sample in samples],
            "conditions": [sample["condition"] for sample in samples],
            "severities": [sample["severity"] for sample in samples],
            "texts": [sample["text"] for sample in samples],
        }

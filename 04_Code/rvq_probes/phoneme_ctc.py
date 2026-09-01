from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from rvq_probes.representation import FrozenCodebook, index_token_paths, load_rvq_representation


class PhonemeDataset(Dataset):
    def __init__(self, rows, token_root: Path, targets_path: Path, split: str,
                 rvq_layer: int, codebook: FrozenCodebook, limit: int = 0):
        targets = {
            row["utt_id"]: row
            for row in (json.loads(line) for line in targets_path.read_text().splitlines() if line.strip())
        }
        self.rows = [row for row in rows if row["split"] == split]
        if limit:
            self.rows = self.rows[:limit]
        self.token_root = token_root.resolve()
        self.token_paths = index_token_paths(self.token_root)
        self.targets = targets
        self.rvq_layer = rvq_layer
        self.codebook = codebook
        if not self.rows:
            raise ValueError(f"No rows for split={split}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = self.token_paths[row["utt_id"]]
        representation = load_rvq_representation(path, self.rvq_layer, self.codebook)
        phones = self.targets[row["utt_id"]]["phonemes"]
        return {"representation": representation, "phonemes": phones, "row": row}


class PhonemeCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, samples):
        representations = pad_sequence(
            [sample["representation"] for sample in samples], batch_first=True
        )
        input_lengths = torch.tensor([len(sample["representation"]) for sample in samples])
        encoded = [torch.tensor(self.tokenizer.encode(sample["phonemes"])) for sample in samples]
        target_lengths = torch.tensor([len(value) for value in encoded])
        if torch.any(input_lengths < target_lengths):
            raise ValueError("CTC input length is shorter than target length")
        return {
            "representations": representations, "input_lengths": input_lengths,
            "targets": torch.cat(encoded), "target_lengths": target_lengths,
            "phonemes": [sample["phonemes"] for sample in samples],
            "rows": [sample["row"] for sample in samples],
        }


class PhonemeCTCProbe(nn.Module):
    def __init__(self, input_dim: int, vocabulary_size: int, bottleneck_dim: int = 128,
                 temporal_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        blocks = []
        for _ in range(temporal_layers):
            blocks.extend([
                nn.Conv1d(bottleneck_dim, bottleneck_dim, 5, padding=2),
                nn.GELU(), nn.Dropout(dropout),
            ])
        self.input_norm = nn.LayerNorm(input_dim)
        self.projection = nn.Linear(input_dim, bottleneck_dim)
        self.temporal = nn.Sequential(*blocks)
        self.classifier = nn.Linear(bottleneck_dim, vocabulary_size)

    def forward(self, values, lengths):
        hidden = torch.nn.functional.gelu(self.projection(self.input_norm(values)))
        hidden = self.temporal(hidden.transpose(1, 2)).transpose(1, 2)
        return self.classifier(hidden), lengths

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from rvq_probes.representation import FrozenCodebook, index_token_paths, load_rvq_representation


class BoundaryDataset(Dataset):
    def __init__(self, rows, token_root: Path, targets_path: Path, split: str,
                 rvq_layer: int, codebook: FrozenCodebook, limit: int = 0):
        targets = {r["utt_id"]: r for r in (json.loads(x) for x in targets_path.read_text().splitlines() if x)}
        self.rows = [r for r in rows if r["split"] == split and r["utt_id"] in targets]
        if limit:
            self.rows = self.rows[:limit]
        if not self.rows:
            raise ValueError(f"No aligned rows for split={split}")
        self.token_root, self.targets = token_root.resolve(), targets
        self.token_paths = index_token_paths(self.token_root)
        self.rvq_layer, self.codebook = rvq_layer, codebook

    def __len__(self): return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = self.token_paths[row["utt_id"]]
        representation = load_rvq_representation(path, self.rvq_layer, self.codebook)
        target = torch.zeros(len(representation), dtype=torch.float32)
        frames = self.targets[row["utt_id"]]["boundary_frames"]
        if frames:
            target[torch.tensor(frames)] = 1.0
        return {"representation": representation, "target": target, "frames": frames, "row": row}


def collate_boundary(samples):
    lengths = torch.tensor([len(x["target"]) for x in samples])
    values = pad_sequence([x["representation"] for x in samples], batch_first=True)
    targets = pad_sequence([x["target"] for x in samples], batch_first=True)
    mask = torch.arange(values.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
    return {"representations": values, "targets": targets, "mask": mask, "lengths": lengths,
            "frames": [x["frames"] for x in samples], "rows": [x["row"] for x in samples]}


class PhonemeBoundaryProbe(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.project = nn.Linear(input_dim, hidden_dim)
        self.local = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, values):
        hidden = torch.nn.functional.gelu(self.project(self.norm(values)))
        hidden = torch.nn.functional.gelu(self.local(hidden.transpose(1, 2)).transpose(1, 2))
        return self.classifier(self.dropout(hidden)).squeeze(-1)

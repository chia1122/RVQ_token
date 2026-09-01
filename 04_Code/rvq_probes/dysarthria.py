from __future__ import annotations
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import Dataset
from rvq_probes.representation import FrozenCodebook, index_token_paths, load_rvq_representation


class DysarthriaDataset(Dataset):
    def __init__(self, rows, token_root: Path, split: str, rvq_layer: int,
                 codebook: FrozenCodebook, limit: int = 0):
        self.rows = [r for r in rows if r["split"] == split]
        if limit: self.rows = self.rows[:limit]
        if not self.rows: raise ValueError(f"No rows for split={split}")
        self.paths = index_token_paths(token_root.resolve())
        self.rvq_layer, self.codebook = rvq_layer, codebook
        self.pooled = []
        with torch.inference_mode():
            for row in self.rows:
                value = load_rvq_representation(self.paths[row["utt_id"]], self.rvq_layer, self.codebook)
                self.pooled.append(value.mean(dim=0))

    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        row = self.rows[index]
        return {"pooled": self.pooled[index], "label": int(row["condition"] == "dysarthric"), "row": row}


def collate_dysarthria(samples):
    return {"pooled": torch.stack([x["pooled"] for x in samples]),
            "labels": torch.tensor([x["label"] for x in samples]),
            "rows": [x["row"] for x in samples]}


class DysarthriaProbe(nn.Module):
    def __init__(self, input_dim):
        super().__init__(); self.norm = nn.LayerNorm(input_dim); self.classifier = nn.Linear(input_dim, 2)
    def forward(self, pooled):
        return self.classifier(self.norm(pooled))

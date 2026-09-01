from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch import nn


def index_token_paths(token_root: Path) -> dict[str, Path]:
    """Index immutable master tokens by utterance ID, independent of rotation split paths."""
    paths = {}
    for path in token_root.glob("*/*/*.pt"):
        if path.stem in paths:
            raise ValueError(f"Duplicate token utterance ID: {path.stem}")
        paths[path.stem] = path.resolve()
    if not paths:
        raise ValueError(f"No token files under {token_root}")
    return paths


def load_token_payload(path: Path) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "codes" not in payload:
        raise ValueError(f"Invalid token payload: {path}")
    return payload


def select_individual_codes(codes: torch.Tensor, rvq_layer: int, num_codebooks: int = 8) -> torch.Tensor:
    if codes.ndim != 2:
        raise ValueError(f"Expected codes [T,N], found {tuple(codes.shape)}")
    if codes.shape[0] == 0:
        raise ValueError("Empty RVQ sequence")
    if codes.shape[1] != num_codebooks:
        raise ValueError(f"Expected {num_codebooks} codebooks, found {codes.shape[1]}")
    if not 1 <= rvq_layer <= num_codebooks:
        raise ValueError(f"rvq_layer must be in [1,{num_codebooks}], found {rvq_layer}")
    return codes[:, rvq_layer - 1].to(torch.long)


class FrozenCodebook(nn.Module):
    """One frozen codec-native codebook table."""

    def __init__(self, table: torch.Tensor, rvq_layer: int, codebook_size: int):
        super().__init__()
        if table.ndim != 2 or table.shape[0] != codebook_size:
            raise ValueError(f"Invalid codebook table shape: {tuple(table.shape)}")
        if not torch.isfinite(table).all():
            raise ValueError("Codebook contains NaN/Inf")
        self.rvq_layer = rvq_layer
        self.codebook_size = codebook_size
        self.embedding_dim = int(table.shape[1])
        self.register_buffer("table", table.detach().to(torch.float32).clone())

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.numel() == 0:
            raise ValueError("Empty token sequence")
        if token_ids.dtype != torch.long:
            token_ids = token_ids.to(torch.long)
        low, high = int(token_ids.min()), int(token_ids.max())
        if low < 0 or high >= self.codebook_size:
            raise ValueError(f"Token ID range [{low},{high}] outside [0,{self.codebook_size - 1}]")
        values = torch.nn.functional.embedding(token_ids, self.table)
        if values.shape[-1] != self.embedding_dim or not torch.isfinite(values).all():
            raise ValueError("Invalid retrieved embeddings")
        return values


def load_speechtokenizer_codebook(
    config_path: Path, checkpoint_path: Path, rvq_layer: int,
) -> tuple[FrozenCodebook, dict]:
    from speechtokenizer import SpeechTokenizer

    config = json.loads(config_path.read_text(encoding="utf-8"))
    num_codebooks = int(config["n_q"])
    codebook_size = int(config["codebook_size"])
    if not 1 <= rvq_layer <= num_codebooks:
        raise ValueError(f"rvq_layer must be in [1,{num_codebooks}]")
    model = SpeechTokenizer.load_from_checkpoint(config_path, checkpoint_path)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    layer = model.quantizer.vq.layers[rvq_layer - 1]
    table = layer.codebook.detach().cpu()
    codebook = FrozenCodebook(table, rvq_layer, codebook_size)
    metadata = {
        "codec": "speechtokenizer_hubert_avg",
        "codec_config": str(config_path.resolve()),
        "codec_checkpoint": str(checkpoint_path.resolve()),
        "codec_checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "num_codebooks": num_codebooks,
        "codebook_size": codebook_size,
        "embedding_dim": codebook.embedding_dim,
        "sample_rate": int(config["sample_rate"]),
        "hop_length": 320,
        "token_frame_rate": float(int(config["sample_rate"]) / 320),
        "frame_duration_seconds": 320 / int(config["sample_rate"]),
        "representation": "frozen_codec_native_individual_codebook",
    }
    del model
    return codebook, metadata


def load_rvq_representation(
    token_path: Path, rvq_layer: int, codebook: FrozenCodebook, num_codebooks: int = 8,
) -> torch.Tensor:
    payload = load_token_payload(token_path)
    token_ids = select_individual_codes(payload["codes"], rvq_layer, num_codebooks)
    values = codebook(token_ids)
    if values.shape[0] != token_ids.shape[0]:
        raise ValueError("Representation length mismatch")
    return values

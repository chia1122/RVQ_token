from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
from torch import nn


class RVQTransformerCTC(nn.Module):
    def __init__(
        self,
        codebook_size: int,
        num_rvq_layers: int,
        vocabulary_size: int,
        max_rvq_layers: Optional[int] = None,
        model_dim: int = 256,
        num_encoder_layers: int = 4,
        num_heads: int = 4,
        feedforward_dim: int = 1024,
        dropout: float = 0.1,
        max_frames: int = 20000,
        time_reduction: int = 4,
        subsampling: str = "conv",
        active_rvq_layers: Optional[Sequence[int]] = None,
        layer_fusion: str = "sum",
    ):
        super().__init__()
        self.num_rvq_layers = num_rvq_layers
        self.max_rvq_layers = max_rvq_layers or num_rvq_layers
        if not 1 <= self.num_rvq_layers <= self.max_rvq_layers:
            raise ValueError("num_rvq_layers must be between 1 and max_rvq_layers")
        self.active_rvq_layers = tuple(
            range(self.num_rvq_layers) if active_rvq_layers is None else active_rvq_layers
        )
        if not self.active_rvq_layers:
            raise ValueError("active_rvq_layers cannot be empty")
        if len(set(self.active_rvq_layers)) != len(self.active_rvq_layers):
            raise ValueError("active_rvq_layers cannot contain duplicates")
        if min(self.active_rvq_layers) < 0 or max(self.active_rvq_layers) >= self.num_rvq_layers:
            raise ValueError("active_rvq_layers must index the configured input layers")
        if layer_fusion not in {"sum", "learned"}:
            raise ValueError("layer_fusion must be 'sum' or 'learned'")
        self.layer_fusion = layer_fusion
        if time_reduction < 1:
            raise ValueError("time_reduction must be positive")
        if subsampling not in {"average", "conv"}:
            raise ValueError("subsampling must be 'average' or 'conv'")
        if subsampling == "conv" and time_reduction & (time_reduction - 1):
            raise ValueError("conv subsampling requires a power-of-two time_reduction")
        self.time_reduction = time_reduction
        self.subsampling = subsampling
        self.code_padding_id = codebook_size
        self.embeddings = nn.ModuleList([
            nn.Embedding(codebook_size + 1, model_dim, padding_idx=codebook_size)
            for _ in range(self.max_rvq_layers)
        ])
        self.layer_weight_logits = (
            nn.Parameter(torch.zeros(len(self.active_rvq_layers)))
            if self.layer_fusion == "learned" else None
        )
        self.input_norm = nn.LayerNorm(model_dim)
        conv_stages = int(math.log2(time_reduction)) if subsampling == "conv" else 0
        self.subsampling_convs = nn.ModuleList([
            nn.Conv1d(model_dim, model_dim, kernel_size=5, stride=2, padding=2)
            for _ in range(conv_stages)
        ])
        self.subsampling_norms = nn.ModuleList([
            nn.LayerNorm(model_dim) for _ in range(conv_stages)
        ])
        position = torch.arange(max_frames).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2) * (-math.log(10000.0) / model_dim))
        positional_encoding = torch.zeros(max_frames, model_dim)
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term[: positional_encoding[:, 1::2].shape[1]])
        self.register_buffer("positional_encoding", positional_encoding, persistent=False)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_encoder_layers, norm=nn.LayerNorm(model_dim))
        self.dropout = nn.Dropout(dropout)
        self.ctc_projection = nn.Linear(model_dim, vocabulary_size)

    def forward(self, codes: torch.Tensor, input_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if codes.ndim != 3 or codes.shape[2] != self.num_rvq_layers:
            raise ValueError(
                f"Expected codes [B,T,{self.num_rvq_layers}], found {tuple(codes.shape)}"
            )
        layer_hidden = [
            self.embeddings[layer_index](codes[:, :, layer_index])
            for layer_index in self.active_rvq_layers
        ]
        if self.layer_fusion == "learned":
            weights = torch.softmax(self.layer_weight_logits, dim=0)
            hidden = sum(weight * value for weight, value in zip(weights, layer_hidden))
        else:
            hidden = sum(layer_hidden) / math.sqrt(len(layer_hidden))
        hidden = self.input_norm(hidden)
        if self.subsampling == "conv":
            for convolution, normalization in zip(self.subsampling_convs, self.subsampling_norms):
                hidden = convolution(hidden.transpose(1, 2)).transpose(1, 2)
                hidden = torch.nn.functional.gelu(normalization(hidden))
                input_lengths = torch.div(input_lengths + 1, 2, rounding_mode="floor")
        elif self.time_reduction > 1:
            hidden = torch.nn.functional.avg_pool1d(
                hidden.transpose(1, 2),
                kernel_size=self.time_reduction,
                stride=self.time_reduction,
                ceil_mode=True,
            ).transpose(1, 2)
            input_lengths = torch.div(
                input_lengths + self.time_reduction - 1,
                self.time_reduction,
                rounding_mode="floor",
            )
        if hidden.shape[1] > self.positional_encoding.shape[0]:
            raise ValueError(
                f"Reduced input has {hidden.shape[1]} frames, exceeding configured maximum"
            )
        hidden = self.dropout(hidden + self.positional_encoding[: hidden.shape[1]])
        time = torch.arange(hidden.shape[1], device=codes.device).unsqueeze(0)
        padding_mask = time >= input_lengths.unsqueeze(1)
        hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)
        return self.ctc_projection(hidden), input_lengths

    def normalized_layer_weights(self) -> list[float]:
        if self.layer_fusion == "learned":
            active_weights = torch.softmax(self.layer_weight_logits.detach(), dim=0).cpu().tolist()
        else:
            active_weights = [1.0 / len(self.active_rvq_layers)] * len(self.active_rvq_layers)
        weights = [0.0] * self.num_rvq_layers
        for layer_index, weight in zip(self.active_rvq_layers, active_weights):
            weights[layer_index] = float(weight)
        return weights

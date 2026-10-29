"""
The temporal models (Step 4): a window of frame features x (batch, frames, D) and a mask (True = real frame) in, one fake logit per window out.

TemporalLSTM: 1-2 layer bidirectional LSTM.
TemporalTransformer: 2 layers, 4 heads, position embedding and a classification token.
"""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class TemporalLSTM(nn.Module):
    def __init__(self, d_in: int, hidden: int = 256, layers: int = 1, bidirectional: bool = True,
                 dropout: float = 0.2):
        super().__init__()
        self.directions = 2 if bidirectional else 1
        self.norm = nn.LayerNorm(d_in)  # features from different backbones come in different scales
        self.lstm = nn.LSTM(d_in, hidden, layers, batch_first=True, bidirectional=bidirectional,
                            dropout=dropout if layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden * self.directions, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        lengths = mask.sum(1).clamp(min=1).cpu()  # padding is skipped, not read as frames
        packed = pack_padded_sequence(self.norm(x), lengths, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        # h: (layers * directions, batch, hidden); the last layer's forward and backward states
        last = h[-self.directions:].permute(1, 0, 2).reshape(x.shape[0], -1)
        return self.head(self.dropout(last)).squeeze(1)


class TemporalTransformer(nn.Module):
    def __init__(self, d_in: int, d_model: int = 256, heads: int = 4, layers: int = 2, dim_ff: int = 512,
                 dropout: float = 0.1, max_len: int = 32):
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.project = nn.Linear(d_in, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.position = nn.Parameter(torch.zeros(1, max_len + 1, d_model))  # slot 0 is the classification token
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, heads, dim_ff, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, frames, _ = x.shape
        h = torch.cat([self.cls.expand(batch, -1, -1), self.project(self.norm(x))], dim=1)
        h = h + self.position[:, :frames + 1]
        ignore = torch.cat([torch.zeros(batch, 1, dtype=torch.bool, device=x.device), ~mask], dim=1)
        h = self.encoder(h, src_key_padding_mask=ignore)
        return self.head(self.out_norm(h[:, 0])).squeeze(1)


MODELS = {"lstm": TemporalLSTM, "transformer": TemporalTransformer}


def build_model(name: str, d_in: int, **options) -> nn.Module:
    """A temporal model by name; `options` are its constructor arguments."""
    return MODELS[name](d_in, **options)

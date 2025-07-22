"""
model.py – Arquitectura Transformer para series temporales de E2.
"""

from __future__ import annotations
import torch
import torch.nn as nn

class HormoneTransformer(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_patients: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        self.num_proj = nn.Linear(num_features, d_model)
        self.pid_emb = nn.Embedding(num_patients, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, activation="relu", batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x, pid, src_key_padding_mask=None):
        """
        x : (b, S, F)
        pid : (b,)
        src_key_padding_mask : (b, S) con True en posiciones a ignorar
        """
        b, s, _ = x.shape
        num_vec = self.num_proj(x)                                # (b, S, d)
        pid_vec = self.pid_emb(pid).unsqueeze(1).expand(b, s, -1) # (b, S, d)
        h = num_vec + pid_vec
        enc = self.encoder(h, src_key_padding_mask=src_key_padding_mask)
        out = self.head(enc).squeeze(-1)                          # (b, S)
        return out

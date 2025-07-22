"""
model.py – Transformer con embeddings de paciente, fase y posición sen-cos.
"""
from __future__ import annotations
import torch, math, torch.nn as nn

class HormoneTransformer(nn.Module):
    def __init__(
        self, *, num_features:int, num_patients:int,
        d_model:int, nhead:int, num_layers:int, d_ff:int, dropout:float
    ):
        super().__init__()
        self.num_proj = nn.Linear(num_features, d_model, bias=False)
        self.pos_dropout = nn.Dropout(dropout)

        self.pid_emb   = nn.Embedding(num_patients, d_model)
        self.phase_emb = nn.Embedding(3, d_model)        # 0,1,2

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, d_ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.head    = nn.Linear(d_model, 1)

    def forward(self, x_num, pid, phase,
                src_key_padding_mask=None):
        """
        x_num : (B,S,F)  – características normalizadas
        pid   : (B,)     – índice paciente
        phase : (B,S)    – 0/1/2
        """
        B,S,_ = x_num.shape
        num_vec = self.num_proj(x_num)                   # (B,S,D)
        pid_vec = self.pid_emb(pid).unsqueeze(1).expand(-1,S,-1)
        ph_vec  = self.phase_emb(phase)                  # (B,S,D)
        src = self.pos_dropout(num_vec + pid_vec + ph_vec)
        enc = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
        return self.head(enc).squeeze(-1)                # (B,S)

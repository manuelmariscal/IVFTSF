class HormoneTransformer(nn.Module):
    def __init__(self, num_features:int, num_patients:int,
                 d_model:int=64, nhead:int=4, num_layers:int=4,
                 d_ff:int=256, dropout:float=0.1):
        super().__init__()

        # ── proyección de features numéricas ───────────────────────────────
        self.num_proj = nn.Linear(num_features, d_model)

        # ── embeddings de paciente y fase ─────────────────────────────────
        self.pid_emb   = nn.Embedding(num_patients, d_model)
        self.phase_emb = nn.Embedding(3, d_model)        # 0/1/2

        # ── encoder Transformer ───────────────────────────────────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, d_ff, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)

        # ── cabeza de regresión ───────────────────────────────────────────
        self.head = nn.Linear(d_model, 1)

    # --------------------------- forward ----------------------------------
    def forward(
        self,
        x:  torch.Tensor,          # (B,S,F)  features
        pid:torch.Tensor,          # (B,)
        *,                         # keyword‑only a partir de aquí
        days:  torch.Tensor,       # (B,S)  (no se usa, pero queda para futuro)
        phase: torch.Tensor,       # (B,S)  0/1/2
        src_key_padding_mask=None,
    ):
        B, S, _ = x.shape
        h = self.num_proj(x)                       # (B,S,D)

        # Añadimos embeddings
        pid_vec   = self.pid_emb(pid).unsqueeze(1)     # (B,1,D) ⇒ broadcast
        phase_vec = self.phase_emb(phase)              # (B,S,D)
        h = h + pid_vec + phase_vec

        h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)
        y = self.head(h).squeeze(-1)                   # (B,S)
        return y

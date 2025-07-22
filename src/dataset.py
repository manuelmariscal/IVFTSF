"""
dataset.py – Carga, normaliza e incluye embeddings de fase & posición.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ── columnas de entrada (E2 es la etiqueta) ───────────────────────────────
CSV_FEATURES = [
    "monitoring_visit", "age",
    "Follicle 6-7mm","Follicle 8-9mm","Follicle 10-11mm","Follicle 12-13mm",
    "Follicle 14-15mm","Follicle 16-17mm","Follicle 18-19mm","Follicle 20-21mm",
    "Follicle 22-23","Follicle >23mm","Follicle <7mm","Follicle 8-11mm",
    "Follicle 12-17mm","Follicle 18-21mm","Follicle >22mm",
]

# ── dataset ───────────────────────────────────────────────────────────────
class HormoneDataset(Dataset):
    """
    Para cada paciente devuelve:
        x      : (S,F)  características normalizadas
        y      : (S,)   concentración de E2 (pg/mL)
        mask   : (S,)   bool (True = valor real)
        days   : (S,)   día del ciclo
        phase  : (S,)   0‑folicular | 1‑ovulatoria | 2‑lútea
        pid    : ()     índice entero de paciente
    """
    def __init__(self, df: pd.DataFrame):
        df = df.copy().sort_values(["id", "measure_day"])

        # ── codificación posicional seno/cos y fase del ciclo ────────────
        ang = 2 * math.pi * (df["measure_day"] - 1) / 28.0
        df["pos_sin"], df["pos_cos"] = np.sin(ang), np.cos(ang)
        df["phase_idx"] = np.select(
            [df["measure_day"] <= 9,
             df["measure_day"] <= 14],
            [0,                # fase folicular
             1],               # fase ovulatoria
            default=2,         # fase lútea
        )

        # ── normalización robusta ───────────────────────────────────────
        self.inp_cols = CSV_FEATURES + ["pos_sin", "pos_cos"]
        feats         = df[self.inp_cols].astype(float)

        self.means = feats.mean()
        self.stds  = feats.std().replace(0, 1.0)          # evita división por cero

        df[self.inp_cols] = (feats - self.means) / self.stds
        df[self.inp_cols] = (df[self.inp_cols]          # ← NUEVO: rellena huecos
                             .replace([np.inf, -np.inf], np.nan)
                             .fillna(0.0))

        # ── agrupa por paciente ─────────────────────────────────────────
        self.groups       = [g for _, g in df.groupby("id")]
        self.max_len      = max(len(g) for g in self.groups)
        self.pid2idx      = {pid: i for i, pid in enumerate(df["id"].unique())}
        self.num_patients = len(self.pid2idx)

    # ---------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int):
        g         = self.groups[idx]
        L         = len(g)             # longitud real del paciente
        S, F      = self.max_len, len(self.inp_cols)

        x   = np.zeros((S, F), np.float32)
        y   = np.zeros((S,),   np.float32)
        msk = np.zeros((S,),   bool)
        day = np.zeros((S,),   np.int64)
        ph  = np.zeros((S,),   np.int64)

        x[:L]   = g[self.inp_cols].to_numpy(dtype=np.float32)
        y[:L]   = g["e2"].to_numpy(dtype=np.float32)
        day[:L] = g["measure_day"].to_numpy(dtype=np.int64)
        ph[:L]  = g["phase_idx"].to_numpy(dtype=np.int64)
        msk[:L] = True

        return {
            "x":    torch.from_numpy(x),
            "y":    torch.from_numpy(y),
            "mask": torch.from_numpy(msk),
            "days": torch.from_numpy(day),
            "phase":torch.from_numpy(ph),
            "pid":  torch.tensor(self.pid2idx[g["id"].iloc[0]], dtype=torch.long),
        }

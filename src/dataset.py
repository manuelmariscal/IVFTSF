"""
dataset.py – Carga y normaliza datos para el Transformer.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Columnas originales (sin e2)
CSV_FEATURES = [
    "monitoring_visit",
    "age",
    "Follicle 6-7mm","Follicle 8-9mm","Follicle 10-11mm","Follicle 12-13mm",
    "Follicle 14-15mm","Follicle 16-17mm","Follicle 18-19mm","Follicle 20-21mm",
    "Follicle 22-23","Follicle >23mm","Follicle <7mm","Follicle 8-11mm",
    "Follicle 12-17mm","Follicle 18-21mm","Follicle >22mm",
]

class HormoneDataset(Dataset):
    """
    Prepara tensores:
        x : (b, S, F) características (sin e2) + pos_sin,pos_cos si existen
        y : (b, S)    valores de e2
        mask : (b, S) posiciones válidas
        days : (b, S) día de medición para regularización gold standard
        pid : (b,)    índice entero de paciente
    """
    def __init__(self, df: pd.DataFrame):
        df = df.copy()
        # ordenar por paciente y día
        df.sort_values(["id", "measure_day"], inplace=True)

        # mapping de paciente → índice
        self.pid2idx = {pid: i for i, pid in enumerate(df["id"].unique())}
        self.num_patients = len(self.pid2idx)

        # estadísticas para normalizar (solo características numéricas input)
        feat_cols = CSV_FEATURES.copy()
        # añadir posicionales si existen
        if "pos_sin" in df.columns and "pos_cos" in df.columns:
            feat_cols += ["pos_sin", "pos_cos"]

        self.features = feat_cols
        feats = df[feat_cols].astype(float)
        self.means = feats.mean()
        self.stds = feats.std().replace(0, 1.0)

        df_norm = df.copy()
        df_norm[feat_cols] = (feats - self.means) / self.stds

        # agrupar por paciente
        groups = []
        for pid, g in df_norm.groupby("id"):
            groups.append(g)

        self.groups = groups
        self.max_len = max(len(g) for g in groups)

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx: int):
        g = self.groups[idx]

        pid_raw = g["id"].iloc[0]
        pid = self.pid2idx[pid_raw]

        S = self.max_len
        F = len(self.features)

        x = np.zeros((S, F), dtype=np.float32)
        y = np.zeros((S,), dtype=np.float32)
        days = np.zeros((S,), dtype=np.int64)
        mask = np.zeros((S,), dtype=bool)

        arr_x = g[self.features].to_numpy(dtype=np.float32)
        arr_y = g["e2"].to_numpy(dtype=np.float32)
        arr_d = g["measure_day"].to_numpy(dtype=np.int64)

        L = len(g)
        x[:L] = arr_x
        y[:L] = arr_y
        days[:L] = arr_d
        mask[:L] = True

        return {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
            "mask": torch.from_numpy(mask),
            "days": torch.from_numpy(days),
            "pid": torch.tensor(pid, dtype=torch.long),
        }

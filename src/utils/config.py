"""
config.py – rutas e hiperparámetros globales
"""
from __future__ import annotations
from pathlib import Path

# ---------------- Rutas ----------------
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA = DATA_DIR / "raw"
SPLIT_DIR = DATA_DIR / "splits"
MODEL_DIR = ROOT_DIR / "models"
FIG_DIR = ROOT_DIR / "figures"
LOG_DIR = ROOT_DIR / "logs"
GOLD_STD = DATA_DIR / "gold_standard_e2.csv"

for p in [DATA_DIR, RAW_DATA, SPLIT_DIR, MODEL_DIR, FIG_DIR, LOG_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# --------------- Hiperparámetros ---------------
HP = {
    # modelo
    "d_model":      96,
    "nhead":        4,
    "num_layers":   4,
    "d_ff":         512,
    "dropout":      0.12,

    # optimización
    "batch_size":   128,
    "lr":           2e-3,
    "weight_decay": 1e-4,
    "epochs":       50,
    "grad_clip":    1.0,

    # scheduler
    "use_sched":    True,
    "warmup_epochs":5,          # warmup lineal
    "tmax":         45,         # para CosineAnnealingRestarts o CosineAnnealingLR

    # pesos pérdidas
    "ALPHA": 1.0,   # MSE principal
    "BETA":  0.6,   # shape contra gold standard
    "GAMMA": 0.15,  # slope penalty (después del pico)

    # pico aproximado para decay
    "PEAK_LOW": 9,
    "PEAK_HI":  14,
}

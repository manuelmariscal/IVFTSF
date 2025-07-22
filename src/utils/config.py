"""
src/utils/config.py – Paths, HP y gold-standard.
"""
from pathlib import Path

# ── Raíz del proyecto ──────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# ── Datos ──────────────────────────────────────────────────────────────────
DATA_DIR   = ROOT_DIR / "data"
RAW_DATA   = DATA_DIR / "dataset_complete.csv"
SPLIT_DIR  = DATA_DIR / "splits"
GOLD_STD   = DATA_DIR / "gold_standard_e2.csv"      # (28 días)

# ── Salidas ────────────────────────────────────────────────────────────────
MODEL_DIR  = ROOT_DIR / "models"
FIG_DIR    = ROOT_DIR / "figures"
LOG_DIR    = ROOT_DIR / "logs"

# ── Hiper-parámetros por defecto (se sobrescriben con Optuna) ──────────────
HP = dict(
    d_model    = 64,
    nhead      = 4,
    num_layers = 4,
    d_ff       = 512,
    dropout    = 0.12,
    lr         = 2e-3,
    weight_decay = 7.1e-4,
    batch_size = 256,
    epochs     = 50,
)

"""
src/utils/config.py – Centralised paths & hyper‑parameters.
"""
from pathlib import Path

# ── Project root ───────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# ── Data paths ─────────────────────────────────────────────────────────────
DATA_DIR  = ROOT_DIR / "data"
RAW_DATA  = DATA_DIR / "dataset_complete.csv"
SPLIT_DIR = DATA_DIR / "splits"

# ── Outputs ────────────────────────────────────────────────────────────────
MODEL_DIR = ROOT_DIR / "models"
FIG_DIR   = ROOT_DIR / "figures"
LOG_DIR   = ROOT_DIR / "logs"

# ── Gold standard path ─────────────────────────────────────────────────────
GOLD_STD = DATA_DIR / "gold_standard_e2.csv"

# ── Training hyper‑parameters ──────────────────────────────────────────────
HP = dict(
    d_model      = 64,
    nhead        = 4,
    num_layers   = 4,
    d_ff         = 512,
    dropout      = 0.12,
    lr           = 2e-3,
    weight_decay = 7.2e-4,
    batch_size   = 256,
    epochs       = 50,
    lambda_gs    = 0.1,   # regularización gold standard
    patience     = 10,    # early stopping
)

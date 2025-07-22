"""
train.py – Entrena el Transformer en log-space con:
  * Positional encoding sin/cos (ciclo de 28 días)
  * Gold standard (media fisiológica) como regularización
  * Limpieza de NaNs / valores negativos
  * Early stopping
  * MAPE como métrica
"""

import math, os
from pathlib import Path
import pandas as pd, numpy as np, torch
from torch.utils.data import DataLoader

from .dataset import HormoneDataset, CSV_FEATURES
from .model import HormoneTransformer
from .utils import (
    HP, SPLIT_DIR, MODEL_DIR, GOLD_STD,
    auto_mkdir, setup_logger, get_device, set_cpu_threads
)

logger = setup_logger("train")
DEVICE  = get_device()          # cpu
THREADS = set_cpu_threads()     # fija hilos para consistencia

# Hiperparámetros extra
EPS = 1e-6
LAMBDA_GS = 0.1          # peso de la regularización gold-standard
EARLY_PATIENCE = 8       # épocas sin mejora antes de parar

# ------------------------------------------------------------------ #
def add_positional(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega columnas sin/cos de la fase (periodicidad 28 días)."""
    df = df.copy()
    ang = 2 * math.pi * (df["measure_day"] - 1) / 28.0
    df["pos_sin"] = np.sin(ang)
    df["pos_cos"] = np.cos(ang)
    return df

def build_gold_standard(train_df: pd.DataFrame) -> torch.Tensor:
    """
    Construye (o carga) el vector de 28 valores medios de E2 (log-space)
    usado para regularización. Se guarda en CSV para reutilizar.
    """
    if GOLD_STD.exists():
        gs = pd.read_csv(GOLD_STD).set_index("measure_day")["e2"]
        if not gs.isna().any():
            return torch.log1p(torch.tensor(gs.values, dtype=torch.float32, device=DEVICE))

    tmp = train_df[train_df["measure_day"].between(1, 28)]
    gs = (
        tmp.groupby("measure_day")["e2"].mean()
           .reindex(range(1, 29))
           .bfill()
           .ffill()
    )
    GOLD_STD.parent.mkdir(parents=True, exist_ok=True)
    gs.to_frame().reset_index().to_csv(GOLD_STD, index=False)
    logger.info("Gold standard saved -> %s", GOLD_STD)
    return torch.log1p(torch.tensor(gs.values, dtype=torch.float32, device=DEVICE))

@auto_mkdir(1)
def save_ckpt(state, path: Path):
    torch.save(state, path, pickle_protocol=4)

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina filas con e2 nulo/no-numérico o negativo."""
    df = df.copy()
    df["e2"] = pd.to_numeric(df["e2"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["e2"])
    df = df[df["e2"] >= 0]
    removed = before - len(df)
    if removed > 0:
        logger.info("Filas descartadas por NaN/negativas en e2: %d", removed)
    return df

# ------------------------------------------------------------------ #
def run_epoch(dl, model, opt=None, gold_log: torch.Tensor | None = None):
    """Ejecuta una época y devuelve MAPE (%)."""
    train = opt is not None
    model.train(train)
    tot, n = 0.0, 0
    for b in dl:
        m = b["mask"].to(DEVICE)
        if m.sum() == 0:
            continue
        x = b["x"].to(DEVICE)
        y_raw = b["y"].to(DEVICE)
        pid = b["pid"].to(DEVICE)
        days = b["days"].to(DEVICE)

        # Evitar NaNs
        valid = m & ~torch.isnan(y_raw)
        if valid.sum() == 0:
            continue

        y_log = torch.log1p(y_raw)
        ŷ_log = model(x, pid, src_key_padding_mask=~m)

        base = torch.nn.functional.mse_loss(ŷ_log[valid], y_log[valid])

        if gold_log is not None and LAMBDA_GS > 0:
            gs_vals = gold_log[days.long().clamp(1, 28) - 1]
            reg = torch.nn.functional.mse_loss(ŷ_log[valid], gs_vals[valid])
            loss = base + LAMBDA_GS * reg
        else:
            loss = base

        if torch.isnan(loss):
            continue

        if train:
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        ŷ_raw = torch.expm1(ŷ_log).clamp_min(0)
        err = torch.abs((ŷ_raw - y_raw) / (y_raw.abs() + EPS))
        tot += err[valid].sum().item()
        n += valid.sum().item()
    return (tot / max(n, 1)) * 100.0

# ------------------------------------------------------------------ #
def main():
    # Cargar y limpiar splits
    tr_df = clean_df(pd.read_csv(SPLIT_DIR / "train.csv"))
    va_df = clean_df(pd.read_csv(SPLIT_DIR / "val.csv"))
    te_df = clean_df(pd.read_csv(SPLIT_DIR / "test.csv"))

    # Añadir encoding posicional
    tr_df = add_positional(tr_df)
    va_df = add_positional(va_df)
    te_df = add_positional(te_df)

    # Gold standard (log-space)
    gold_log = build_gold_standard(tr_df)

    # Datasets y dataloaders
    tr_ds, va_ds, te_ds = map(HormoneDataset, [tr_df, va_df, te_df])
    opts = dict(num_workers=max(1, os.cpu_count() - 1), pin_memory=False)
    tr_ld = DataLoader(tr_ds, batch_size=HP["batch_size"], shuffle=True, **opts)
    va_ld = DataLoader(va_ds, batch_size=HP["batch_size"], **opts)
    te_ld = DataLoader(te_ds, batch_size=HP["batch_size"], **opts)

    # Modelo
    num_feats = len(CSV_FEATURES) + 2  # pos_sin + pos_cos
    model = HormoneTransformer(
        num_features=num_feats,
        num_patients=tr_ds.num_patients,
        d_model=HP["d_model"], nhead=HP["nhead"],
        num_layers=HP["num_layers"], d_ff=HP["d_ff"],
        dropout=HP["dropout"],
    ).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=HP["lr"])

    best = float("inf")
    no_improve = 0
    for ep in range(1, HP["epochs"] + 1):
        tr_mape = run_epoch(tr_ld, model, optim, gold_log=gold_log)
        va_mape = run_epoch(va_ld, model, gold_log=gold_log)
        logger.info(
            "Epoch %2d/%d | train_MAPE=%.2f%% | val_MAPE=%.2f%%",
            ep, HP["epochs"], tr_mape, va_mape
        )

        if va_mape + 1e-6 < best:
            best = va_mape
            no_improve = 0
            save_ckpt(
                {
                    "model_state": model.state_dict(),
                    "mean": tr_ds.means.values,
                    "std": tr_ds.stds.values,
                },
                MODEL_DIR / "model.pt",
            )
        else:
            no_improve += 1

        if no_improve >= EARLY_PATIENCE:
            logger.info("Early stopping en epoch %d", ep)
            break

    # Evaluación test
    ckpt = torch.load(MODEL_DIR / "model.pt", map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    test_mape = run_epoch(te_ld, model, gold_log=gold_log)
    logger.info("🧪  Test MAPE: %.2f%% (best val=%.2f%%)", test_mape, best)

if __name__ == "__main__":
    main()

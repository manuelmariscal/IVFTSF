"""
src/train.py – Entrena el Transformer con:
  • pérdida MSE en log(E2)
  • shape_loss contra gold standard fisiológico
  • slope_penalty para forzar caída después del pico
  • early stopping
"""

from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

from .dataset import HormoneDataset, CSV_FEATURES
from .model   import HormoneTransformer
from .utils   import (
    HP, SPLIT_DIR, MODEL_DIR, GOLD_STD,
    auto_mkdir, setup_logger
)

logger = setup_logger("train")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPS    = 1e-6

# ---------------- Gold standard ------------------
def create_gold_standard() -> pd.Series:
    if GOLD_STD.exists():
        return pd.read_csv(GOLD_STD, index_col="measure_day")["e2"]
    tr_df = pd.read_csv(SPLIT_DIR / "train.csv")
    gs = tr_df.groupby("measure_day")["e2"].mean()
    gs = gs.reindex(range(1, 29)).interpolate("linear").bfill().ffill()
    gs.to_csv(GOLD_STD, header=True)
    logger.info("Gold standard saved -> %s", GOLD_STD)
    return gs

GOLD      = create_gold_standard()
GOLD_LOG  = torch.log1p(torch.tensor(GOLD.values, dtype=torch.float32, device=DEVICE))

# --------- helpers to save ----------
@auto_mkdir(1)
def save_ckpt(state, path: Path): torch.save(state, path)

# --------------- losses ----------------
def shape_loss(pred_log: torch.Tensor, days: torch.Tensor) -> torch.Tensor:
    """
    L2 entre predicción y gold standard en días 1..28
    pred_log: (B,S)
    days    : (B,S)
    """
    mask = (days >= 1) & (days <= 28)
    if not mask.any():  # puede pasar en forecast largo pero nunca en train
        return torch.tensor(0., device=pred_log.device)
    idx  = (days[mask] - 1).long()
    gs   = GOLD_LOG[idx]
    return ((pred_log[mask] - gs) ** 2).mean()

def slope_penalty(pred_log: torch.Tensor,
                  days: torch.Tensor,
                  peak_lo: int = 10,
                  peak_hi: int = 14) -> torch.Tensor:
    """
    Penaliza subidas DESPUÉS del pico fisiológico (~14).
    pred_log : (B,S)
    days     : (B,S)
    """
    # diferencia hacia adelante
    dy = pred_log[:, 1:] - pred_log[:, :-1]   # (B,S-1)
    d  = days[:, :-1]

    after_mask = d >= peak_hi
    if after_mask.any():
        pos_after = torch.relu(dy)            # solo subidas
        loss_after = (pos_after[after_mask] ** 2).mean()
    else:
        loss_after = torch.tensor(0., device=pred_log.device)

    return loss_after

# --------------- dataloaders ---------------
def make_loader(ds: HormoneDataset, bs: int, shuffle: bool):
    # Peso por paciente: duplica los que tienen más días altos (fase lútea)
    weights = np.array(
        [2.0 if g["measure_day"].mean() >= 15 else 1.0 for g in ds.groups],
        dtype=np.float32
    )
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True) if shuffle else None
    return DataLoader(
        ds, batch_size=bs,
        sampler=sampler,
        shuffle=shuffle and sampler is None,
        num_workers=max(1, os.cpu_count() - 1),
        pin_memory=torch.cuda.is_available()
    )

# --------------- epoch loop ----------------
def run_epoch(dl,
              model,
              opt=None,
              alpha: float = 0.5,
              beta:  float = 0.2,
              peak_lo: int = 10,
              peak_hi: int = 14):
    train = opt is not None
    model.train(train)

    tot_err, n_pts = 0.0, 0
    mse_acc = sh_acc = sp_acc = 0.0
    nb = 0

    for b in dl:
        m     = b["mask"].to(DEVICE)
        if m.sum() == 0:  # lote vacío
            continue

        x     = b["x"].to(DEVICE)
        y_raw = b["y"].to(DEVICE)
        pid   = b["pid"].to(DEVICE)
        days  = b["days"].to(DEVICE)
        phase = b["phase"].to(DEVICE)

        y_log = torch.log1p(y_raw.clamp_min(0))
        pred_log = model(x, pid, days=days, phase=phase, src_key_padding_mask=~m)

        mse = torch.nn.functional.mse_loss(pred_log[m], y_log[m])
        sh  = shape_loss(pred_log, days)
        sp  = slope_penalty(pred_log, days, peak_lo, peak_hi)

        loss = mse + alpha * sh + beta * sp

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()

        pred_raw = torch.expm1(pred_log).clamp_min(0)
        err = torch.abs((pred_raw - y_raw) / (y_raw.abs() + EPS))

        tot_err += err[m].sum().item()
        n_pts   += m.sum().item()
        mse_acc += mse.item()
        sh_acc  += sh.item()
        sp_acc  += sp.item()
        nb      += 1

    mape = (tot_err / max(n_pts, 1)) * 100.0
    if nb == 0: nb = 1
    return mape, mse_acc/nb, sh_acc/nb, sp_acc/nb

# --------------- main ----------------------
def main():
    # Parámetros adicionales para las pérdidas
    ALPHA = 0.5  # peso shape_loss
    BETA  = 0.2  # peso slope_penalty
    PEAK_LO, PEAK_HI = 10, 14

    tr_df = pd.read_csv(SPLIT_DIR/"train.csv")
    va_df = pd.read_csv(SPLIT_DIR/"val.csv")
    te_df = pd.read_csv(SPLIT_DIR/"test.csv")

    tr_ds, va_ds, te_ds = map(HormoneDataset, [tr_df, va_df, te_df])

    tr_ld = make_loader(tr_ds, HP["batch_size"], shuffle=True)
    va_ld = DataLoader(va_ds, batch_size=HP["batch_size"], num_workers=2)
    te_ld = DataLoader(te_ds, batch_size=HP["batch_size"], num_workers=2)

    model = HormoneTransformer(
        num_features=len(CSV_FEATURES) + 2,  # + sin/cos
        num_patients=tr_ds.num_patients,
        d_model=HP["d_model"], nhead=HP["nhead"],
        num_layers=HP["num_layers"], d_ff=HP["d_ff"],
        dropout=HP["dropout"]
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"])

    best_val, patience, P = float("inf"), 0, 12  # patience
    for ep in range(1, HP["epochs"] + 1):
        tr_mape, tr_mse, tr_sh, tr_sp = run_epoch(tr_ld, model, opt,
                                                  ALPHA, BETA, PEAK_LO, PEAK_HI)
        va_mape, va_mse, va_sh, va_sp = run_epoch(va_ld, model, None,
                                                  ALPHA, BETA, PEAK_LO, PEAK_HI)

        if ep % 5 == 1 or ep == HP["epochs"]:
            logger.info(
                "Epoch %2d/%d | train_MAPE=%.2f%% (mse %.4f sh %.4f sp %.4f) | "
                "val_MAPE=%.2f%% (mse %.4f sh %.4f sp %.4f)",
                ep, HP["epochs"],
                tr_mape, tr_mse, tr_sh, tr_sp,
                va_mape, va_mse, va_sh, va_sp
            )
        else:
            logger.info("Epoch %2d/%d | train_MAPE=%.2f%% | val_MAPE=%.2f%%",
                        ep, HP["epochs"], tr_mape, va_mape)

        # early stop
        if va_mape < best_val - 1e-3:
            best_val, patience = va_mape, 0
            save_ckpt(
                {
                    "model_state": model.state_dict(),
                    "mean": tr_ds.means.values,
                    "std":  tr_ds.stds.values
                },
                MODEL_DIR/"model.pt"
            )
        else:
            patience += 1
            if patience >= P:
                logger.info("⏹️  Early‑stop (no mejora %d epochs)", P)
                break

    # Test
    ckpt = torch.load(MODEL_DIR/"model.pt", map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    te_mape, te_mse, te_sh, te_sp = run_epoch(te_ld, model, None, ALPHA, BETA, PEAK_LO, PEAK_HI)
    logger.info("🧪  Test MAPE: %.2f%% (mse %.4f sh %.4f sp %.4f)",
                te_mape, te_mse, te_sh, te_sp)


if __name__ == "__main__":
    main()

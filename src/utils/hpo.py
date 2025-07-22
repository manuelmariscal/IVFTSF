"""
hpo.py – Hyper‑parameter optimisation con MAPE (%), positional encoding y
         regularización gold‑standard.
"""

from __future__ import annotations
import argparse, json, math, os
from pathlib import Path
from typing import Dict, Any

import numpy as np, optuna, pandas as pd, torch
from torch.utils.data import DataLoader

from . import (
    SPLIT_DIR, MODEL_DIR, GOLD_STD,
    auto_mkdir, setup_logger, get_device, set_cpu_threads
)
from ..dataset import HormoneDataset, CSV_FEATURES
from ..model import HormoneTransformer

log = setup_logger("hpo")
DEVICE  = get_device()          # cpu
THREADS = set_cpu_threads()

HPO_DIR = MODEL_DIR / "hpo"; HPO_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = HPO_DIR / "hpo_study.db"

EPS = 1e-6
LAMBDA_GS = 0.1          # peso regularización gold‑standard

# ───────────────────────────────── helpers ───────────────────────────────── #
def add_positional(df: pd.DataFrame) -> pd.DataFrame:
    ang = 2 * math.pi * (df["measure_day"] - 1) / 28.0
    df = df.copy()
    df["pos_sin"] = np.sin(ang)
    df["pos_cos"] = np.cos(ang)
    return df

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["e2"] = pd.to_numeric(df["e2"], errors="coerce")
    df = df.dropna(subset=["e2"])
    df = df[df["e2"] >= 0]
    return df

def build_gold_standard(train_df: pd.DataFrame) -> torch.Tensor:
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
    log.info("Gold standard saved -> %s", GOLD_STD)
    return torch.log1p(torch.tensor(gs.values, dtype=torch.float32, device=DEVICE))

def loaders(bs: int):
    tr_df = add_positional(clean_df(pd.read_csv(SPLIT_DIR / "train.csv")))
    va_df = add_positional(clean_df(pd.read_csv(SPLIT_DIR / "val.csv")))
    tr_ds = HormoneDataset(tr_df)
    va_ds = HormoneDataset(va_df)
    opts = dict(num_workers=max(1, os.cpu_count()-1), pin_memory=False)
    return (
        tr_ds,
        DataLoader(tr_ds, batch_size=bs, shuffle=True, **opts),
        DataLoader(va_ds, batch_size=bs, **opts),
        tr_df,
    )

def mape_epoch(dl, model, gold_log: torch.Tensor, opt=None):
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

        valid = m & ~torch.isnan(y_raw)
        if valid.sum() == 0:
            continue

        y_log = torch.log1p(y_raw)
        y_hat = model(x, pid, src_key_padding_mask=~m)

        base = torch.nn.functional.mse_loss(y_hat[valid], y_log[valid])
        gs_vals = gold_log[days.long().clamp(1, 28) - 1]
        reg = torch.nn.functional.mse_loss(y_hat[valid], gs_vals[valid])
        loss = base + LAMBDA_GS * reg

        if torch.isnan(loss):
            continue

        if train:
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        y_hat_raw = torch.expm1(y_hat).clamp_min(0)
        err = torch.abs((y_hat_raw - y_raw) / (y_raw.abs() + EPS))
        tot += err[valid].sum().item()
        n += valid.sum().item()
    return (tot / max(n, 1)) * 100.0

# ─────────────────────────── Optuna objective ──────────────────────────── #
def objective(trial: optuna.Trial) -> float:
    p: Dict[str, Any] = {
        "d_model":      trial.suggest_int("d_model", 64, 128, step=16),
        "nhead":        trial.suggest_categorical("nhead", [4, 8]),
        "num_layers":   trial.suggest_int("num_layers", 3, 6),
        "d_ff":         trial.suggest_int("d_ff", 256, 768, step=128),
        "dropout":      trial.suggest_float("dropout", 0.05, 0.25),
        "lr":           trial.suggest_float("lr", 5e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 1e-3),
        "batch_size":   trial.suggest_categorical("batch_size", [64, 128, 256]),
        "epochs":       trial.suggest_int("epochs", 10, 25),
        "use_sched":    trial.suggest_categorical("use_sched", [False, True]),
    }

    tr_ds, tr_ld, va_ld, tr_df = loaders(p["batch_size"])
    gold_log = build_gold_standard(tr_df)

    model = HormoneTransformer(
        num_features=len(CSV_FEATURES)+2,
        num_patients=tr_ds.num_patients,
        d_model=p["d_model"], nhead=p["nhead"],
        num_layers=p["num_layers"], d_ff=p["d_ff"],
        dropout=p["dropout"],
    ).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt,  T_max=p["epochs"]) if p["use_sched"] else None)
    best = np.inf
    for ep in range(1, p["epochs"]+1):
        mape_epoch(tr_ld, model, gold_log, opt)
        if sched: sched.step()
        val = mape_epoch(va_ld, model, gold_log)
        best = min(best, val)
        log.info("Trial %3d | Epoch %2d/%d | val_MAPE=%.2f%% | best=%.2f%%",
                 trial.number, ep, p["epochs"], val, best)
        trial.report(val, ep)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return best

# ───────────────────────────────── main ────────────────────────────────── #
@auto_mkdir(1)
def save_json(obj, path): path.write_text(json.dumps(obj, indent=2))

def main(trials: int, jobs: int, fresh: bool):
    if fresh and DB_PATH.exists():
        DB_PATH.unlink()
        log.info("🔄  Previous study removed – starting fresh")

    study = optuna.create_study(
        study_name="E2‑MAPE‑HPO‑log",
        direction="minimize",
        storage=f"sqlite:///{DB_PATH}",
        load_if_exists=not fresh,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
    )
    log.info("Optuna: %d trials | jobs=%d", trials, jobs)
    study.optimize(objective, n_trials=trials, n_jobs=jobs)

    log.info("🏆  Best MAPE %.2f%%  params=%s", study.best_value, study.best_params)
    save_json(study.best_params, HPO_DIR / "best_params.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--jobs",   type=int, default=os.cpu_count())
    ap.add_argument("--fresh",  type=int, choices=[0,1], default=1)
    main(**vars(ap.parse_args()))

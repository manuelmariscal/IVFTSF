"""
train.py – Entrenamiento con pérdidas: MSE + shape + slope (derivada negativa),
scheduler, grad clip y early stopping.
"""
from __future__ import annotations
import os, math, torch, numpy as np, pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.nn.utils import clip_grad_norm_

from .dataset import HormoneDataset, CSV_FEATURES
from .model   import HormoneTransformer
from .utils   import HP, SPLIT_DIR, MODEL_DIR, GOLD_STD, auto_mkdir, setup_logger

logger = setup_logger("train")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPS    = 1e-6

# ---------------- gold standard -----------------
def create_gold_standard() -> pd.Series:
    if GOLD_STD.exists():
        return pd.read_csv(GOLD_STD, index_col="measure_day")["e2"]
    df = pd.read_csv(SPLIT_DIR / "train.csv")
    gs = df.groupby("measure_day")["e2"].mean().reindex(range(1,29))
    gs = gs.interpolate("linear").bfill().ffill()
    gs.to_csv(GOLD_STD, header=True)
    logger.info("Gold standard saved -> %s", GOLD_STD)
    return gs

GOLD = create_gold_standard()
gold_log = torch.log1p(torch.tensor(GOLD.values, dtype=torch.float32, device=DEVICE))

# ---------------- helpers -----------------------
@auto_mkdir(1)
def save_ckpt(state, path: Path): torch.save(state, path)

def shape_loss(y_log, days):
    idx = (days >= 1) & (days <= 28)
    if idx.sum() == 0: return torch.tensor(0., device=y_log.device)
    gs = gold_log[(days[idx] - 1).long()]
    return torch.mean((y_log[idx] - gs) ** 2)

def slope_penalty(y_log, days):
    """
    Penaliza pendiente positiva a partir de PEAK_HI (debería bajar).
    """
    peak_hi = HP["PEAK_HI"]
    # derivada aprox
    dy = y_log[:,1:] - y_log[:,:-1]
    dday = days[:,1:].float() - days[:,:-1].float()
    slope = dy / (dday + 1e-6)

    post = days[:,1:] > peak_hi
    bad  = slope[post]  # donde debe ser <=0

    if bad.numel() == 0: 
        return torch.tensor(0., device=y_log.device)
    return torch.relu(bad).mean()

def make_loader(ds: HormoneDataset, bs: int, shuffle: bool):
    weights = np.array(
        [2.0 if g["measure_day"].mean() >= 15 else 1.0 for g in ds.groups],
        dtype=np.float32
    )
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True) if shuffle else None
    return DataLoader(
        ds, batch_size=bs,
        sampler=sampler,
        shuffle=shuffle and sampler is None,
        num_workers=max(1, os.cpu_count()-1),
        pin_memory=torch.cuda.is_available(),
    )

def mape(y_pred, y_true, mask):
    err = torch.abs((y_pred - y_true) / (y_true.abs() + EPS))
    return (err[mask].mean() * 100).item()

def run_epoch(dl, model, opt=None, sched=None):
    train = opt is not None
    model.train(train)
    tot_mape = 0.0; n_batches = 0
    mse_tot = sh_tot = sp_tot = 0.0

    for b in dl:
        m     = b["mask"].to(DEVICE)
        if m.sum() == 0: continue
        x     = b["x"].to(DEVICE)
        y_raw = b["y"].to(DEVICE)
        pid   = b["pid"].to(DEVICE)
        days  = b["days"].to(DEVICE)
        phase = b["phase"].to(DEVICE)

        y_log   = torch.log1p(y_raw.clamp_min(0))
        y_hat_l = model(x, pid, days=days, phase=phase, src_key_padding_mask=~m)

        mse = torch.nn.functional.mse_loss(y_hat_l[m], y_log[m])
        sh  = shape_loss(y_hat_l, days)
        sp  = slope_penalty(y_hat_l.unsqueeze(0), days.unsqueeze(0))  # adjust dims

        loss = HP["ALPHA"]*mse + HP["BETA"]*sh + HP["GAMMA"]*sp

        if train:
            opt.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), HP["grad_clip"])
            opt.step()
            if sched is not None:
                sched.step()

        y_hat = torch.expm1(y_hat_l).clamp_min(0)
        tot_mape += mape(y_hat, y_raw, m)
        n_batches += 1
        mse_tot += mse.item(); sh_tot += sh.item(); sp_tot += sp.item()

    return tot_mape / max(n_batches,1), mse_tot / max(n_batches,1), sh_tot / max(n_batches,1), sp_tot / max(n_batches,1)

# ---------------- main --------------------------
def main():
    tr_df = pd.read_csv(SPLIT_DIR/"train.csv")
    va_df = pd.read_csv(SPLIT_DIR/"val.csv")
    te_df = pd.read_csv(SPLIT_DIR/"test.csv")

    tr_ds, va_ds, te_ds = map(HormoneDataset, [tr_df, va_df, te_df])
    tr_ld = make_loader(tr_ds, HP["batch_size"], shuffle=True)
    va_ld = DataLoader(va_ds, batch_size=HP["batch_size"], num_workers=2)
    te_ld = DataLoader(te_ds, batch_size=HP["batch_size"], num_workers=2)

    model = HormoneTransformer(
        num_features=len(CSV_FEATURES)+2,
        num_patients=tr_ds.num_patients,
        d_model=HP["d_model"],
        nhead=HP["nhead"],
        num_layers=HP["num_layers"],
        d_ff=HP["d_ff"],
        dropout=HP["dropout"]
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"])

    if HP["use_sched"]:
        total_steps = HP["epochs"] * len(tr_ld)
        warmup = HP["warmup_epochs"] * len(tr_ld)
        def lr_lambda(step):
            if step < warmup:
                return float(step) / float(max(1, warmup))
            progress = (step - warmup) / float(max(1, total_steps - warmup))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    else:
        sched = None

    best_val = float("inf")
    patience = 0
    PATIENCE_MAX = 25

    for ep in range(1, HP["epochs"]+1):
        tr_mape, tr_mse, tr_sh, tr_sp = run_epoch(tr_ld, model, opt, sched)
        va_mape, va_mse, va_sh, va_sp = run_epoch(va_ld, model, None, None)

        logger.info(
            "Epoch %2d/%d | train_MAPE=%.2f%% (mse %.4f sh %.4f sp %.4f) | "
            "val_MAPE=%.2f%% (mse %.4f sh %.4f sp %.4f)",
            ep, HP["epochs"], tr_mape, tr_mse, tr_sh, tr_sp,
            va_mape, va_mse, va_sh, va_sp
        )

        if va_mape < best_val - 1e-3:
            best_val, patience = va_mape, 0
            save_ckpt({
                "model_state": model.state_dict(),
                "mean": tr_ds.means.values,
                "std":  tr_ds.stds.values
            }, MODEL_DIR/"model.pt")
        else:
            patience += 1
            if patience >= PATIENCE_MAX:
                logger.info("⏹️  Early‑stop (no mejora %d epochs)", PATIENCE_MAX)
                break

    # -------- Test --------
    ckpt = torch.load(MODEL_DIR/"model.pt", map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    te_mape, te_mse, te_sh, te_sp = run_epoch(te_ld, model, None, None)
    logger.info("🧪  Test MAPE: %.2f%% (mse %.4f sh %.4f sp %.4f)", te_mape, te_mse, te_sh, te_sp)

if __name__ == "__main__":
    main()

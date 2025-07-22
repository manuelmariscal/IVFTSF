"""
train.py – Entrena con shape-loss (gold-standard) + early stopping.
"""
from __future__ import annotations
import os, math, random, pandas as pd, torch, numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from pathlib import Path
from .dataset import HormoneDataset, CSV_FEATURES
from .model import HormoneTransformer
from .utils import HP, SPLIT_DIR, MODEL_DIR, GOLD_STD, auto_mkdir, setup_logger
logger = setup_logger("train")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPS = 1e-6

# ---------- gold-standard (28 días) ---------------------------------------
def create_gold_standard():
    if GOLD_STD.exists():
        gs = pd.read_csv(GOLD_STD, index_col="measure_day")["e2"]
        return gs
    tr_df = pd.read_csv(SPLIT_DIR/"train.csv")
    gs = tr_df.groupby("measure_day")["e2"].mean()        # 1-28
    gs = gs.reindex(range(1,29)).interpolate("linear").bfill().ffill()
    gs.to_csv(GOLD_STD, header=True)
    logger.info("Gold standard saved -> %s", GOLD_STD)
    return gs

GOLD = create_gold_standard()
gold_log = torch.log1p(torch.tensor(GOLD.values, dtype=torch.float32, device=DEVICE))

# ---------- util ----------------------------------------------------------
@auto_mkdir(1)
def save_ckpt(state, path:Path): torch.save(state, path)

def shape_loss(ŷ_log, days):
    """L2 contra la gold-standard en los días reales (1-28)."""
    idx = (days>=1) & (days<=28)
    gs = gold_log[(days[idx]-1).long()]          # alin
    return ((ŷ_log[idx] - gs) ** 2).mean()

# ---------- DataLoaders equilibrados --------------------------------------
def make_loader(ds: HormoneDataset, bs: int, shuffle: bool):
    """
    Un peso por PACIENTE: >1 si la media del día de ciclo ≥15 (luteal),
    1.0 en caso contrario. (len(weights) == len(ds))
    """
    w = np.array(
        [2.0 if g["measure_day"].mean() >= 15 else 1.0
         for g in ds.groups],
        dtype=np.float32,
    )
    sampler = (
        WeightedRandomSampler(w, num_samples=len(w), replacement=True)
        if shuffle else None
    )
    return DataLoader(
        ds, batch_size=bs,
        sampler=sampler,
        shuffle=shuffle and sampler is None,
        num_workers=max(1, os.cpu_count() - 1),
        pin_memory=torch.cuda.is_available(),
    )

# ---------- epoch ---------------------------------------------------------
def run_epoch(dl, model, opt=None):
    train = opt is not None
    model.train(train)
    tot, n = 0.0, 0
    for b in dl:
        m = b["mask"].to(DEVICE); real = m.sum()
        if real==0: continue
        x  = b["x"].to(DEVICE)
        y  = b["y"].to(DEVICE)
        pid= b["pid"].to(DEVICE)
        ph = b["phase"].to(DEVICE)
        dy = b["days"].to(DEVICE)

        ŷ_log = model(x, pid, ph, src_key_padding_mask=~m)
        loss_mape  = (torch.abs((torch.expm1(ŷ_log)-y)/(y.abs()+EPS))[m]).mean()
        loss_shape = shape_loss(ŷ_log, dy)[None]          # broadcast
        loss = loss_mape + 0.1*loss_shape

        if train: opt.zero_grad(); loss.backward(); opt.step()

        tot += loss_mape.item()*real.item()
        n   += real.item()
    return (tot/max(n,1))*100.0         # MAPE %

# ---------- main ----------------------------------------------------------
def main():
    tr_df = pd.read_csv(SPLIT_DIR/"train.csv")
    va_df = pd.read_csv(SPLIT_DIR/"val.csv")
    te_df = pd.read_csv(SPLIT_DIR/"test.csv")

    tr_ds, va_ds, te_ds = map(HormoneDataset, [tr_df,va_df,te_df])

    tr_ld = make_loader(tr_ds, HP["batch_size"], shuffle=True)
    va_ld = DataLoader(va_ds, batch_size=HP["batch_size"], num_workers=2)
    te_ld = DataLoader(te_ds, batch_size=HP["batch_size"], num_workers=2)

    model = HormoneTransformer(
        num_features=len(CSV_FEATURES)+2,
        num_patients=tr_ds.num_patients,
        d_model=HP["d_model"], nhead=HP["nhead"],
        num_layers=HP["num_layers"], d_ff=HP["d_ff"],
        dropout=HP["dropout"]).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=HP["lr"],
                            weight_decay=HP["weight_decay"])

    best, patience, P = float("inf"), 0, 6
    for ep in range(1, HP["epochs"]+1):
        tr = run_epoch(tr_ld, model, opt)
        va = run_epoch(va_ld, model)
        logger.info("Epoch %2d/%d | train_MAPE=%.2f%% | val_MAPE=%.2f%%",
                    ep, HP["epochs"], tr, va)

        if va < best-1e-3:
            best, patience = va, 0
            save_ckpt({"model_state":model.state_dict(),
                       "mean":tr_ds.means.values,
                       "std": tr_ds.stds.values}, MODEL_DIR/"model.pt")
        else:
            patience +=1
            if patience>=P:
                logger.info("⏹️  Early-stop (no mejora %d epochs)", P)
                break

    # ---------- test ----------
    ckpt = torch.load(MODEL_DIR/"model.pt", map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt)  # weights_only True → dict plano
    test = run_epoch(te_ld, model)
    logger.info("🧪  Test MAPE: %.2f%%", test)

if __name__ == "__main__":
    main()

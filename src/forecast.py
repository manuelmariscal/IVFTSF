"""
forecast.py – Genera la curva 1‑33 con MC‑Dropout.
"""

from __future__ import annotations
import json, math, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from .model   import HormoneTransformer
from .dataset import CSV_FEATURES
from .utils   import MODEL_DIR, SPLIT_DIR, setup_logger

logger  = setup_logger("forecast")
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"


# ───────────────────── helpers ──────────────────────
def load_ckpt():
    return torch.load(MODEL_DIR / "model.pt",
                      map_location=DEVICE,
                      weights_only=False)


def phase_from_day(days: np.ndarray) -> np.ndarray:
    """0 folicular, 1 ovulatoria, 2 lútea (días 29‑33 ≡ fase 0)."""
    phase = np.zeros_like(days)
    phase[(days >= 10) & (days <= 14)] = 1
    phase[(days >= 15) & (days <= 28)] = 2
    return phase


@torch.no_grad()
def mc_dropout_predict(model, x, pid, days, phase, mask, mc=100):
    model.train(True)                                      # dropout ON
    preds = []
    for _ in range(mc):
        y_log = model(x, pid, days=days, phase=phase,
                       src_key_padding_mask=~mask)
        preds.append(torch.expm1(y_log).clamp_min(0).cpu())
    arr = torch.stack(preds)                               # (mc,1,S)
    return arr.mean(0).squeeze(0).numpy(), \
           arr.std (0).squeeze(0).numpy()


def create_sequence(js, means, stds):
    S = 33
    full_days = np.arange(1, S + 1)

    obs = pd.DataFrame(js["observations"]).copy()
    ang = 2 * math.pi * (obs["measure_day"] - 1) / 28.0
    obs["pos_sin"], obs["pos_cos"] = np.sin(ang), np.cos(ang)

    feat = CSV_FEATURES + ["pos_sin", "pos_cos"]
    for c in feat:
        obs.setdefault(c, 0.0)

    obs = (obs.set_index("measure_day")
               .reindex(full_days)
               .interpolate("linear").bfill().ffill())

    if 1 not in obs.index:
        logger.warning("No hay observación en día 1; interpolando.")
    if 28 not in obs.index:
        logger.warning("No hay observación en día 28; interpolando.")

    obs[feat] = (obs[feat] - means[feat]) / stds[feat]

    x     = torch.tensor(obs[feat].values, dtype=torch.float32).unsqueeze(0)
    days  = torch.tensor(full_days,  dtype=torch.int64 ).unsqueeze(0)
    phase = torch.tensor(phase_from_day(full_days),
                         dtype=torch.int64 ).unsqueeze(0)
    mask  = torch.ones(1, S, dtype=torch.bool)
    return x, days, phase, mask


# ───────────────────── main ─────────────────────────
def forecast_from_json(json_path):
    js     = json.loads(Path(json_path).read_text())
    ckpt   = load_ckpt()

    cols   = [*CSV_FEATURES, "pos_sin", "pos_cos"]
    means  = pd.Series(ckpt["mean"], index=cols)
    stds   = pd.Series(ckpt["std"],  index=cols)

    tr_df  = pd.read_csv(SPLIT_DIR / "train.csv")
    pid2ix = {pid: i for i, pid in enumerate(tr_df["id"].unique())}
    n_pid  = len(pid2ix)

    state   = ckpt["model_state"]
    d_model = state["num_proj.weight"].shape[0]
    model   = HormoneTransformer(len(CSV_FEATURES)+2, n_pid,
                                 d_model=d_model, nhead=4,
                                 num_layers=4, d_ff=512, dropout=0.12).to(DEVICE)
    model.load_state_dict(state, strict=False)

    # embedding de paciente
    pid_raw = js["patient_id"]
    if pid_raw in pid2ix:
        pid_ix = pid2ix[pid_raw]
    else:
        logger.warning("Paciente %s no visto en entrenamiento – embedding medio.",
                       pid_raw)
        with torch.no_grad():
            mean_emb = model.pid_emb.weight.mean(0, keepdim=True)
            model.pid_emb = torch.nn.Embedding.from_pretrained(
                mean_emb.repeat(n_pid, 1), freeze=False)
        pid_ix = 0
    pid = torch.tensor([pid_ix], device=DEVICE)

    x, days, phase, mask = create_sequence(js, means, stds)
    x, days, phase, mask = [t.to(DEVICE) for t in (x, days, phase, mask)]

    μ, σ = mc_dropout_predict(model, x, pid, days, phase, mask,
                              mc=js.get("mc_samples", 100))

    out_png = Path(js.get("output", f"figures/forecast_{pid_raw}.png"))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "day": np.arange(1, 34),
        "prediction": μ,
        "lower": np.clip(μ - 2*σ, 0, None),
        "upper": μ + 2*σ,
    }).to_csv(out_png.with_suffix(".csv"), index=False)

    plt.figure(figsize=(10, 4))
    plt.plot(range(1, 34), μ, label="predicción")
    plt.fill_between(range(1, 34), μ-2*σ, μ+2*σ,
                     alpha=.25, label="±2σ")
    plt.xlabel("Día de ciclo (1‑33)"); plt.ylabel("E2 (pg/mL)")
    plt.legend(); plt.tight_layout(); plt.savefig(out_png, dpi=150)
    plt.close()
    logger.info("✅ Forecast guardado: %s", out_png)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python -m src.forecast input.json"); sys.exit(1)
    forecast_from_json(sys.argv[1])

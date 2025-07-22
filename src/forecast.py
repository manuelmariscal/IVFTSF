"""
forecast.py – Forecast 1..33 con MC Dropout y mismas entradas (days/phase).
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

def phase_from_day(days: np.ndarray) -> np.ndarray:
    phase = np.full_like(days, 3)
    phase[days <= 28] = 2
    phase[days <= 14] = 1
    phase[days <= 9]  = 0
    return phase

def load_ckpt():
    return torch.load(MODEL_DIR/"model.pt", map_location=DEVICE, weights_only=False)

def mc_dropout_predict(model, x, pid, days, phase, mask, mc=100):
    model.train(True)
    preds = []
    with torch.no_grad():
        for _ in range(mc):
            y_log = model(x, pid, days=days, phase=phase, src_key_padding_mask=~mask)
            preds.append(torch.expm1(y_log).clamp_min(0).cpu().numpy())
    arr = np.stack(preds)
    return arr.mean(0).squeeze(0), arr.std(0).squeeze(0)

def create_sequence(js, means, stds):
    obs = pd.DataFrame(js["observations"])
    S   = 33
    full_days = np.arange(1, S+1)

    ang = 2*math.pi*(obs["measure_day"]-1)/28.0
    obs["pos_sin"], obs["pos_cos"] = np.sin(ang), np.cos(ang)

    feat = CSV_FEATURES + ["pos_sin","pos_cos"]
    for c in feat:
        if c not in obs.columns:
            obs[c] = 0.0

    obs = obs.set_index("measure_day").sort_index()
    if 1 not in obs.index:  logger.warning("No hay observación en día 1; interpolando.")
    if 28 not in obs.index: logger.warning("No hay observación en día 28; interpolando.")

    obs = obs.reindex(full_days, method=None)
    obs[feat] = obs[feat].interpolate("linear").bfill().ffill()

    obs[feat] = (obs[feat] - means[feat]) / stds[feat]

    x     = torch.tensor(obs[feat].to_numpy(np.float32)).unsqueeze(0)
    days  = torch.tensor(full_days, dtype=torch.int64).unsqueeze(0)
    phase = torch.tensor(phase_from_day(full_days), dtype=torch.int64).unsqueeze(0)
    mask  = torch.ones(1, S, dtype=torch.bool)
    return x, days, phase, mask

def forecast_from_json(json_path: str | Path):
    js = json.loads(Path(json_path).read_text())
    ckpt = load_ckpt()

    stats_cols = [*CSV_FEATURES, "pos_sin","pos_cos"]
    means = pd.Series(ckpt["mean"], index=stats_cols)
    stds  = pd.Series(ckpt["std"],  index=stats_cols)

    tr_df   = pd.read_csv(SPLIT_DIR/"train.csv")
    pid2idx = {pid:i for i,pid in enumerate(tr_df["id"].unique())}
    n_pids  = len(pid2idx)

    state   = ckpt["model_state"]
    d_model = state["num_proj.weight"].shape[0]

    model = HormoneTransformer(
        num_features=len(CSV_FEATURES)+2,
        num_patients=n_pids,
        d_model=d_model, nhead=4, num_layers=4, d_ff=512, dropout=0.12
    ).to(DEVICE)
    model.load_state_dict(state, strict=False)

    patient = js["patient_id"]
    if patient in pid2idx:
        pid_idx = pid2idx[patient]
    else:
        logger.warning("Paciente %s no visto en entrenamiento – embedding medio.", patient)
        with torch.no_grad():
            mean_emb = model.pid_emb.weight.mean(0, keepdim=True)
            model.pid_emb = torch.nn.Embedding.from_pretrained(
                mean_emb.repeat(n_pids,1), freeze=False)
        pid_idx = 0

    pid = torch.tensor([pid_idx], dtype=torch.long, device=DEVICE)

    x, days, phase, mask = create_sequence(js, means, stds)
    x, days, phase, mask = [t.to(DEVICE) for t in (x, days, phase, mask)]

    mc = js.get("mc_samples", 100)
    mu, sigma = mc_dropout_predict(model, x, pid, days, phase, mask, mc)

    out_png = Path(js.get("output", f"figures/forecast_{patient}.png"))
    out_csv = out_png.with_suffix(".csv")
    out_png.parent.mkdir(parents=True, exist_ok=True)

    dd = np.arange(1,34)
    df = pd.DataFrame({
        "day": dd,
        "prediction": mu,
        "lower": np.clip(mu-2*sigma, 0, None),
        "upper": mu+2*sigma
    })
    df.to_csv(out_csv, index=False)

    plt.figure(figsize=(10,4))
    plt.plot(df["day"], df["prediction"], label="predicción")
    plt.fill_between(df["day"], df["lower"], df["upper"], alpha=.25, label="±2σ")
    plt.xlabel("Día de ciclo (1–33)")
    plt.ylabel("E2 (pg/mL)")
    plt.title(f"Forecast paciente {patient}")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_png, dpi=150); plt.close()

    logger.info("✅ Forecast guardado: %s (+ CSV)", out_png)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python -m src.forecast forecast_input.json")
        sys.exit(1)
    forecast_from_json(sys.argv[1])

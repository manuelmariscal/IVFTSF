"""
forecast.py – Genera curva 1..33 con MC‑Dropout (±2 σ) y soporte para pacientes nuevos.
"""

from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from .model    import HormoneTransformer
from .dataset  import CSV_FEATURES          # ← solo las columnas de entrada
from .utils    import MODEL_DIR, SPLIT_DIR, setup_logger

logger = setup_logger("forecast")
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
EPS     = 1e-6

# ────────────────────────────────────────────────────────────────────────────
def load_ckpt():
    return torch.load(MODEL_DIR / "model.pt", map_location=DEVICE, weights_only=False)

# -- MC‑Dropout --------------------------------------------------------------
def mc_dropout_predict(
        model: HormoneTransformer,
        x: torch.Tensor, pid: torch.Tensor,
        days: torch.Tensor, phase: torch.Tensor,
        mask: torch.Tensor, mc_samples: int = 100
):
    """
    x      : (1,S,F)
    pid    : (1,)
    days   : (1,S)
    phase  : (1,S)
    mask   : (1,S)  True -> token real
    """
    model.train(True)                 # activar dropout
    preds = []
    with torch.no_grad():
        for _ in range(mc_samples):
            y_log = model(
                x, pid, days, phase,
                src_key_padding_mask=~mask
            )                         # (1,S)
            preds.append(torch.expm1(y_log).clamp_min(0).cpu().numpy())
    arr = np.stack(preds)             # (mc,1,S)
    return arr.mean(0).squeeze(0), arr.std(0).squeeze(0)

# -- Construye la secuencia 1..33 -------------------------------------------
def create_sequence_from_json(js: dict,
                              means: pd.Series,
                              stds:  pd.Series):
    """
    Devuelve:
        x      : (1,S=33,F)  features normalizadas
        days   : (1,33)      1..33
        phase  : (1,33)      0=follicular,1=ovulatoria,2=lútea
        mask   : (1,33)      todos True (no hay padding)
    """
    obs = pd.DataFrame(js["observations"])

    # día 1‑33 completo
    days_full = np.arange(1, 34)

    # posicionales seno/coseno
    ang = 2 * math.pi * (obs["measure_day"] - 1) / 28.0
    obs["pos_sin"] = np.sin(ang)
    obs["pos_cos"] = np.cos(ang)

    feat_cols = CSV_FEATURES + ["pos_sin", "pos_cos"]

    # asegúrate de que todas las columnas existan
    for c in feat_cols:
        if c not in obs.columns:
            obs[c] = 0.0

    # re‑indexar a 1..33 e interpolar linealmente
    obs = (
        obs.set_index("measure_day")
           .sort_index()
           .reindex(days_full)
    )
    if obs.iloc[0].isna().all():
        logger.warning("No hay observación en día 1; se extrapola hacia atrás.")
    if obs.iloc[27].isna().all():
        logger.warning("No hay observación en día 28; extrapolación hacia adelante.")
    obs[feat_cols] = obs[feat_cols].interpolate("linear").bfill().ffill()

    # normalizar
    obs_norm = obs[feat_cols].copy()
    for c in feat_cols:
        obs_norm[c] = (obs_norm[c] - means[c]) / stds[c]

    x     = torch.tensor(obs_norm.to_numpy(np.float32)).unsqueeze(0)          # (1,33,F)
    mask  = torch.ones(1, 33, dtype=torch.bool)
    days  = torch.tensor(days_full, dtype=torch.int64).unsqueeze(0)           # (1,33)
    phase = torch.where(days <= 9, 0,
             torch.where(days <= 14, 1, 2)).to(torch.int64)                  # (1,33)

    return x, days, phase, mask

# -- Pipeline completo -------------------------------------------------------
def forecast_from_json(json_path: str | Path):
    js   = json.loads(Path(json_path).read_text())
    ckpt = load_ckpt()

    means = pd.Series(ckpt["mean"], index=[*CSV_FEATURES, "pos_sin", "pos_cos"])
    stds  = pd.Series(ckpt["std"],  index=[*CSV_FEATURES, "pos_sin", "pos_cos"])

    # pacientes vistos en entrenamiento
    train_df  = pd.read_csv(SPLIT_DIR / "train.csv")
    pid2idx   = {pid: i for i, pid in enumerate(train_df["id"].unique())}
    N_patients = len(pid2idx)

    # reconstruir modelo con mismo d_model
    d_model = ckpt["model_state"]["num_proj.weight"].shape[0]
    model   = HormoneTransformer(
        num_features=len(CSV_FEATURES)+2,
        num_patients=N_patients,
        d_model=d_model, nhead=4,
        num_layers=4, d_ff=512, dropout=0.12
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"], strict=False)

    # id de paciente
    patient_id = js["patient_id"]
    if patient_id in pid2idx:
        pid_idx = pid2idx[patient_id]
    else:
        logger.warning("Paciente %s no visto en entrenamiento – usando embedding medio.", patient_id)
        pid_idx = 0
        with torch.no_grad():
            emb_mean = model.pid_emb.weight.mean(0, keepdim=True)
            model.pid_emb = torch.nn.Embedding.from_pretrained(
                emb_mean.repeat(N_patients,1), freeze=False
            )

    # ---- tensorización de la secuencia
    x, days, phase, mask = create_sequence_from_json(js, means, stds)
    pid_tensor = torch.tensor([pid_idx], dtype=torch.long)

    mc_samples = js.get("mc_samples", 100)
    μ, σ = mc_dropout_predict(
        model,
        x.to(DEVICE),
        pid_tensor.to(DEVICE),
        days.to(DEVICE),
        phase.to(DEVICE),
        mask.to(DEVICE),
        mc_samples=mc_samples
    )

    # -- guardar resultados --------------------------------------------------
    out_png = Path(js.get("output", f"figures/forecast_{patient_id}.png"))
    out_csv = out_png.with_suffix(".csv")
    out_png.parent.mkdir(parents=True, exist_ok=True)

    days_np = np.arange(1,34)
    df_out  = pd.DataFrame({
        "day": days_np,
        "prediction": μ,
        "lower": np.clip(μ - 2*σ, 0, None),
        "upper": μ + 2*σ
    })
    df_out.to_csv(out_csv, index=False)

    plt.figure(figsize=(10,4))
    plt.plot(days_np, μ, label="Predicción")
    plt.fill_between(days_np, df_out["lower"], df_out["upper"],
                     alpha=0.3, label="±2σ")
    plt.xlabel("Día ciclo (1‑33)")
    plt.ylabel("Estradiol (E2)")
    plt.title(f"Forecast paciente {patient_id}")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_png, dpi=150); plt.close()

    logger.info("Forecast guardado → %s  y  %s", out_png, out_csv)

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    forecast_from_json(sys.argv[1])

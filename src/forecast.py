"""
forecast.py – Genera curva 1..33 con MC Dropout y soporte para pacientes nuevos.
"""

from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from .model import HormoneTransformer
from .dataset import CSV_FEATURES
from .utils import MODEL_DIR, SPLIT_DIR, GOLD_STD, setup_logger

logger = setup_logger("forecast")
DEVICE = "cpu"
EPS = 1e-6

def load_ckpt():
    ckpt = torch.load(MODEL_DIR / "model.pt", map_location=DEVICE, weights_only=False)
    return ckpt

def mc_dropout_predict(model, x, pid, mask, mc_samples=100):
    model.train(True)  # activar dropout
    preds = []
    with torch.no_grad():
        for _ in range(mc_samples):
            y_log = model(x, pid, src_key_padding_mask=~mask)
            preds.append(torch.expm1(y_log).clamp_min(0).cpu().numpy())
    arr = np.stack(preds)  # (mc, 1, S)
    return arr.mean(0).squeeze(0), arr.std(0).squeeze(0)

def create_sequence_from_json(js: dict, means: pd.Series, stds: pd.Series):
    """
    Construye matriz (1,S,F) para días 1..33 interpolando características.
    """
    obs = pd.DataFrame(js["observations"])
    # días requeridos 1..33
    days_full = np.arange(1, 34)

    # añadir posicionales a las observaciones
    def add_pos(df):
        ang = 2 * math.pi * (df["measure_day"] - 1) / 28.0
        df["pos_sin"] = np.sin(ang)
        df["pos_cos"] = np.cos(ang)
        return df
    obs = add_pos(obs)

    feat_cols = CSV_FEATURES + ["pos_sin", "pos_cos"]
    # normalizar usando stats entrenamiento
    for c in feat_cols:
        if c not in obs.columns:
            obs[c] = 0.0
    # interpolación por día
    obs = obs.set_index("measure_day").sort_index()
    obs = obs.reindex(days_full, method=None)
    # advertencias si faltan extremos
    if 1 not in obs.index or pd.isna(obs.loc[1, feat_cols]).all():
        logger.warning("No hay observación en día 1; se extrapola hacia atrás.")
    if 28 not in obs.index or pd.isna(obs.loc[28, feat_cols]).all():
        logger.warning("No hay observación en día 28; extrapolación hacia adelante.")
    obs[feat_cols] = obs[feat_cols].interpolate(method="linear").bfill().ffill()

    # normalizar
    for c in feat_cols:
        obs[c] = (obs[c] - means[c]) / stds[c]

    x = torch.tensor(obs[feat_cols].to_numpy(dtype=np.float32)).unsqueeze(0)
    mask = torch.ones(1, len(days_full), dtype=bool)
    return x, mask

def forecast_from_json(json_path: str | Path):
    js = json.loads(Path(json_path).read_text())
    ckpt = load_ckpt()
    means = pd.Series(ckpt["mean"], index=[*CSV_FEATURES, "pos_sin", "pos_cos"])
    stds  = pd.Series(ckpt["std"],  index=[*CSV_FEATURES, "pos_sin", "pos_cos"])

    # mapa de pacientes vistos
    train_df = pd.read_csv(SPLIT_DIR / "train.csv")
    pid2idx = {pid: i for i, pid in enumerate(train_df["id"].unique())}
    num_patients = len(pid2idx)

    model = HormoneTransformer(
        num_features=len(CSV_FEATURES)+2,
        num_patients=num_patients,
        d_model=ckpt["model_state"]["num_proj.weight"].shape[0],
        nhead=4, num_layers=4, d_ff=512, dropout=0.12
    )
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.to(DEVICE)

    patient_id = js["patient_id"]
    if patient_id in pid2idx:
        pid_idx = pid2idx[patient_id]
    else:
        logger.warning("Paciente %s no visto en entrenamiento – usando embedding medio.", patient_id)
        # usar índice 0 y luego promediar embedding manualmente
        pid_idx = 0
        with torch.no_grad():
            emb = model.pid_emb.weight.mean(0, keepdim=True)
            model.pid_emb = torch.nn.Embedding.from_pretrained(emb.repeat(num_patients,1), freeze=False)

    x, mask = create_sequence_from_json(js, means, stds)
    pid_tensor = torch.tensor([pid_idx], dtype=torch.long)

    mc_samples = js.get("mc_samples", 100)
    mean_pred, std_pred = mc_dropout_predict(model, x.to(DEVICE), pid_tensor.to(DEVICE), mask.to(DEVICE), mc_samples)

    days = np.arange(1, 34)
    df_out = pd.DataFrame({
        "day": days,
        "prediction": mean_pred,
        "lower": np.clip(mean_pred - 2*std_pred, 0, None),
        "upper": mean_pred + 2*std_pred,
    })

    out_png = Path(js.get("output", f"figures/forecast_{patient_id}.png"))
    out_csv = out_png.with_suffix(".csv")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False)

    plt.figure(figsize=(10,4))
    plt.plot(df_out["day"], df_out["prediction"], label="Predicción")
    plt.fill_between(df_out["day"], df_out["lower"], df_out["upper"], alpha=0.3, label="±2σ")
    plt.xlabel("Día ciclo (1..33)")
    plt.ylabel("E2")
    plt.title(f"Forecast paciente {patient_id}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    logger.info("Forecast guardado en %s y %s", out_png, out_csv)

if __name__ == "__main__":
    import sys
    json_file = sys.argv[1]
    forecast_from_json(json_file)

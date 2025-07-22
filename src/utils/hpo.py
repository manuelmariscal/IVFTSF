"""
utils/hpo.py – búsqueda Optuna (epochs bajos, muchos trials, GPU/CPU).
"""
from __future__ import annotations
import argparse, json, os, optuna, pandas as pd, torch, numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
from ..dataset import HormoneDataset, CSV_FEATURES
from ..model import HormoneTransformer
from . import SPLIT_DIR, MODEL_DIR, auto_mkdir, setup_logger
log = setup_logger("hpo")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPS=1e-6; HPO_DIR=MODEL_DIR/"hpo"; HPO_DIR.mkdir(parents=True, exist_ok=True)

# ---------- loaders ----------
def loaders(bs):
    tr = HormoneDataset(pd.read_csv(SPLIT_DIR/"train.csv"))
    va = HormoneDataset(pd.read_csv(SPLIT_DIR/"val.csv"))
    opts=dict(num_workers=2,pin_memory=torch.cuda.is_available())
    return tr, DataLoader(tr,batch_size=bs,shuffle=True,**opts), \
           DataLoader(va,batch_size=bs,**opts)

def mape_epoch(dl,model,opt=None):
    train=opt is not None; model.train(train)
    tot,n=0.,0
    for b in dl:
        m=b["mask"].to(DEVICE); real=m.sum()
        if real==0:continue
        x=b["x"].to(DEVICE); y=b["y"].to(DEVICE)
        pid=b["pid"].to(DEVICE); ph=b["phase"].to(DEVICE)
        ŷ_log=model(x,pid,ph,src_key_padding_mask=~m)
        loss=torch.abs((torch.expm1(ŷ_log)-y)/(y.abs()+EPS))[m].mean()
        if train: opt.zero_grad(); loss.backward(); opt.step()
        tot+=loss.item()*real.item(); n+=real.item()
    return (tot/max(n,1))*100.

# ---------- objective ----------
def objective(trial):
    p=dict(
        d_model=trial.suggest_int("d_model",32,96,step=32),
        nhead=trial.suggest_categorical("nhead",[2,4]),
        num_layers=trial.suggest_int("num_layers",2,4),
        d_ff=trial.suggest_int("d_ff",128,512,step=128),
        dropout=trial.suggest_float("dropout",0.05,0.25),
        lr=trial.suggest_float("lr",1e-4,3e-3,log=True),
        batch_size=trial.suggest_categorical("batch_size",[64,128]),
        epochs=trial.study.user_attrs.get("epochs"),
    )
    tr_ds,tr_ld,va_ld=loaders(p["batch_size"])
    model=HormoneTransformer(
        num_features=len(CSV_FEATURES)+2,
        num_patients=tr_ds.num_patients,**{k:p[k] for k in
         ["d_model","nhead","num_layers","d_ff","dropout"]}).to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=p["lr"])
    best=np.inf
    for ep in range(1,p["epochs"]+1):
        mape_epoch(tr_ld,model,opt); val=mape_epoch(va_ld,model)
        best=min(best,val)
        trial.report(val,ep)
        if trial.should_prune(): raise optuna.TrialPruned()
    return best

# ---------- main ----------
@auto_mkdir(1)
def save_json(o,p):p.write_text(json.dumps(o,indent=2))

def main(n_trials:int,n_jobs:int,epochs:int):
    db=HPO_DIR/"hpo.db"
    if db.exists(): db.unlink()
    study=optuna.create_study(direction="minimize",
            storage=f"sqlite:///{db}",study_name="IVF_HPO",load_if_exists=False)
    study.set_user_attr("epochs",epochs)
    log.info("Optuna %d trials × %d epochs | jobs=%d",n_trials,epochs,n_jobs)
    study.optimize(objective,n_trials=n_trials,n_jobs=n_jobs)
    log.info("🏆  Best %.2f%% – %s",study.best_value,study.best_params)
    save_json(study.best_params,HPO_DIR/"best_params.json")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--n_trials",type=int,default=80)
    ap.add_argument("--n_jobs",type=int,default=os.cpu_count()//2)
    ap.add_argument("--epochs",type=int,default=12,
                    help="epochs por trial (mantener bajos)")
    main(**vars(ap.parse_args()))

"""
run.py – Lanzador principal (split, train, hpo, forecast).
"""

import argparse
from pathlib import Path

from .utils import setup_logger, SPLIT_DIR, RAW_DATA
from .utils.split_dataset import main as split_main
from .train import main as train_main
from .forecast import main as forecast_main

logger = setup_logger("run")

# ─────────────── wrapper funcs ─────────────────────────────────────────── #
def pipeline_all(args):
    split_main(); train_main()
    logger.info("✔  Pipeline 'all' completed")

def pipeline_split(args):
    split_main()

def pipeline_train(args):
    if not (SPLIT_DIR / "train.csv").exists():
        logger.error("Splits not found. Run --run split_data first."); return
    train_main()

def pipeline_hpo(args):
    if not (SPLIT_DIR / "train.csv").exists():
        logger.error("Splits not found. Run --run split_data first."); return
    from .utils.hpo import main as run_hpo
    run_hpo(n_trials=args.trials, n_jobs=args.jobs, fresh=bool(args.fresh))

def pipeline_forecast(args):
    forecast_main(Path(RAW_DATA), args.patient_id, args.age)

# ─────────────── CLI ───────────────────────────────────────────────────── #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run",
                    choices=["all", "split_data", "train", "hpo", "forecast"],
                    required=True)
    ap.add_argument("--patient_id", type=int, help="For forecast")
    ap.add_argument("--age", type=int, help="For forecast")
    ap.add_argument("--trials", type=int, default=400, help="For HPO")
    ap.add_argument("--jobs",   type=int, default=8)
    ap.add_argument("--fresh",  type=int, choices=[0,1], default=1,
                    help="1=start new study (delete DB), 0=resume")
    args = ap.parse_args()

    {
        "all":        pipeline_all,
        "split_data": pipeline_split,
        "train":      pipeline_train,
        "hpo":        pipeline_hpo,
        "forecast":   pipeline_forecast,
    }[args.run](args)

if __name__ == "__main__":
    main()

"""
src/utils/split_dataset.py – 70‑20‑10 split by patient ID.

Run:
    python -m src.utils.split_dataset
"""

import numpy as np
import pandas as pd

from .config import RAW_DATA, SPLIT_DIR
from .helpers import setup_logger

logger = setup_logger("split_dataset")

ID_COL = "id"
TRAIN_PCT, VAL_PCT = 0.7, 0.2          # test = 0.1


def main():
    logger.info("Reading raw CSV: %s", RAW_DATA)
    df = pd.read_csv(RAW_DATA)

    ids = df[ID_COL].unique()
    rng = np.random.default_rng(42)
    rng.shuffle(ids)

    n_total = len(ids)
    n_train = int(n_total * TRAIN_PCT)
    n_val   = int(n_total * VAL_PCT)

    splits = {
        "train": ids[:n_train],
        "val":   ids[n_train : n_train + n_val],
        "test":  ids[n_train + n_val :],
    }

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for name, subset in splits.items():
        out_file = SPLIT_DIR / f"{name}.csv"
        df[df[ID_COL].isin(subset)].to_csv(out_file, index=False)
        logger.info("Saved %s (%d patients) → %s", name, len(subset), out_file)

    logger.info("✔  All splits completed")


if __name__ == "__main__":
    main()

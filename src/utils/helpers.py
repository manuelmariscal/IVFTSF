"""
helpers.py – logger DEBUG en consola y archivos, autodirectorios,
             utilidades de hardware (solo CPU).
"""
from functools import wraps
import logging, sys
from pathlib import Path
from .config import LOG_DIR

# ---------------- auto‑mkdir ---------------- #
def auto_mkdir(idx: int = 0):
    def deco(func):
        @wraps(func)
        def wrapper(*args, **kw):
            Path(args[idx]).parent.mkdir(parents=True, exist_ok=True)
            return func(*args, **kw)
        return wrapper
    return deco

# ---------------- logger ---------------- #
def setup_logger(name: str) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger(name)
    if lg.handlers:
        return lg

    lg.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")

    for path in (LOG_DIR / f"{name}.log", LOG_DIR / "all.log"):
        fh = logging.FileHandler(path, mode="a")
        fh.setFormatter(fmt); fh.setLevel(logging.DEBUG); lg.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt); sh.setLevel(logging.DEBUG); lg.addHandler(sh)

    lg.propagate = False
    return lg

# ------------- hardware utils (CPU‑only) ------------- #
def get_device() -> str:
    """Devuelve siempre 'cpu' para forzar cómputo en CPU."""
    return "cpu"

def set_cpu_threads():
    import torch, multiprocessing as mp
    n = max(1, mp.cpu_count() - 1)
    torch.set_num_threads(n)
    return n

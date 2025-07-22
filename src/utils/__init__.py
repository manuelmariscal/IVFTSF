"""
src/utils/__init__.py – Re‑exporta utilidades SIN importar hpo (evita ciclo).
"""
from .config import (
    ROOT_DIR, DATA_DIR, RAW_DATA, SPLIT_DIR,
    MODEL_DIR, FIG_DIR, LOG_DIR, HP, GOLD_STD
)
from .helpers import (
    auto_mkdir,
    setup_logger,
    get_device,
    set_cpu_threads,
)

__all__ = [
    # paths
    "ROOT_DIR", "DATA_DIR", "RAW_DATA", "SPLIT_DIR",
    "MODEL_DIR", "FIG_DIR", "LOG_DIR", "GOLD_STD",
    # hyper‑params
    "HP",
    # helpers
    "auto_mkdir", "setup_logger", "get_device", "set_cpu_threads",
]

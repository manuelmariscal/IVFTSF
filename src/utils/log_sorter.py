"""
log_sorter.py – Ordena incrementalmente un archivo de log por timestamp.

Uso:
    # ordena cada 20 s en segundo plano
    python -m src.utils.log_sorter --file logs/hpo.log --interval 20
"""
import argparse, time, fcntl, os, shutil
from pathlib import Path
from datetime import datetime
from typing import List

TS_LEN = 23  # 'YYYY‑MM‑DD HH:MM:SS,mmm' -> 23 chars

def parse_ts(line: str) -> float:
    try:
        return datetime.strptime(line[:TS_LEN], "%Y-%m-%d %H:%M:%S,%f").timestamp()
    except ValueError:
        return 0.0

def sort_chunk(lines: List[str]) -> List[str]:
    return sorted(lines, key=parse_ts)

def incremental_sort(path: Path, interval: int):
    tmp = path.with_suffix(".tmp")
    last_size = 0
    while True:
        try:
            with open(path, "r+") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                f.seek(last_size)
                new = f.readlines()
                if new:
                    last_size = f.tell()
                    lines = sort_chunk(new)
                    f.seek(last_size - len(new[0])*len(new))  # reposition
                    f.writelines(lines)
                fcntl.flock(f, fcntl.LOCK_UN)
        except FileNotFoundError:
            pass
        time.sleep(interval)

def main(file: Path, interval: int):
    if not file.exists():
        print(f"{file} does not exist yet – waiting for creation.")
    incremental_sort(file, interval)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path, required=True, help="log file to order")
    ap.add_argument("--interval", type=int, default=20, help="seconds between passes")
    args = ap.parse_args()
    main(args.file, args.interval)

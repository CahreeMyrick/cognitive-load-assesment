
from __future__ import annotations
import os, yaml
import numpy as np

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def downsample_dataframe(df, num_samples: int):
    """Deterministic downsampling by index stride (like your notebook)."""
    if len(df) == num_samples:
        return df.reset_index(drop=True)
    factor = len(df) / float(num_samples)
    idx = np.floor(np.arange(0, len(df), factor)).astype(int)
    idx = np.clip(idx, 0, len(df)-1)
    return df.iloc[idx].reset_index(drop=True)

def block_specific_filename(block: int, fname: str) -> str:
    """Apply block-based renames used in the original notebook for blocks 2 and 8."""
    base, dot, ext = fname.rpartition(".")
    if not dot:  # no extension
        base, ext = fname, ""
    suffix = ""
    if block == 2:
        suffix = "mathtest"
    elif block == 8:
        suffix = "IQtest"
    if suffix:
        return f"{base}{suffix}.{ext}" if ext else f"{base}{suffix}"
    return fname

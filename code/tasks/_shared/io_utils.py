"""
I/O utilities conforming to WCC naming conventions.

CSV: UTF-8 no BOM, comma separator, header row, no index column.
NPZ: float32 default, standard compression.
JSON: UTF-8, indent=2.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def save_csv(
    df: pd.DataFrame,
    path: Path,
    expected_columns: Optional[list[str]] = None,
) -> None:
    """Save DataFrame as CSV with WCC conventions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if expected_columns is not None:
        missing = set(expected_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing expected columns: {missing}")
    df.to_csv(path, index=False, encoding="utf-8")


def save_npz(
    arrays: dict[str, np.ndarray],
    path: Path,
    dtype=np.float32,
) -> None:
    """Save dict of arrays as NPZ, casting to dtype."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cast = {k: v.astype(dtype) for k, v in arrays.items()}
    np.savez(path, **cast)


def save_json(data: dict, path: Path) -> None:
    """Save dict as JSON with UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    """Load NPZ and return as plain dict (not lazy NpzFile)."""
    with np.load(path) as npz:
        return {k: npz[k] for k in npz.files}


def load_head_activations(path: Path) -> np.ndarray:
    """
    Load per-head activations from NPZ.

    Expected shape: single array keyed "activations" → (N, L, H, D)
    Or multiple prompt-keyed arrays → stacked into (N, L, H, D).
    """
    data = load_npz(path)
    if "activations" in data:
        return data["activations"]
    # Stack prompt-keyed arrays
    keys = sorted(data.keys(), key=lambda k: int(k) if k.isdigit() else k)
    return np.stack([data[k] for k in keys], axis=0)

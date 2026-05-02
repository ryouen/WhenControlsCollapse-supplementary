"""
Shared scoring utilities: perturbation metrics + cell classification.

Canonical column names per spec/WCC_naming_conventions.md §6:
- Perturbation: pert_raw_l2, pert_relative_l2, pert_cosine_distance, pert_standardized_l2
- Cell: hi_imp, hi_pert, cell (A/B/C/D)

Cell label semantics (use exactly these definitions):
    A = hi_imp AND hi_pert    ("true positive")
    B = hi_imp AND lo_pert    ("quiet specialist")
    C = lo_imp AND hi_pert    ("false positive" — paper's central concern)
    D = lo_imp AND lo_pert    ("true null baseline")
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# Perturbation metrics
# ---------------------------------------------------------------------------

def compute_perturbation_metrics(
    clean_z: np.ndarray,
    corrupt_z: np.ndarray,
) -> pd.DataFrame:
    """
    Compute 4 perturbation metrics for each (layer, head).

    Parameters
    ----------
    clean_z : (N, L, H, D) float32 — per-head activations under clean input
    corrupt_z : (N, L, H, D) float32 — per-head activations under corrupt input

    Returns
    -------
    DataFrame with columns:
        layer, head, pert_raw_l2, pert_relative_l2,
        pert_cosine_distance, pert_standardized_l2
    """
    assert clean_z.shape == corrupt_z.shape, (
        f"Shape mismatch: clean {clean_z.shape} vs corrupt {corrupt_z.shape}"
    )
    N, L, H, D = clean_z.shape

    diff = clean_z - corrupt_z  # (N, L, H, D)

    # Raw L2: mean_i(||clean[i] - corrupt[i]||_2)
    raw_l2 = np.linalg.norm(diff, axis=-1).mean(axis=0)  # (L, H)

    # Relative L2: mean_i(||diff[i]||_2 / ||clean[i]||_2)
    clean_norm = np.linalg.norm(clean_z, axis=-1)  # (N, L, H)
    # Avoid division by zero
    clean_norm_safe = np.where(clean_norm > 0, clean_norm, 1.0)
    relative_l2 = (np.linalg.norm(diff, axis=-1) / clean_norm_safe).mean(axis=0)

    # Cosine distance: mean_i(1 - cos(clean[i], corrupt[i]))
    dot = (clean_z * corrupt_z).sum(axis=-1)  # (N, L, H)
    corrupt_norm = np.linalg.norm(corrupt_z, axis=-1)
    denom = clean_norm * corrupt_norm
    denom_safe = np.where(denom > 0, denom, 1.0)
    cosine_sim = dot / denom_safe
    cosine_distance = (1.0 - cosine_sim).mean(axis=0)  # (L, H)

    # Standardized L2: mean_i(||(diff[i] - mean_diff) / std_diff||_2)
    # std computed per (layer, head, dim) across prompts
    diff_mean = diff.mean(axis=0, keepdims=True)  # (1, L, H, D)
    diff_std = diff.std(axis=0, keepdims=True)     # (1, L, H, D)
    diff_std_safe = np.where(diff_std > 0, diff_std, 1.0)
    standardized = (diff - diff_mean) / diff_std_safe
    standardized_l2 = np.linalg.norm(standardized, axis=-1).mean(axis=0)  # (L, H)

    rows = []
    for layer in range(L):
        for head in range(H):
            rows.append({
                "layer": layer,
                "head": head,
                "pert_raw_l2": float(raw_l2[layer, head]),
                "pert_relative_l2": float(relative_l2[layer, head]),
                "pert_cosine_distance": float(cosine_distance[layer, head]),
                "pert_standardized_l2": float(standardized_l2[layer, head]),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cell classification
# ---------------------------------------------------------------------------

def classify_cells(
    importance: pd.Series,
    perturbation: pd.Series,
) -> pd.DataFrame:
    """
    Median-split cell classification.

    Parameters
    ----------
    importance : Series indexed by (layer, head) or flat index
    perturbation : Series with matching index

    Returns
    -------
    DataFrame with columns:
        imp_value, pert_value, hi_imp, hi_pert, cell,
        imp_median, pert_median
    """
    assert len(importance) == len(perturbation), (
        f"Length mismatch: importance={len(importance)}, perturbation={len(perturbation)}"
    )

    imp_median = float(importance.median())
    pert_median = float(perturbation.median())

    hi_imp = importance >= imp_median
    hi_pert = perturbation >= pert_median

    # Cell assignment — CRITICAL: must match WCC_naming_conventions.md §2.2
    cell = pd.Series("D", index=importance.index)
    cell[hi_imp & hi_pert] = "A"
    cell[hi_imp & ~hi_pert] = "B"
    cell[~hi_imp & hi_pert] = "C"
    cell[~hi_imp & ~hi_pert] = "D"

    return pd.DataFrame({
        "imp_value": importance.values,
        "pert_value": perturbation.values,
        "hi_imp": hi_imp.values,
        "hi_pert": hi_pert.values,
        "cell": cell.values,
        "imp_median": imp_median,
        "pert_median": pert_median,
    })


def validate_cell_assignment(cell_df: pd.DataFrame) -> None:
    """
    Verify cell labels match the canonical definition.
    Raises AssertionError if any label is wrong.
    """
    for _, row in cell_df.iterrows():
        hi_i, hi_p, c = row["hi_imp"], row["hi_pert"], row["cell"]
        if hi_i and hi_p:
            assert c == "A", f"hi_imp=True, hi_pert=True should be A, got {c}"
        elif hi_i and not hi_p:
            assert c == "B", f"hi_imp=True, hi_pert=False should be B, got {c}"
        elif not hi_i and hi_p:
            assert c == "C", f"hi_imp=False, hi_pert=True should be C, got {c}"
        else:
            assert c == "D", f"hi_imp=False, hi_pert=False should be D, got {c}"


def compute_metric_summary(
    importance_df: pd.DataFrame,
    perturbation_df: pd.DataFrame,
    imp_col: str,
    pert_col: str,
) -> dict:
    """
    Compute summary statistics for a given importance × perturbation metric pair.

    Returns dict with: spearman_rho, spearman_p, pearson_r, pearson_p,
                       cell_counts (A/B/C/D), imp_median, pert_median.
    """
    imp = importance_df[imp_col]
    pert = perturbation_df[pert_col]

    rho, rho_p = spearmanr(imp, pert)
    r, r_p = spearmanr(imp.abs(), pert) if imp_col.endswith("_signed") else (rho, rho_p)

    cells = classify_cells(imp, pert)
    validate_cell_assignment(cells)
    counts = cells["cell"].value_counts().to_dict()

    return {
        "importance_metric": imp_col,
        "perturbation_metric": pert_col,
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
        "cell_counts": {c: counts.get(c, 0) for c in "ABCD"},
        "imp_median": float(imp.median()),
        "pert_median": float(pert.median()),
    }

"""
Shared Phase 90: Post-hoc analysis (regression, matched-head, per-layer R²).

Produces inputs for paper Tables 1, B1, B2 and the cross-task comparison
in inventory.md §4.

All tasks share the same regression model:
    OLS: delta_target ~ importance + perturbation + interaction
    → R²(imp), R²(pert), partial R² for each
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu
from pathlib import Path

from .config_writer import phase_timer
from .io_utils import save_csv, save_json


def run_regression(
    importance: pd.Series,
    perturbation: pd.Series,
    patching_effect: pd.Series,
) -> dict:
    """
    OLS regression: patching_effect ~ importance + perturbation + interaction.

    Returns dict with R² breakdown and partial R² values.
    """
    from sklearn.linear_model import LinearRegression

    X_full = np.column_stack([
        importance.values,
        perturbation.values,
        importance.values * perturbation.values,
    ])
    y = patching_effect.abs().values

    # Full model
    lr_full = LinearRegression().fit(X_full, y)
    r2_full = lr_full.score(X_full, y)

    # Importance only
    X_imp = importance.values.reshape(-1, 1)
    r2_imp = LinearRegression().fit(X_imp, y).score(X_imp, y)

    # Perturbation only
    X_pert = perturbation.values.reshape(-1, 1)
    r2_pert = LinearRegression().fit(X_pert, y).score(X_pert, y)

    # Partial R²
    partial_r2_imp = r2_full - r2_pert
    partial_r2_pert = r2_full - r2_imp

    # Spearman correlations
    rho_imp, rho_imp_p = spearmanr(importance, patching_effect.abs())
    rho_pert, rho_pert_p = spearmanr(perturbation, patching_effect.abs())

    return {
        "r2_full": float(r2_full),
        "r2_importance": float(r2_imp),
        "r2_perturbation": float(r2_pert),
        "partial_r2_importance": float(partial_r2_imp),
        "partial_r2_perturbation": float(partial_r2_pert),
        "spearman_rho_imp": float(rho_imp),
        "spearman_rho_imp_p": float(rho_imp_p),
        "spearman_rho_pert": float(rho_pert),
        "spearman_rho_pert_p": float(rho_pert_p),
    }


def run_matched_head_analysis(
    cell_df: pd.DataFrame,
    patching_effects: pd.Series,
) -> dict:
    """
    Matched-head analysis: compare high-perturbation vs low-perturbation heads
    matched on importance. Mann-Whitney U test.

    Uses Cell C (lo_imp, hi_pert) vs Cell D (lo_imp, lo_pert) —
    matched on "low importance" stratum.
    """
    cell_c_effects = patching_effects[cell_df["cell"] == "C"].abs()
    cell_d_effects = patching_effects[cell_df["cell"] == "D"].abs()

    if len(cell_c_effects) == 0 or len(cell_d_effects) == 0:
        return {"error": "Empty cell C or D"}

    stat, p_value = mannwhitneyu(cell_c_effects, cell_d_effects, alternative="greater")

    # Effect size: rank-biserial correlation
    n_c, n_d = len(cell_c_effects), len(cell_d_effects)
    rank_biserial = 1 - (2 * stat) / (n_c * n_d)

    return {
        "mann_whitney_u": float(stat),
        "p_value": float(p_value),
        "rank_biserial": float(rank_biserial),
        "cell_c_median": float(cell_c_effects.median()),
        "cell_d_median": float(cell_d_effects.median()),
        "cell_c_mean": float(cell_c_effects.mean()),
        "cell_d_mean": float(cell_d_effects.mean()),
        "effect_ratio": float(cell_c_effects.mean() / cell_d_effects.mean())
        if cell_d_effects.mean() > 0 else float("inf"),
    }


def run_per_layer_r2(
    importance: pd.Series,
    perturbation: pd.Series,
    patching_effect: pd.Series,
    layers: pd.Series,
    n_layers: int,
) -> pd.DataFrame:
    """Per-layer R² breakdown."""
    from sklearn.linear_model import LinearRegression

    rows = []
    for layer in range(n_layers):
        mask = layers == layer
        if mask.sum() < 3:
            continue

        imp_l = importance[mask].values.reshape(-1, 1)
        pert_l = perturbation[mask].values.reshape(-1, 1)
        y_l = patching_effect[mask].abs().values

        r2_imp = LinearRegression().fit(imp_l, y_l).score(imp_l, y_l)
        r2_pert = LinearRegression().fit(pert_l, y_l).score(pert_l, y_l)

        rows.append({
            "layer": layer,
            "n_heads": int(mask.sum()),
            "r2_importance": float(r2_imp),
            "r2_perturbation": float(r2_pert),
        })

    return pd.DataFrame(rows)


def run_phase90(
    task_root: Path,
    primary_importance: str,
    primary_perturbation: str,
    n_layers: int,
):
    """
    Execute Phase 90 for any task.

    Reads Phase 20 (scoring) and Phase 30 (patching) outputs.
    """
    output_dir = task_root / "90_post_hoc"

    imp_df = pd.read_csv(task_root / "20_scoring" / "importance_per_head.csv")
    pert_df = pd.read_csv(task_root / "20_scoring" / "perturbation_per_head.csv")
    cell_df = pd.read_csv(task_root / "20_scoring" / "cell_classification.csv")
    effects_df = pd.read_csv(task_root / "30_patching" / "individual_head_effects.csv")

    # Mean absolute effect per head
    mean_effects = (
        effects_df.groupby(["layer", "head"])["delta_logit_signed"]
        .agg(mean_abs=lambda x: x.abs().mean())
        .reset_index()
    )

    # Merge
    merged = imp_df.merge(pert_df, on=["layer", "head"])
    merged = merged.merge(mean_effects, on=["layer", "head"])
    merged = merged.merge(cell_df[["layer", "head", "cell"]], on=["layer", "head"])

    with phase_timer(output_dir) as config:

        # Regression
        print("[Phase 90] Running regression...")
        reg_result = run_regression(
            importance=merged[primary_importance],
            perturbation=merged[primary_perturbation],
            patching_effect=merged["mean_abs"],
        )
        save_json(reg_result, output_dir / "regression_summary.json")

        # Matched-head analysis
        print("[Phase 90] Matched-head analysis...")
        matched = run_matched_head_analysis(
            cell_df=merged,
            patching_effects=merged["mean_abs"],
        )
        save_json(matched, output_dir / "matched_head_summary.json")

        # Per-layer R²
        print("[Phase 90] Per-layer R²...")
        layer_r2 = run_per_layer_r2(
            importance=merged[primary_importance],
            perturbation=merged[primary_perturbation],
            patching_effect=merged["mean_abs"],
            layers=merged["layer"],
            n_layers=n_layers,
        )
        save_csv(layer_r2, output_dir / "per_layer_r2.csv")

        # Cross-model row for inventory aggregation
        cross_row = {
            "task_id": task_root.name,
            "primary_importance": primary_importance,
            "primary_perturbation": primary_perturbation,
            **reg_result,
            **{f"matched_{k}": v for k, v in matched.items()},
        }
        save_csv(pd.DataFrame([cross_row]), output_dir / "cross_model_row.csv")

        config.update(reg_result)
        config.update({f"matched_{k}": v for k, v in matched.items()})

    print(f"[Phase 90] Done. R²(imp)={reg_result['r2_importance']:.3f}, "
          f"R²(pert)={reg_result['r2_perturbation']:.3f}")
    return config

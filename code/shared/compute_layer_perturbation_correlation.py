"""Task 1: Layer-depth bias analysis .

For each unit (model or task), computes:
  A1: Spearman ρ(layer, perturbation_l2), p-value, layer_top / layer_0 median ratio
  A2: Global vs within-layer cell-classification agreement + Cell C/D Jaccard
  A3: Cell C layer distribution (global vs within-layer)
  A4: Within-layer C/D gap approximation (using importance-per-head analog)

Outputs:
  cross_model/layer_bias_analysis.csv     — per-unit summary
  cross_model/layer_analysis/{unit}_dual_classification.csv — per-head
  reports/task1_layer_bias.md              — narrative report
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
OUT_CSV = DATA / "cross_model" / "layer_bias_analysis.csv"
OUT_DIR = DATA / "cross_model" / "layer_analysis"
OUT_MD = OUT_DIR / "task1_layer_bias.md"

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Unit configuration
# ============================================================
NOVEL_WORD_PRIMARY = [
    "gemma-2-9b-it",
    "gemma-3-27b-it",
    "gpt-j-6b-fp32",
    "llama-3.1-8b-instruct",
    "llama-3.1-70b-instruct-4bit",
    "olmo-2-13b-instruct",
    "qwen-2.5-7b-instruct",
    "qwen-2.5-14b-instruct",
]
NOVEL_WORD_SUPPLEMENTARY: list[str] = []  # llama-3.1-8b base not bundled in supplementary
GPT2_TASKS = ["ioi", "induction", "greater_than"]  # icl pipeline not in supplementary


def load_unit(unit: str, unit_type: str) -> pd.DataFrame:
    """Return DataFrame with columns: layer, head, imp, pert, cell_global."""
    if unit_type == "novel_word":
        cell = pd.read_csv(DATA / "novel_word" / unit / "20_scoring" / "cell_classification.csv")
        pert_col = "perturbation_L2" if "perturbation_L2" in cell.columns else "perturbation_l2"
        # Cell labels in supplementary CSVs use {hihp, hilp, C, D} naming.
        # Map to {A, B, C, D} so that comparison with within-layer cells (also
        # produced as {A, B, C, D}) is meaningful: hihp == A (hi imp, hi pert),
        # hilp == B (hi imp, lo pert).
        cell_global = cell["cell"].replace({"hihp": "A", "hilp": "B"})
        df = pd.DataFrame(
            {
                "layer": cell["layer"],
                "head": cell["head"],
                "imp": cell["rsa_max"],
                "pert": cell[pert_col],
                "cell_global": cell_global,
            }
        )
    elif unit_type == "gpt2_task":
        cell_all = pd.read_csv(DATA / unit / "20_scoring" / "cell_classification.csv")
        # IOI has concatenated importance_metric rows; filter to signed_ld (paper primary)
        if "importance_metric" in cell_all.columns:
            cell = cell_all[cell_all["importance_metric"] == "signed_ld"].copy()
            if len(cell) == 0:
                # fallback: some tasks may have a single metric with different name
                cell = cell_all.copy()
        else:
            cell = cell_all.copy()
        df = pd.DataFrame(
            {
                "layer": cell["layer"],
                "head": cell["head"],
                "imp": cell["imp_value"] if "imp_value" in cell.columns else cell["importance_signed"],
                "pert": cell["pert_value"] if "pert_value" in cell.columns else cell["perturbation_l2"],
                "cell_global": cell["cell"] if "cell" in cell.columns else cell["cell_signed"],
            }
        )
    else:
        raise ValueError(unit_type)
    return df


def load_head_effect(unit: str, unit_type: str) -> pd.Series | None:
    """Return per-head mean |delta| effect, indexed by (layer, head). None if unavailable."""
    if unit_type == "novel_word":
        p = DATA / "novel_word" / unit / "30_patching" / "individual_head_effects.csv"
        if not p.exists():
            return None
        df = pd.read_csv(p)
        effect = (
            df.assign(abs_delta=df["delta_target_logit_signed"].abs())
            .groupby(["layer", "head"])["abs_delta"]
            .mean()
        )
        return effect
    else:
        # Tasks: use importance_per_head.csv analog
        p = DATA / unit / "20_scoring" / "importance_per_head.csv"
        if not p.exists():
            return None
        df = pd.read_csv(p)
        if unit == "ioi":
            col = "imp_abs_ld"
        elif unit == "induction":
            col = "imp_patching_loss"
        elif unit == "greater_than":
            col = "imp_abs_gt_score_drop"
        elif unit == "icl":
            col = "imp_mean_ablation"
        else:
            return None
        # imp_patching_loss might be signed (loss change); use abs
        effect = df.set_index(["layer", "head"])[col].abs()
        return effect


def compute_within_layer_cell(df: pd.DataFrame) -> pd.Series:
    """Return within-layer cell classification as a Series aligned with df's index.

    Within-layer: compute median imp and median pert per layer, then classify each
    head by (imp > layer_imp_median, pert > layer_pert_median).
    """
    df = df.copy()
    df["imp_layer_median"] = df.groupby("layer")["imp"].transform("median")
    df["pert_layer_median"] = df.groupby("layer")["pert"].transform("median")
    df["hi_imp_within"] = df["imp"] > df["imp_layer_median"]
    df["hi_pert_within"] = df["pert"] > df["pert_layer_median"]
    # Cell labels:
    # A: hi_imp, hi_pert
    # B: hi_imp, lo_pert
    # C: lo_imp, hi_pert
    # D: lo_imp, lo_pert
    cell = np.where(
        df["hi_imp_within"] & df["hi_pert_within"], "A",
        np.where(
            df["hi_imp_within"] & ~df["hi_pert_within"], "B",
            np.where(
                ~df["hi_imp_within"] & df["hi_pert_within"], "C", "D",
            ),
        ),
    )
    return pd.Series(cell, index=df.index, name="cell_within")


def analyze_unit(unit: str, unit_type: str) -> dict:
    """Run all Task 1 analyses on one unit."""
    df = load_unit(unit, unit_type)
    n_heads = len(df)
    n_layers = df["layer"].nunique()

    # A1: Spearman ρ(layer, perturbation)
    rho, p_rho = stats.spearmanr(df["layer"], df["pert"])

    # layer_top / layer_0 median ratio
    top_layer = df["layer"].max()
    bot_layer = df["layer"].min()
    top_median = df.loc[df["layer"] == top_layer, "pert"].median()
    bot_median = df.loc[df["layer"] == bot_layer, "pert"].median()
    if bot_median > 0:
        layer_top_l0_ratio = top_median / bot_median
    else:
        layer_top_l0_ratio = np.inf if top_median > 0 else np.nan

    # A2: Within-layer cell + agreement
    df["cell_within"] = compute_within_layer_cell(df)
    agreement = (df["cell_global"] == df["cell_within"]).mean()

    # Per-cell Jaccard
    def jaccard(a: pd.Series, b: pd.Series) -> float:
        inter = (a & b).sum()
        union = (a | b).sum()
        return inter / union if union > 0 else np.nan

    jacs = {}
    for cell in ["A", "B", "C", "D"]:
        g_mask = df["cell_global"] == cell
        w_mask = df["cell_within"] == cell
        jacs[f"jaccard_{cell}"] = jaccard(g_mask, w_mask)

    # A3: Cell C layer distribution (concentration metrics)
    c_global = df[df["cell_global"] == "C"]
    if len(c_global) > 0:
        # fraction in top-2 layers
        top2_layers = sorted(df["layer"].unique())[-2:]
        c_top2_frac = c_global["layer"].isin(top2_layers).mean()
        # median layer of Cell C
        c_median_layer = c_global["layer"].median()
    else:
        c_top2_frac = np.nan
        c_median_layer = np.nan

    # A4: within-layer C/D gap (via per-head effect analog)
    effect = load_head_effect(unit, unit_type)
    cd_gap_global_approx = np.nan
    cd_gap_within_approx = np.nan
    if effect is not None:
        df = df.merge(
            effect.reset_index(name="effect"), on=["layer", "head"], how="left"
        )
        c_eff_global = df.loc[df["cell_global"] == "C", "effect"].mean()
        d_eff_global = df.loc[df["cell_global"] == "D", "effect"].mean()
        if d_eff_global and d_eff_global > 0:
            cd_gap_global_approx = c_eff_global / d_eff_global
        c_eff_within = df.loc[df["cell_within"] == "C", "effect"].mean()
        d_eff_within = df.loc[df["cell_within"] == "D", "effect"].mean()
        if d_eff_within and d_eff_within > 0:
            cd_gap_within_approx = c_eff_within / d_eff_within

    # Pattern flag
    if pd.isna(cd_gap_global_approx) or pd.isna(cd_gap_within_approx):
        rel_change = np.nan
    else:
        rel_change = abs(cd_gap_global_approx - cd_gap_within_approx) / cd_gap_global_approx

    def classify_pattern(r: float, rc: float) -> str:
        if pd.isna(r):
            return "?"
        if r < 0.3:
            if pd.isna(rc) or rc < 0.20:
                return "A"
            else:
                return "A (C/D shift large)"
        elif r < 0.6:
            if pd.isna(rc) or rc < 0.50:
                return "B"
            else:
                return "B (C/D shift large)"
        else:
            return "C"

    pattern = classify_pattern(rho, rel_change)

    # Save per-head dual classification
    per_head_df = df[
        ["layer", "head", "imp", "pert", "cell_global", "cell_within"]
        + (["effect"] if "effect" in df.columns else [])
    ].copy()
    per_head_df.to_csv(OUT_DIR / f"{unit}_dual_classification.csv", index=False)

    return {
        "unit": unit,
        "unit_type": unit_type,
        "n_layers": n_layers,
        "total_heads": n_heads,
        "rho_layer_pert": rho,
        "p_layer_pert": p_rho,
        "layer_top_l0_ratio": layer_top_l0_ratio,
        "global_within_agreement": agreement,
        **jacs,
        "cell_C_top2_frac_global": c_top2_frac,
        "cell_C_median_layer_global": c_median_layer,
        "cd_gap_global_approx": cd_gap_global_approx,
        "cd_gap_within_approx": cd_gap_within_approx,
        "cd_gap_rel_change": rel_change,
        "pattern_flag": pattern,
    }


def main() -> None:
    t_start = time.perf_counter()
    timings = {}

    results = []
    for m in NOVEL_WORD_PRIMARY:
        t0 = time.perf_counter()
        results.append(analyze_unit(m, "novel_word"))
        timings[m] = time.perf_counter() - t0
    for m in NOVEL_WORD_SUPPLEMENTARY:
        t0 = time.perf_counter()
        results.append(analyze_unit(m, "novel_word"))
        timings[m] = time.perf_counter() - t0
    for t in GPT2_TASKS:
        t0 = time.perf_counter()
        results.append(analyze_unit(t, "gpt2_task"))
        timings[t] = time.perf_counter() - t0

    df_out = pd.DataFrame(results)
    df_out.to_csv(OUT_CSV, index=False)

    total = time.perf_counter() - t_start

    # Build markdown report
    lines = [
        "# Layer-depth bias analysis ",
        "",
        "**Generated by**: `code/shared/compute_layer_perturbation_correlation.py`",
        f"**Scope**: {len(NOVEL_WORD_PRIMARY) + len(NOVEL_WORD_SUPPLEMENTARY)} novel-word models + {len(GPT2_TASKS)} GPT-2 tasks.",
        "Backs Appendix B §B.2.1 (Qwen layer–perturbation entanglement).",
        "",
        "---",
        "",
        "## A1. Spearman ρ(layer, perturbation_l2) and layer-depth concentration",
        "",
    ]
    lines.append(
        "| Unit | Type | Layers | Heads | ρ(layer, pert) | p | top/bot median ratio |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for r in results:
        lines.append(
            f"| {r['unit']} | {r['unit_type']} | {r['n_layers']} | {r['total_heads']} | "
            f"{r['rho_layer_pert']:.3f} | {r['p_layer_pert']:.2e} | "
            f"{r['layer_top_l0_ratio']:.2f} |"
        )

    lines.append("")
    lines.append("## A2. Global vs within-layer cell classification agreement")
    lines.append("")
    lines.append(
        "| Unit | Agreement | Jaccard A | Jaccard B | Jaccard C | Jaccard D |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in results:
        lines.append(
            f"| {r['unit']} | {r['global_within_agreement']:.3f} | "
            f"{r['jaccard_A']:.3f} | {r['jaccard_B']:.3f} | "
            f"{r['jaccard_C']:.3f} | {r['jaccard_D']:.3f} |"
        )

    lines.append("")
    lines.append("## A3. Cell C (global) layer distribution")
    lines.append("")
    lines.append(
        "| Unit | Cell C in top-2 layers (frac) | Cell C median layer |"
    )
    lines.append("|---|---:|---:|")
    for r in results:
        lines.append(
            f"| {r['unit']} | {r['cell_C_top2_frac_global']:.3f} | "
            f"{r['cell_C_median_layer_global']:.1f} |"
        )

    lines.append("")
    lines.append("## A4. Within-layer C/D gap approximation (via per-head effect analog)")
    lines.append("")
    lines.append(
        "| Unit | cd_gap global approx | cd_gap within approx | Rel change | Pattern |"
    )
    lines.append("|---|---:|---:|---:|---|")
    for r in results:
        glb = r["cd_gap_global_approx"]
        wit = r["cd_gap_within_approx"]
        rc = r["cd_gap_rel_change"]
        glb_s = f"{glb:.2f}" if not pd.isna(glb) else "—"
        wit_s = f"{wit:.2f}" if not pd.isna(wit) else "—"
        rc_s = f"{rc*100:.1f}%" if not pd.isna(rc) else "—"
        lines.append(
            f"| {r['unit']} | {glb_s} | {wit_s} | {rc_s} | {r['pattern_flag']} |"
        )

    lines.append("")
    lines.append("## Pattern distribution")
    lines.append("")
    pc = df_out["pattern_flag"].value_counts().to_dict()
    lines.append(f"Pattern counts across 12 units: {pc}")
    lines.append("")
    lines.append("Pattern legend:")
    lines.append("- **A**: ρ < 0.3 and within-layer C/D change <20%. Layer bias minor.")
    lines.append("- **B**: 0.3 ≤ ρ < 0.6 and within-layer C/D shrinks 20-50%.")
    lines.append("- **C**: ρ ≥ 0.6 and/or within-layer C/D shrinks >50%. Main claim impact.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Compute Time")
    lines.append("")
    lines.append("| Unit | Seconds |")
    lines.append("|---|---:|")
    for k, v in timings.items():
        lines.append(f"| {k} | {v:.3f} |")
    lines.append(f"| **Total** | **{total:.3f}** |")
    lines.append("")
    lines.append(
        "Platform: Python 3.10+ with pandas/numpy/scipy, CPU-only (no GPU). "
        "All operations vectorised (pandas groupby + numpy where)."
    )

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")
    print(f"Per-head dual classifications: {OUT_DIR}/")
    print(f"Total compute time: {total:.3f}s")


if __name__ == "__main__":
    main()

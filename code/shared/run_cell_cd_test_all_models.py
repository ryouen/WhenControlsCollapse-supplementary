"""
Cross-model Cell-C vs Cell-D patching effect comparison.

For each model with Phase 30 data, tests whether |delta_target_logit_signed|
differs between Cell-C (lo_imp, hi_pert) and Cell-D (lo_imp, lo_pert) heads,
using head-level means across trials.

Output: Mann-Whitney U, Cohen's d, rank-biserial r, permutation p-value.
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

SEED = 42
N_PERM = 10_000
BASE = Path("${WCC_ROOT}")

MODELS = [
    "llama-3.1-8b-instruct",
    "gpt-j-6b",
    "gemma-2-9b-it",
    "qwen-2.5-7b-instruct",
    "qwen-2.5-14b-instruct",
    "olmo-2-13b-instruct",
]

rows = []
for m in MODELS:
    patch_path = BASE / f"models/{m}/30_patching/individual_head_effects.csv"
    cell_path = BASE / f"models/{m}/20_scoring/cell_classification.csv"
    if not patch_path.exists() or not cell_path.exists():
        print(f"SKIP {m}: missing data")
        continue

    patch = pd.read_csv(patch_path)
    cells = pd.read_csv(cell_path)[["layer", "head", "cell"]]
    df = patch.merge(cells, on=["layer", "head"], how="left")
    df["abs_d"] = df["delta_target_logit_signed"].abs()

    hm = df.groupby(["layer", "head", "cell"])["abs_d"].mean().reset_index()
    c_vals = hm.loc[hm["cell"] == "C", "abs_d"].values
    d_vals = hm.loc[hm["cell"] == "D", "abs_d"].values

    nc, nd = len(c_vals), len(d_vals)
    u_stat, p_mw = stats.mannwhitneyu(c_vals, d_vals, alternative="two-sided")

    pooled_sd = np.sqrt(
        ((nc - 1) * c_vals.std(ddof=1) ** 2 + (nd - 1) * d_vals.std(ddof=1) ** 2)
        / (nc + nd - 2)
    )
    cohens_d = (c_vals.mean() - d_vals.mean()) / pooled_sd if pooled_sd > 0 else np.nan
    r_rb = 1 - 2 * u_stat / (nc * nd)

    rng = np.random.default_rng(SEED)
    combined = np.concatenate([c_vals, d_vals])
    obs_diff = c_vals.mean() - d_vals.mean()
    exceed = 0
    for _ in range(N_PERM):
        rng.shuffle(combined)
        if abs(combined[:nc].mean() - combined[nc:].mean()) >= abs(obs_diff):
            exceed += 1
    p_perm = exceed / N_PERM

    rows.append(
        dict(
            model=m, n_C=nc, n_D=nd,
            mean_C=c_vals.mean(), sd_C=c_vals.std(ddof=1),
            mean_D=d_vals.mean(), sd_D=d_vals.std(ddof=1),
            diff=obs_diff, cohen_d=cohens_d, r_rb=r_rb,
            U=u_stat, p_MW=p_mw, p_perm=p_perm,
        )
    )
    print(f"{m}: C={nc}, D={nd}, d={cohens_d:.2f}, p={p_mw:.2e}")

res = pd.DataFrame(rows)
print("\n" + "=" * 80)
print("| Model | n_C | n_D | Mean C | Mean D | Diff | Cohen d | r_rb | p (MW) | p (perm) |")
print("|-------|-----|-----|--------|--------|------|---------|------|--------|----------|")
for _, r in res.iterrows():
    print(
        f'| {r["model"]:25s} | {r["n_C"]:3.0f} | {r["n_D"]:3.0f} '
        f'| {r["mean_C"]:.4f} | {r["mean_D"]:.4f} | {r["diff"]:+.4f} '
        f'| {r["cohen_d"]:.2f} | {r["r_rb"]:.2f} | {r["p_MW"]:.2e} | {r["p_perm"]:.4f} |'
    )

"""
Matched-Head Analysis — Both Definitions
=========================================
Definition A: RSA quartile → hi-pert vs lo-pert (§3.1: pert predicts effect within imp bands)
Definition B: Perturbation quartile → hi-rsa vs lo-rsa (supplementary: rsa adds info within pert bands)

Output:
  cross_model/matched_head_def_a_rsa_quartile_hipert_vs_lopert.csv
  cross_model/matched_head_def_b_pert_quartile_hirsa_vs_lorsa.csv
"""

import pandas as pd
import numpy as np
import sys
from scipy.stats import mannwhitneyu

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

MODELS = [
    "gpt-j-6b-fp32",
    "llama-3.1-8b-instruct",
    "gemma-2-9b-it",
    "qwen-2.5-7b-instruct",
    "qwen-2.5-14b-instruct",
    "olmo-2-13b-instruct",
]

BASE = "${WCC_ROOT}"


def run_matched_head(hm, quartile_col, split_col, model_name):
    """Generic matched-head: split into 4 quartiles of quartile_col,
    within each quartile compare hi vs lo of split_col."""
    hm = hm.copy()
    hm["quartile"] = pd.qcut(hm[quartile_col], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    rows = []
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        qd = hm[hm["quartile"] == q]
        med = qd[split_col].median()
        hi = qd[qd[split_col] >= med]["abs_d"].values
        lo = qd[qd[split_col] < med]["abs_d"].values
        if len(hi) >= 2 and len(lo) >= 2:
            u, p = mannwhitneyu(hi, lo, alternative="two-sided")
            nc, nd = len(hi), len(lo)
            pooled = np.sqrt(
                ((nc - 1) * hi.std(ddof=1) ** 2 + (nd - 1) * lo.std(ddof=1) ** 2)
                / (nc + nd - 2)
            )
            d = (hi.mean() - lo.mean()) / pooled if pooled > 0 else 0
            rows.append({
                "model": model_name,
                "quartile": q,
                "n_hi": nc, "n_lo": nd,
                "mean_hi": round(hi.mean(), 6),
                "mean_lo": round(lo.mean(), 6),
                "diff": round(hi.mean() - lo.mean(), 6),
                "cohens_d": round(d, 4),
                "mannwhitney_p": p,
                "significant": p < 0.05,
            })
    return rows


results_a = []
results_b = []

for model_id in MODELS:
    print(f"Processing {model_id}...")
    cell = pd.read_csv(f"{BASE}/models/{model_id}/20_scoring/cell_classification.csv")
    ih = pd.read_csv(f"{BASE}/models/{model_id}/30_patching/individual_head_effects.csv")
    ih = ih.merge(cell[["layer", "head", "cell"]], on=["layer", "head"], how="left")
    ih["abs_d"] = ih["delta_target_logit_signed"].abs()
    hm = ih.groupby(["layer", "head", "cell"])["abs_d"].mean().reset_index()
    hm = hm.merge(
        cell[["layer", "head", "rsa_max", "perturbation_l2"]],
        on=["layer", "head"], how="left",
    )

    # Definition A: RSA quartile → hi-pert vs lo-pert
    results_a.extend(run_matched_head(hm, "rsa_max", "perturbation_l2", model_id))

    # Definition B: Perturbation quartile → hi-rsa vs lo-rsa
    results_b.extend(run_matched_head(hm, "perturbation_l2", "rsa_max", model_id))

df_a = pd.DataFrame(results_a)
df_b = pd.DataFrame(results_b)

out_a = f"{BASE}/cross_model/matched_head_def_a_rsa_quartile_hipert_vs_lopert.csv"
out_b = f"{BASE}/cross_model/matched_head_def_b_pert_quartile_hirsa_vs_lorsa.csv"
df_a.to_csv(out_a, index=False)
df_b.to_csv(out_b, index=False)

print(f"\nSaved: {out_a} ({len(df_a)} rows)")
print(f"Saved: {out_b} ({len(df_b)} rows)")

# Summary
print("\n=== Definition A: RSA quartile → hi-pert vs lo-pert ===")
print(f"{'Model':30s} {'Q1':>8s} {'Q2':>8s} {'Q3':>8s} {'Q4':>8s} {'All sig?':>8s}")
for m in MODELS:
    sub = df_a[df_a["model"] == m]
    sigs = []
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        row = sub[sub["quartile"] == q]
        if len(row):
            p = row.iloc[0]["mannwhitney_p"]
            s = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            sigs.append(s)
        else:
            sigs.append("?")
    all_sig = "YES" if all(s != "ns" and s != "?" for s in sigs) else "NO"
    print(f"  {m:30s} {sigs[0]:>8s} {sigs[1]:>8s} {sigs[2]:>8s} {sigs[3]:>8s} {all_sig:>8s}")

print("\n=== Definition B: Pert quartile → hi-rsa vs lo-rsa ===")
print(f"{'Model':30s} {'Q1':>8s} {'Q2':>8s} {'Q3':>8s} {'Q4':>8s} {'All sig?':>8s}")
for m in MODELS:
    sub = df_b[df_b["model"] == m]
    sigs = []
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        row = sub[sub["quartile"] == q]
        if len(row):
            p = row.iloc[0]["mannwhitney_p"]
            s = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            sigs.append(s)
        else:
            sigs.append("?")
    all_sig = "YES" if all(s != "ns" and s != "?" for s in sigs) else "NO"
    print(f"  {m:30s} {sigs[0]:>8s} {sigs[1]:>8s} {sigs[2]:>8s} {sigs[3]:>8s} {all_sig:>8s}")

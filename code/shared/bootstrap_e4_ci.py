"""
Reproduce Appendix E.4 bootstrap mean-difference CI, Cohen's d, and Cliff's
delta CI for the IOI Cell-C-ex-NNM vs Cell-D head-effect comparison.

Reproduces the headline values:
  Mean difference (C ex-NNM - D) = +0.059 logit units
  95% bootstrap CI = [-0.001, 0.143]
  Cohen's d (pooled SD) ~ 0.48 (body §4.2)
  Cliff's delta = +0.257
  95% bootstrap CI = [-0.024, 0.519]

Two-sample bootstrap (the C-ex-NNM and D head sets are disjoint, so this is
not a paired-sample design): resample each set independently with replacement
and recompute the statistic. 10,000 resamples; np.random.seed(42).

Cohen's d uses the pooled-SD formula:
  d = (mean(C) - mean(D)) / sqrt(((n_C - 1)*var(C, ddof=1) +
                                  (n_D - 1)*var(D, ddof=1)) / (n_C + n_D - 2))

Usage:
    export WCC_ROOT=/path/to/unzipped/supplementary
    python supplementary/code/shared/bootstrap_e4_ci.py
"""
import os
import numpy as np
import pandas as pd

WCC_ROOT = os.environ.get('WCC_ROOT', '..')
DATA = os.path.join(WCC_ROOT, 'data')

# Load IOI signed cell classification + absolute-importance per-head
cc_signed = pd.read_csv(os.path.join(DATA, 'ioi', '20_scoring', 'cell_classification_signed_ld.csv'))
cc_abs = pd.read_csv(os.path.join(DATA, 'ioi', '20_scoring', 'cell_classification_abs.csv'))

# NNM heads (Wang et al. 2022 IOI ground-truth)
nnm_set = {(10, 7), (11, 10)}

# importance_signed = |head_effect| from cc_abs, mapped by (layer, head)
imp_map = {(int(r['layer']), int(r['head'])): float(r['importance_signed'])
           for _, r in cc_abs.iterrows()}

C_heads = cc_signed[cc_signed.cell == 'C'][['layer', 'head']]
D_heads = cc_signed[cc_signed.cell == 'D'][['layer', 'head']]

C = np.array([abs(imp_map.get((int(r['layer']), int(r['head'])), 0))
              for _, r in C_heads.iterrows()
              if (int(r['layer']), int(r['head'])) not in nnm_set])
D = np.array([abs(imp_map.get((int(r['layer']), int(r['head'])), 0))
              for _, r in D_heads.iterrows()])

print(f'n_C_ex_NNM = {len(C)}, n_D = {len(D)}')
print(f'mean C = {C.mean():.4f}, mean D = {D.mean():.4f}')
print(f'mean diff = {C.mean() - D.mean():.4f}')

# Bootstrap mean difference
np.random.seed(42)
n_boot = 10_000
mean_diffs = np.empty(n_boot)
for i in range(n_boot):
    cb = np.random.choice(C, size=len(C), replace=True)
    db = np.random.choice(D, size=len(D), replace=True)
    mean_diffs[i] = cb.mean() - db.mean()
ci_low, ci_high = np.percentile(mean_diffs, [2.5, 97.5])
print(f'Bootstrap mean-diff 95% CI = [{ci_low:.4f}, {ci_high:.4f}]')


def cohens_d(x, y):
    x = np.asarray(x); y = np.asarray(y)
    nx, ny = len(x), len(y)
    pooled_var = ((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2)
    return (x.mean() - y.mean()) / np.sqrt(pooled_var)


d_cohen = cohens_d(C, D)
print(f"Cohen's d (pooled SD) = {d_cohen:.4f}")

np.random.seed(42)
ds = np.empty(n_boot)
for i in range(n_boot):
    cb = np.random.choice(C, size=len(C), replace=True)
    db = np.random.choice(D, size=len(D), replace=True)
    ds[i] = cohens_d(cb, db)
d_cohen_low, d_cohen_high = np.percentile(ds, [2.5, 97.5])
print(f"Cohen's d 95% CI = [{d_cohen_low:.4f}, {d_cohen_high:.4f}]")


def cliffs_delta(x, y):
    x = np.asarray(x); y = np.asarray(y)
    g = sum(np.sum(xi > y) for xi in x)
    l = sum(np.sum(xi < y) for xi in x)
    return (g - l) / (len(x) * len(y))


delta = cliffs_delta(C, D)
print(f"Cliff's delta = {delta:.4f}")

np.random.seed(42)
deltas = np.empty(n_boot)
for i in range(n_boot):
    cb = np.random.choice(C, size=len(C), replace=True)
    db = np.random.choice(D, size=len(D), replace=True)
    deltas[i] = cliffs_delta(cb, db)
d_low, d_high = np.percentile(deltas, [2.5, 97.5])
print(f"Cliff's delta 95% CI = [{d_low:.4f}, {d_high:.4f}]")

# Optional: write a small summary CSV
out_dir = os.path.join(DATA, 'ioi', '30_patching')
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, 'e4_bootstrap_ci.csv')
pd.DataFrame([{
    'n_C_ex_NNM': len(C), 'n_D': len(D),
    'mean_C_ex_NNM': round(C.mean(), 4), 'mean_D': round(D.mean(), 4),
    'mean_diff': round(C.mean() - D.mean(), 4),
    'mean_diff_ci_low': round(ci_low, 4), 'mean_diff_ci_high': round(ci_high, 4),
    'cohens_d': round(d_cohen, 4),
    'cohens_d_ci_low': round(d_cohen_low, 4), 'cohens_d_ci_high': round(d_cohen_high, 4),
    'cliffs_delta': round(delta, 4),
    'cliffs_delta_ci_low': round(d_low, 4), 'cliffs_delta_ci_high': round(d_high, 4),
    'n_boot': n_boot, 'random_seed': 42,
}]).to_csv(out, index=False)
print(f'\nWrote {out}')

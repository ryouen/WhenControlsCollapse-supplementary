"""
Per-model joint OLS regression of |Delta target logit| ~ rsa_max + perturbation_L2 + layer_index
(z-scored) for the 8-model novel-word battery. Backs Appendix B.3 / Table B4.

Output: per-model data/novel_word/{m}/20_scoring/perhead_regression.csv with columns
  model, n_heads, beta_imp, beta_pert, beta_layer,
  p_imp, p_pert, p_layer,
  R2_full, R2_imp_only, R2_pert_only, R2_layer_only, partial_R2_pert.

Usage:
    export WCC_ROOT=/path/to/unzipped/supplementary
    python supplementary/code/shared/compute_perhead_regression.py
"""
import os
import pandas as pd
import statsmodels.api as sm

WCC_ROOT = os.environ.get('WCC_ROOT', '..')
DATA = os.path.join(WCC_ROOT, 'data')

models = [
    'llama-3.1-8b-instruct', 'gemma-2-9b-it', 'qwen-2.5-7b-instruct',
    'qwen-2.5-14b-instruct', 'olmo-2-13b-instruct', 'gpt-j-6b-fp32',
    'gemma-3-27b-it', 'llama-3.1-70b-instruct-4bit',
]

def zs(x):
    return (x - x.mean()) / x.std() if x.std() > 0 else x * 0

for m in models:
    base = os.path.join(DATA, 'novel_word', m)
    rsa = pd.read_csv(os.path.join(base, '20_scoring', 'rsa_per_head.csv'))
    pert = pd.read_csv(os.path.join(base, '20_scoring', 'perturbation_per_head.csv'))
    ihe = pd.read_csv(os.path.join(base, '30_patching', 'individual_head_effects.csv'))

    # Find the signed-delta column
    delta_cols = [c for c in ihe.columns if 'delta' in c.lower() and 'target' in c.lower() and 'logit' in c.lower()]
    if not delta_cols:
        print(f'  {m}: SKIP - no delta column'); continue
    ihe['abs_delta'] = ihe[delta_cols[0]].abs()
    per_head = ihe.groupby(['layer', 'head'])['abs_delta'].mean().reset_index()
    per_head.columns = ['layer', 'head', 'mean_abs_delta']

    df = rsa.merge(pert, on=['layer', 'head']).merge(per_head, on=['layer', 'head'])
    df = df.dropna(subset=['rsa_max', 'perturbation_L2', 'mean_abs_delta'])
    if len(df) < 4:
        continue

    df['z_imp'] = zs(df['rsa_max'])
    df['z_pert'] = zs(df['perturbation_L2'])
    df['z_layer'] = zs(df['layer'])
    df['z_y'] = zs(df['mean_abs_delta'])

    X = sm.add_constant(df[['z_imp', 'z_pert', 'z_layer']])
    y = df['z_y']
    fit = sm.OLS(y, X).fit()
    fit_no_pert = sm.OLS(y, sm.add_constant(df[['z_imp', 'z_layer']])).fit()
    partial_R2_pert = (fit_no_pert.ssr - fit.ssr) / fit_no_pert.ssr

    out_row = pd.DataFrame([{
        'model': m, 'n_heads': len(df),
        'beta_imp': round(fit.params['z_imp'], 4),
        'beta_pert': round(fit.params['z_pert'], 4),
        'beta_layer': round(fit.params['z_layer'], 4),
        'p_imp': fit.pvalues['z_imp'],
        'p_pert': fit.pvalues['z_pert'],
        'p_layer': fit.pvalues['z_layer'],
        'R2_full': round(fit.rsquared, 4),
        'R2_imp_only': round(sm.OLS(y, sm.add_constant(df[['z_imp']])).fit().rsquared, 4),
        'R2_pert_only': round(sm.OLS(y, sm.add_constant(df[['z_pert']])).fit().rsquared, 4),
        'R2_layer_only': round(sm.OLS(y, sm.add_constant(df[['z_layer']])).fit().rsquared, 4),
        'partial_R2_pert': round(partial_R2_pert, 4),
    }])
    out = os.path.join(base, '20_scoring', 'perhead_regression.csv')
    out_row.to_csv(out, index=False)
    print(f'  {m}: partial_R2_pert={partial_R2_pert:.4f} -> {out}')

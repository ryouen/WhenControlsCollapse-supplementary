"""
Per-model per-crel-pair C-vs-D breakdown at the diagnostic ratio (r = 0.10).
Cited from Appendix B §B.5 ("Per-crel trial counts ship in the supplementary CSV
models/{model}/20_scoring/behavioral_crel_breakdown.csv").

Output: `data/novel_word/{m}/20_scoring/behavioral_crel_breakdown.csv` per model
        with columns:
  crel_pair, n_C_trials, n_D_trials, c_mean_abs_delta, d_mean_abs_delta, c_minus_d

The 6 crel_pair categories enumerate the unordered pairings of {SAME, OPP, MORE, LESS}.
Each (crel_pair, group) row averages over the 10 pair_ids x 2 attr_dims = 20 tuples
that the crel_pair contributes to dose_response_v2.csv at r = 0.10 for that group.

This producer mirrors the historic audit script `code/audit/v763/compute_per_crel.py`
that originally produced the figures cited in Appendix B Table B10 and the §B.5
"per-crel breakdown" paragraph.

Usage:
    export WCC_ROOT=/path/to/unzipped/supplementary
    python supplementary/code/shared/compute_behavioral_crel_breakdown.py
"""
import os
import pandas as pd

WCC_ROOT = os.environ.get('WCC_ROOT', '..')
DATA = os.path.join(WCC_ROOT, 'data')

models = [
    'llama-3.1-8b-instruct', 'gemma-2-9b-it', 'qwen-2.5-7b-instruct',
    'qwen-2.5-14b-instruct', 'olmo-2-13b-instruct', 'gpt-j-6b-fp32',
    'gemma-3-27b-it', 'llama-3.1-70b-instruct-4bit',
]

CREL_PAIRS = ['SAME-OPP', 'SAME-MORE', 'SAME-LESS', 'OPP-MORE', 'OPP-LESS', 'MORE-LESS']

print(f'{"model":<32} {"crel_pair":>10} {"n_C":>5} {"n_D":>5} {"c_mean":>8} {"d_mean":>8} {"c_minus_d":>10}')
print('-' * 85)

for m in models:
    p = os.path.join(DATA, 'novel_word', m, '30_patching', 'dose_response_v2.csv')
    df = pd.read_csv(p)
    sub = df[df['ratio'] == 0.10]

    rows = []
    for cp in CREL_PAIRS:
        c_rows = sub[(sub['group'] == 'C_imp_asc') & (sub['crel_pair'] == cp)]
        d_rows = sub[(sub['group'] == 'D_imp_asc') & (sub['crel_pair'] == cp)]
        c_mean = c_rows['delta_target_logit_signed'].abs().mean()
        d_mean = d_rows['delta_target_logit_signed'].abs().mean()
        rows.append({
            'crel_pair': cp,
            'n_C_trials': int(len(c_rows)),
            'n_D_trials': int(len(d_rows)),
            'c_mean_abs_delta': round(float(c_mean), 4),
            'd_mean_abs_delta': round(float(d_mean), 4),
            'c_minus_d': round(float(c_mean - d_mean), 4),
        })
        print(f'{m:<32} {cp:>10} {len(c_rows):>5} {len(d_rows):>5} '
              f'{c_mean:>8.4f} {d_mean:>8.4f} {c_mean - d_mean:>10.4f}')

    out_dir = os.path.join(DATA, 'novel_word', m, '20_scoring')
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, 'behavioral_crel_breakdown.csv')
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f'  -> {out}')

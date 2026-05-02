"""
Compute per-model saturation onsets r_sat = min(|C|, |D|) / N for the 8-model battery.
Output: data/cross_model_per_model/saturation_onsets.csv (Appendix G.3 / Table G2).

Usage:
    export WCC_ROOT=/path/to/unzipped/supplementary
    python supplementary/code/shared/compute_saturation_onsets.py
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

rows = []
for m in models:
    cls = pd.read_csv(os.path.join(DATA, 'novel_word', m, '20_scoring', 'cell_classification.csv'))
    n_total = len(cls)
    counts = cls['cell'].value_counts()
    n_C = int(counts.get('C', 0)); n_D = int(counts.get('D', 0))
    n_A = int(counts.get('hihp', 0)); n_B = int(counts.get('hilp', 0))
    rows.append({
        'model': m, 'n_total_heads': n_total,
        'n_A_hihp': n_A, 'n_B_hilp': n_B, 'n_C': n_C, 'n_D': n_D,
        'r_sat': round(min(n_C, n_D) / n_total, 4),
    })

df = pd.DataFrame(rows)
out = os.path.join(DATA, 'cross_model_per_model', 'saturation_onsets.csv')
df.to_csv(out, index=False)
print(df.to_string(index=False))
print(f'\nWrote {out}')
print(f'r_sat range: {df["r_sat"].min():.3f} - {df["r_sat"].max():.3f}, mean={df["r_sat"].mean():.3f}')

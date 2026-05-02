"""
Aggregate per-model factual-recall dose-response into a single CSV cited from Appendix C.3.

Concatenates data/factual_recall/{m}/30_patching/behavioral_gamma_summary.csv across the
8-model battery, prepending the 'model' column. Output:
  data/factual_recall/cross_model/fr_dose_response_per_model.csv

Usage:
    export WCC_ROOT=/path/to/unzipped/supplementary
    python supplementary/code/shared/compute_fr_dose_response_per_model.py
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

dfs = []
for m in models:
    p = os.path.join(DATA, 'factual_recall', m, '30_patching', 'behavioral_gamma_summary.csv')
    if not os.path.exists(p):
        print(f'  SKIP: {m}')
        continue
    sub = pd.read_csv(p)
    sub.insert(0, 'model', m)
    dfs.append(sub)

df = pd.concat(dfs, ignore_index=True)
out_dir = os.path.join(DATA, 'factual_recall', 'cross_model')
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, 'fr_dose_response_per_model.csv')
df.to_csv(out, index=False)
print(f'Wrote {out} ({len(df)} rows, {len(df.columns)} cols)')

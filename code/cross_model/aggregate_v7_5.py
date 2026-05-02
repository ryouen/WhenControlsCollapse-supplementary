#!/usr/bin/env python3
"""Cross-model aggregator for v7.5.0 (Main) + v7.5.5 (Prospect).

Produces paper-ready tables under ``$WCC_ROOT/data/cross_model_per_model/``:

  1. ``cell_counts_per_model.csv``            — from 20_scoring/cell_classification.csv
  2. ``main_dose_response_gap_per_model.csv`` — from 30_patching/dose_response_v2.csv
  3. ``prospect_gap_per_model.csv``           — from 50_prospect/prospect_dose_response_v2.csv
  4. ``behavioral_flip_rates_per_model.csv``  — from 38_behavioral/flip_rates.csv

Inputs: ``$WCC_ROOT/data/novel_word/{model}/{phase_dir}/...``
Outputs: ``$WCC_ROOT/data/cross_model_per_model/*.csv``

Usage:
    export WCC_ROOT=/path/to/unzipped/supplementary
    python supplementary/code/cross_model/aggregate_v7_5.py

The internal research version of this script reads/writes Dropbox via private
helpers; this distribution version reads/writes the local filesystem rooted at
``$WCC_ROOT`` (matches the convention documented in ``supplementary/README.md``).
"""
import io
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WCC_ROOT = Path(os.environ.get('WCC_ROOT', '.')).resolve()
DATA_BASE = WCC_ROOT / 'data' / 'novel_word'
OUT_BASE = WCC_ROOT / 'data' / 'cross_model_per_model'

MODELS = [
    'llama-3.1-8b-instruct',
    'qwen-2.5-7b-instruct',
    'qwen-2.5-14b-instruct',
    'gemma-2-9b-it',
    'gemma-3-27b-it',
    'olmo-2-13b-instruct',
    'gpt-j-6b-fp32',
]
KEY_RATIOS = [0.05, 0.10, 0.20, 0.50]
ORDERINGS_8 = ['hihp_imp_desc', 'hihp_imp_asc', 'hilp_imp_desc', 'hilp_imp_asc',
               'C_imp_asc', 'D_imp_asc', 'diag_rank_asc', 'std_imp_asc']


def safe_read(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def table_cell_counts():
    """Per-model cell counts (hihp, hilp, C, D) from 20_scoring/cell_classification.csv."""
    rows = []
    for model in MODELS:
        df = safe_read(DATA_BASE / model / '20_scoring' / 'cell_classification.csv')
        if df is None or 'cell' not in df.columns:
            rows.append({'model': model, 'n_hihp': None, 'n_hilp': None, 'n_C': None, 'n_D': None, 'n_total_heads': None})
            continue
        cnts = df['cell'].value_counts().to_dict()
        rows.append({
            'model': model,
            'n_hihp': int(cnts.get('hihp', 0)),
            'n_hilp': int(cnts.get('hilp', 0)),
            'n_C': int(cnts.get('C', 0)),
            'n_D': int(cnts.get('D', 0)),
            'n_total_heads': int(len(df)),
        })
    return pd.DataFrame(rows)


def table_main_gap():
    """Per-model gap_CD and gap_hihp_hilp at key ratios, from 30_patching/dose_response_v2.csv.

    Main gap = mean(|signed_delta|) over (pair_id, attr_dim, crel_pair) per (model, group, ratio).
      gap_CD       = mean|Δ|_C_imp_asc       − mean|Δ|_D_imp_asc
      gap_hihp_hilp= mean|Δ|_hihp_imp_desc   − mean|Δ|_hilp_imp_desc
    """
    rows = []
    for model in MODELS:
        df = safe_read(DATA_BASE / model / '30_patching' / 'dose_response_v2.csv')
        if df is None:
            continue
        val_col = 'delta_target_logit_signed' if 'delta_target_logit_signed' in df.columns else 'mean_abs_delta'
        df['abs_val'] = df[val_col].abs()
        for r in KEY_RATIOS:
            sub = df[np.isclose(df['ratio'], r, atol=0.005)]
            agg = sub.groupby('group')['abs_val'].mean().to_dict()
            rows.append({
                'model': model,
                'ratio': r,
                'mean_abs_C_imp_asc': agg.get('C_imp_asc'),
                'mean_abs_D_imp_asc': agg.get('D_imp_asc'),
                'gap_CD': (agg.get('C_imp_asc', np.nan) - agg.get('D_imp_asc', np.nan)) if agg.get('D_imp_asc') is not None else None,
                'mean_abs_hihp_imp_desc': agg.get('hihp_imp_desc'),
                'mean_abs_hilp_imp_desc': agg.get('hilp_imp_desc'),
                'gap_hihp_hilp': (agg.get('hihp_imp_desc', np.nan) - agg.get('hilp_imp_desc', np.nan)) if agg.get('hilp_imp_desc') is not None else None,
                'mean_abs_std_imp_asc': agg.get('std_imp_asc'),
                'mean_abs_diag_rank_asc': agg.get('diag_rank_asc'),
                'mean_abs_D_pert_asc': agg.get('D_pert_asc'),
            })
    return pd.DataFrame(rows)


def table_prospect_gap():
    """Per-model Prospect gap summary: mean, SE, sign-consistency across 10 seeds.

    Prospect value col = ``mean_abs_delta`` (pre-computed per seed).
    """
    rows = []
    for model in MODELS:
        df = safe_read(DATA_BASE / model / '50_prospect' / 'prospect_dose_response_v2.csv')
        if df is None:
            continue
        # Limit to canonical seeds 43-52 (exclude Llama's extended 53-72)
        df = df[df['seed'].isin(range(43, 53))]
        for r in KEY_RATIOS:
            sub = df[np.isclose(df['ratio'], r, atol=0.005)]
            pivot = sub.pivot_table(index='seed', columns='group', values='mean_abs_delta', aggfunc='first')
            if 'C_imp_asc' in pivot.columns and 'D_imp_asc' in pivot.columns:
                gap_cd = pivot['C_imp_asc'].abs() - pivot['D_imp_asc'].abs()
                cd_n = int(np.isfinite(gap_cd).sum())
                cd_mean = float(np.nanmean(gap_cd)) if cd_n else np.nan
                cd_se = float(np.nanstd(gap_cd, ddof=1) / np.sqrt(cd_n)) if cd_n > 1 else np.nan
                cd_positive = int((gap_cd > 0).sum())
            else:
                cd_mean = cd_se = np.nan
                cd_positive = cd_n = 0
            if 'hihp_imp_desc' in pivot.columns and 'hilp_imp_desc' in pivot.columns:
                gap_hh = pivot['hihp_imp_desc'].abs() - pivot['hilp_imp_desc'].abs()
                hh_n = int(np.isfinite(gap_hh).sum())
                hh_mean = float(np.nanmean(gap_hh)) if hh_n else np.nan
                hh_se = float(np.nanstd(gap_hh, ddof=1) / np.sqrt(hh_n)) if hh_n > 1 else np.nan
                hh_positive = int((gap_hh > 0).sum())
            else:
                hh_mean = hh_se = np.nan
                hh_positive = hh_n = 0
            rows.append({
                'model': model, 'ratio': r,
                'gap_CD_mean': cd_mean, 'gap_CD_se': cd_se,
                'gap_CD_positive_n_seeds': cd_positive, 'gap_CD_total_seeds': cd_n,
                'gap_hihp_hilp_mean': hh_mean, 'gap_hihp_hilp_se': hh_se,
                'gap_hihp_hilp_positive_n_seeds': hh_positive, 'gap_hihp_hilp_total_seeds': hh_n,
            })
    return pd.DataFrame(rows)


def table_behavioral():
    """Per-model flip rates from 38_behavioral/flip_rates.csv."""
    parts = []
    for model in MODELS:
        df = safe_read(DATA_BASE / model / '38_behavioral' / 'flip_rates.csv')
        if df is None:
            continue
        df['model'] = model
        parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _save(df: pd.DataFrame, name: str) -> None:
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    out = OUT_BASE / name
    df.to_csv(out, index=False)
    print(f'  → {out}  ({out.stat().st_size:,} B, {len(df)} rows)')


if __name__ == '__main__':
    if not DATA_BASE.exists():
        sys.stderr.write(
            f'ERROR: {DATA_BASE} does not exist.\n'
            f'Set WCC_ROOT to the directory containing data/novel_word/{{model}}/...\n'
            f'(See supplementary/README.md for layout.)\n'
        )
        sys.exit(1)

    print('=== cell_counts_per_model ===')
    t1 = table_cell_counts()
    print(t1.to_string(index=False))
    _save(t1, 'cell_counts_per_model.csv')

    print('\n=== main_dose_response_gap_per_model ===')
    t2 = table_main_gap()
    print(t2.round(4).to_string(index=False))
    _save(t2, 'main_dose_response_gap_per_model.csv')

    print('\n=== prospect_gap_per_model (seeds 43-52) ===')
    t3 = table_prospect_gap()
    print(t3.round(4).to_string(index=False))
    _save(t3, 'prospect_gap_per_model.csv')

    print(f'\n=== behavioral_flip_rates_per_model ===')
    t4 = table_behavioral()
    if len(t4):
        print(f'  ({len(t4)} rows across {t4["model"].nunique()} models)')
        _save(t4, 'behavioral_flip_rates_per_model.csv')
    else:
        print('  (no data)')

    print(f'\nAll outputs written under {OUT_BASE}')

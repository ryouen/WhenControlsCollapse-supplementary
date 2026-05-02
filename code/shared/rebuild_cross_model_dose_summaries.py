#!/usr/bin/env python3
"""
rebuild_cross_model_dose_summaries.py
=====================================
Rebuild `cross_model/dose_response_v2_summary.csv` and
`cross_model/diagonal_dose_response_novel_word{,_summary}.csv` from the
current (BPE-fix) `models/*/30_patching/dose_response_v2.csv`.

GPU-free. Reads only the per-model v2 CSVs.

Usage:
    python code/shared/rebuild_cross_model_dose_summaries.py                   # dry-run
    python code/shared/rebuild_cross_model_dose_summaries.py --apply           # write

Outputs (when --apply):
  cross_model/dose_response_v2_summary.csv          9 orderings × model table at ratio=0.10
  cross_model/diagonal_dose_response_novel_word.csv  long-format diag_* rows
  cross_model/diagonal_dose_response_novel_word_summary.csv  per-model diag summary
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
OUT_DIR = PROJECT_ROOT / "cross_model"

MODEL_ORDER = [
    "gpt-j-6b-fp32",
    "gemma-3-27b-it",
    "llama-3.1-8b-instruct",
    "llama-3.1-8b",
    "llama-3.1-70b-instruct-4bit",
    "gemma-2-9b-it",
    "qwen-2.5-7b-instruct",
    "qwen-2.5-14b-instruct",
    "olmo-2-13b-instruct",
]
V2_GROUPS = [
    "C_imp_asc", "C_pert_asc", "C_pert_desc",
    "D_imp_asc", "D_pert_asc", "D_pert_desc",
    "std_imp_asc", "diag_zscore_asc", "diag_rank_asc",
]
DIAG_GROUPS = ["diag_zscore_asc", "diag_rank_asc"]


def per_model_v2(mid: str):
    fp = MODELS_DIR / mid / "30_patching" / "dose_response_v2.csv"
    if not fp.exists():
        return None
    return pd.read_csv(fp)


def build_summary_at_ratio(ratio: float) -> pd.DataFrame:
    rows = []
    for mid in MODEL_ORDER:
        df = per_model_v2(mid)
        if df is None or not {"group", "ratio", "delta_target_logit_signed"} <= set(df.columns):
            continue
        df = df[np.isclose(df["ratio"], ratio)].copy()
        df["abs_delta"] = df["delta_target_logit_signed"].abs()
        row = {"model": mid}
        for g in V2_GROUPS:
            sub = df[df["group"] == g]
            row[g] = round(float(sub["abs_delta"].mean()), 6) if len(sub) else None
        rows.append(row)
    cols = ["model"] + V2_GROUPS
    return pd.DataFrame(rows, columns=cols)


def build_diagonal_long() -> pd.DataFrame:
    frames = []
    for mid in MODEL_ORDER:
        df = per_model_v2(mid)
        if df is None:
            continue
        sub = df[df["group"].isin(DIAG_GROUPS)].copy()
        if len(sub):
            sub.insert(0, "model", mid) if "model" not in sub.columns else None
            frames.append(sub)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_diagonal_summary(ratios=(0.05, 0.10, 0.15, 0.20)) -> pd.DataFrame:
    rows = []
    for mid in MODEL_ORDER:
        df = per_model_v2(mid)
        if df is None:
            continue
        for g in DIAG_GROUPS:
            for r in ratios:
                s = df[(df["group"] == g) & np.isclose(df["ratio"], r)]
                if not len(s):
                    continue
                s_abs = s["delta_target_logit_signed"].abs()
                rows.append({
                    "model": mid,
                    "group": g,
                    "ratio": round(r, 4),
                    "n_trials": int(len(s)),
                    "mean_abs_delta": round(float(s_abs.mean()), 6),
                    "mean_signed_delta": round(float(s["delta_target_logit_signed"].mean()), 6),
                    "median_abs_delta": round(float(s_abs.median()), 6),
                })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--ratio", type=float, default=0.10)
    args = ap.parse_args()

    summary = build_summary_at_ratio(args.ratio)
    diag_long = build_diagonal_long()
    diag_sum = build_diagonal_summary()

    print(f"=== dose_response_v2_summary (@ ratio={args.ratio}) ===")
    print(summary.to_string(index=False))
    print()
    print(f"=== diagonal_dose_response_novel_word (long) — {len(diag_long)} rows ===")
    if len(diag_long):
        print(diag_long.head(5).to_string(index=False))
    print()
    print(f"=== diagonal_dose_response_novel_word_summary — {len(diag_sum)} rows ===")
    print(diag_sum.to_string(index=False))

    if args.apply:
        OUT_DIR.mkdir(exist_ok=True)
        bak_suffix = ".bak.20260420"
        def _write(df: pd.DataFrame, name: str):
            out = OUT_DIR / name
            if out.exists():
                bak = out.with_suffix(out.suffix + bak_suffix)
                if not bak.exists():
                    bak.write_bytes(out.read_bytes())
            df.to_csv(out, index=False)
            print(f"  wrote {out.relative_to(PROJECT_ROOT)} ({len(df)} rows)")

        _write(summary, "dose_response_v2_summary.csv")
        _write(diag_long, "diagonal_dose_response_novel_word.csv")
        _write(diag_sum, "diagonal_dose_response_novel_word_summary.csv")
        print("done.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Greater-Than Full Pipeline -- Phase 10, 20, 30, 35, 90
GPT-2 Small, TransformerLens.

Paper role: "Null-difference control" -- confound ABSENT in localized circuit.
Based on: neural_exam_project/.../run_greater_than_2x2.py

Key numbers to validate:
    rho(importance, perturbation) = +0.1747
    C/D @ 10% (raw_l2, resample) = 3.02x
    AUROC vs GT = 0.4295 (BELOW chance)
    All 5 GT heads land in Cell A
"""

import csv
import json
import math
import os
import random
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, mannwhitneyu

from transformer_lens import HookedTransformer, utils

# --- Paths ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
TASK_ROOT = os.path.join(PROJECT_ROOT, "tasks", "greater_than")

# --- Constants ---
N_PROMPTS = 200
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RATIOS_DOSE = [0.01, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.20, 0.30, 0.50]

# Hanna et al. (2023) GT circuit -- 5 attention heads
GT_CIRCUIT = [
    (7, 10, "GT"), (7, 11, "GT"), (8, 10, "GT"), (8, 11, "GT"), (9, 1, "GT"),
]
GT_CIRCUIT_HEADS = {(l, h) for l, h, _ in GT_CIRCUIT}
GT_CIRCUIT_MAP = {(l, h): sc for l, h, sc in GT_CIRCUIT}
N_GT_HEADS = len(GT_CIRCUIT_HEADS)

TEMPLATES = [
    "The war lasted from the year 17{} to the year 17",
    "The meeting was scheduled from the year 17{} to the year 17",
    "The dynasty ruled from the year 17{} to the year 17",
    "The construction took from the year 17{} to the year 17",
    "The expedition ran from the year 17{} to the year 17",
]


# --- Helpers ---
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def write_csv(path, rows, fieldnames=None):
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

def save_config(output_dir, **fields):
    cfg = {"completed_at": datetime.now(timezone.utc).isoformat(),
           "platform": "windows_rtx3060", "model_id": "gpt2", **fields}
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, default=str)


# --- GT Data ---
def generate_gt_data(n=200, seed=42):
    rng = random.Random(seed)
    prompts, start_years = [], []
    for _ in range(n):
        t = rng.choice(TEMPLATES)
        yy = rng.randint(2, 89)
        prompts.append(t.format(f"{yy:02d}"))
        start_years.append(yy)
    return prompts, start_years

def generate_corrupted(prompts, start_years):
    corrupted = []
    for prompt, yy in zip(prompts, start_years):
        corrupted.append(prompt.replace(f"17{yy:02d}", "1701"))
    return corrupted

def compute_gt_score(logits, model, start_years, last_pos):
    """GT score: mean(P(year>start) - P(year<=start)) over prompts."""
    N = len(start_years)
    scores = []
    for i in range(N):
        yy = start_years[i]
        probs = torch.softmax(logits[i, last_pos, :], dim=-1)
        greater, lesser = 0.0, 0.0
        for yr in range(100):
            tok = model.to_single_token(f"{yr:02d}")
            if tok is not None:
                p = probs[tok].item()
                if yr > yy:
                    greater += p
                else:
                    lesser += p
        scores.append(greater - lesser)
    return scores  # per-prompt list

def compute_gt_score_mean(logits, model, start_years, last_pos):
    return float(np.mean(compute_gt_score(logits, model, start_years, last_pos)))


# --- CD ratio with Def A + B ---
def compute_cd_summary(dose_df, importance_metrics, ablation_type="resample"):
    rows = []
    for imp_name in importance_metrics:
        sub = dose_df[dose_df["importance_metric"] == imp_name]
        for ratio in sorted(sub["ratio"].unique()):
            c = sub[(sub["group"] == "Cell_C") & (sub["ratio"] == ratio)]["delta_gt_score_signed"]
            d = sub[(sub["group"] == "Cell_D") & (sub["ratio"] == ratio)]["delta_gt_score_signed"]
            if len(c) == 0 or len(d) == 0:
                continue
            c_k = int(sub[(sub["group"] == "Cell_C") & (sub["ratio"] == ratio)]["k_actual"].iloc[0])
            d_k = int(sub[(sub["group"] == "Cell_D") & (sub["ratio"] == ratio)]["k_actual"].iloc[0])

            c_def_a = float(c.abs().mean())
            d_def_a = float(d.abs().mean())
            cd_a = c_def_a / d_def_a if d_def_a > 0 else float("nan")

            c_def_b = float(abs(c.mean()))
            d_def_b = float(abs(d.mean()))
            cd_b = c_def_b / d_def_b if d_def_b > 0 else float("nan")

            rows.append({
                "importance_metric": imp_name, "ablation_type": ablation_type,
                "delta_metric": "gt_score", "ratio": ratio,
                "cell_c_k": c_k, "cell_d_k": d_k, "k_equal": c_k == d_k,
                "cd_ratio_def_b": cd_b, "cd_ratio_def_a": cd_a,
                "cell_c_mean_def_b": c_def_b, "cell_d_mean_def_b": d_def_b,
                "cell_c_mean_def_a": c_def_a, "cell_d_mean_def_a": d_def_a,
            })
    return rows


# ======================================================================
# MAIN
# ======================================================================
def main():
    t0_total = time.time()
    log("Greater-Than Full Pipeline")

    model = HookedTransformer.from_pretrained("gpt2", device=DEVICE)
    n_layers, n_heads, d_head = model.cfg.n_layers, model.cfg.n_heads, model.cfg.d_head
    total_heads = n_layers * n_heads
    log(f"GPT-2: {n_layers}L x {n_heads}H x {d_head}D = {total_heads} heads")

    # --- Data ---
    prompts, start_years = generate_gt_data(N_PROMPTS, SEED)
    corrupted = generate_corrupted(prompts, start_years)
    clean_toks = model.to_tokens(prompts, prepend_bos=True)
    corrupt_toks = model.to_tokens(corrupted, prepend_bos=True)
    last_pos = clean_toks.shape[1] - 1

    # ================================================================
    # PHASE 10: Collection
    # ================================================================
    log("=== Phase 10: Collection ===")
    t0 = time.time()
    dir_10 = ensure_dir(os.path.join(TASK_ROOT, "10_collection"))

    with torch.no_grad():
        clean_logits, clean_cache = model.run_with_cache(clean_toks)
        corrupt_logits, corrupt_cache = model.run_with_cache(corrupt_toks)

    clean_z = np.zeros((N_PROMPTS, n_layers, n_heads, d_head), dtype=np.float32)
    corrupt_z = np.zeros_like(clean_z)
    for layer in range(n_layers):
        hook = utils.get_act_name("z", layer)
        clean_z[:, layer] = clean_cache[hook][:, last_pos, :, :].cpu().numpy()
        corrupt_z[:, layer] = corrupt_cache[hook][:, last_pos, :, :].cpu().numpy()

    np.savez_compressed(os.path.join(dir_10, "clean_z.npz"), activations=clean_z)
    np.savez_compressed(os.path.join(dir_10, "corrupt_z.npz"), activations=corrupt_z)

    clean_score = compute_gt_score_mean(clean_logits, model, start_years, last_pos)
    log(f"Clean GT score: {clean_score:.4f}")

    prompt_rows = [{"prompt_id": i, "text": prompts[i], "start_year": start_years[i]}
                   for i in range(N_PROMPTS)]
    write_csv(os.path.join(dir_10, "trial_definitions.csv"), prompt_rows)

    gt_rows = [{"layer": l, "head": h, "subclass": sc} for l, h, sc in GT_CIRCUIT]
    write_csv(os.path.join(dir_10, "gt_circuit.csv"), gt_rows)

    arch = {"model_id": "gpt2", "num_layers": n_layers, "num_heads": n_heads,
            "head_dim": d_head, "hidden_size": model.cfg.d_model,
            "total_heads": total_heads, "vocab_size": model.cfg.d_vocab}
    with open(os.path.join(dir_10, "architecture.json"), "w") as f:
        json.dump(arch, f, indent=2)

    save_config(dir_10, n_prompts=N_PROMPTS, seed=SEED,
                clean_gt_score=clean_score, elapsed_sec=round(time.time()-t0, 1))
    log(f"Phase 10 done ({time.time()-t0:.0f}s)")

    # ================================================================
    # PHASE 20: Scoring
    # ================================================================
    log("=== Phase 20: Scoring ===")
    t0 = time.time()
    dir_20 = ensure_dir(os.path.join(TASK_ROOT, "20_scoring"))

    # Importance: GT score drop when patching each head
    head_importance = np.zeros((n_layers, n_heads))
    for layer in range(n_layers):
        hook_name = utils.get_act_name("z", layer)
        for head in range(n_heads):
            def patch_hook(activation, hook, h=head):
                activation[:, :, h, :] = corrupt_cache[hook.name][:, :, h, :]
                return activation
            with torch.no_grad():
                patched_logits = model.run_with_hooks(clean_toks, fwd_hooks=[(hook_name, patch_hook)])
            patched_score = compute_gt_score_mean(patched_logits, model, start_years, last_pos)
            head_importance[layer, head] = clean_score - patched_score
        if (layer + 1) % 4 == 0:
            log(f"  Importance: layer {layer+1}/{n_layers}")

    # Perturbation (4 metrics)
    pert_rows = []
    for layer in range(n_layers):
        for head in range(n_heads):
            c = clean_z[:, layer, head, :]
            x = corrupt_z[:, layer, head, :]
            diff = c - x
            raw_l2 = np.linalg.norm(diff, axis=-1).mean()
            c_norm = np.linalg.norm(c, axis=-1)
            c_safe = np.where(c_norm > 0, c_norm, 1.0)
            rel_l2 = (np.linalg.norm(diff, axis=-1) / c_safe).mean()
            dot = (c * x).sum(axis=-1)
            x_norm = np.linalg.norm(x, axis=-1)
            denom = np.where(c_norm * x_norm > 0, c_norm * x_norm, 1.0)
            cos_dist = (1.0 - dot / denom).mean()
            dm = diff.mean(axis=0, keepdims=True)
            ds = diff.std(axis=0, keepdims=True)
            ds_safe = np.where(ds > 0, ds, 1.0)
            std_l2 = np.linalg.norm((diff - dm) / ds_safe, axis=-1).mean()
            pert_rows.append({"layer": layer, "head": head,
                              "pert_raw_l2": float(raw_l2), "pert_relative_l2": float(rel_l2),
                              "pert_cosine_distance": float(cos_dist),
                              "pert_standardized_l2": float(std_l2)})
    write_csv(os.path.join(dir_20, "perturbation_per_head.csv"), pert_rows)

    # Importance CSV
    imp_flat = head_importance.flatten()
    imp_abs_flat = np.abs(imp_flat)
    pert_flat = np.array([r["pert_raw_l2"] for r in pert_rows])

    imp_rows = [{"layer": l, "head": h,
                 "imp_gt_score_drop": float(imp_flat[l*n_heads+h]),
                 "imp_abs_gt_score_drop": float(imp_abs_flat[l*n_heads+h])}
                for l in range(n_layers) for h in range(n_heads)]
    write_csv(os.path.join(dir_20, "importance_per_head.csv"), imp_rows)

    # Cell classification (both signed and abs)
    pert_median = float(np.median(pert_flat))
    cell_classifications = {}

    for imp_name, imp_vals in [("signed", imp_flat), ("abs", imp_abs_flat)]:
        imp_med = float(np.median(imp_vals))
        rows, heads, cmap = [], {"A":[],"B":[],"C":[],"D":[]}, {}
        for l in range(n_layers):
            for h in range(n_heads):
                idx = l * n_heads + h
                iv, pv = imp_vals[idx], pert_flat[idx]
                hi_i, hi_p = iv >= imp_med, pv >= pert_median
                c = "A" if hi_i and hi_p else "B" if hi_i else "C" if hi_p else "D"
                heads[c].append((l, h))
                cmap[(l, h)] = c
                rows.append({"layer": l, "head": h, "importance_metric": imp_name,
                             "imp_value": float(iv), "pert_value": float(pv),
                             "hi_imp": hi_i, "hi_pert": hi_p, "cell": c,
                             "imp_median": imp_med, "pert_median": pert_median,
                             "in_gt_circuit": (l, h) in GT_CIRCUIT_HEADS,
                             "gt_subclass": GT_CIRCUIT_MAP.get((l, h), "")})
        cell_classifications[imp_name] = {"rows": rows, "heads": heads, "map": cmap}
        counts = {c: len(heads[c]) for c in "ABCD"}
        gt_in = {c: sum(1 for lh in heads[c] if lh in GT_CIRCUIT_HEADS) for c in "ABCD"}
        log(f"  [{imp_name}] Cells: {counts}, GT: A={gt_in['A']} B={gt_in['B']} C={gt_in['C']} D={gt_in['D']}")

    all_cell_rows = cell_classifications["signed"]["rows"] + cell_classifications["abs"]["rows"]
    write_csv(os.path.join(dir_20, "cell_classification.csv"), all_cell_rows)

    rho, _ = spearmanr(imp_flat, pert_flat)
    rho_abs, _ = spearmanr(imp_abs_flat, pert_flat)
    log(f"rho(signed, pert): {rho:.4f}, rho(abs, pert): {rho_abs:.4f}")

    save_config(dir_20, n_heads=total_heads, pert_median=pert_median,
                rho_signed=rho, rho_abs=rho_abs,
                elapsed_sec=round(time.time()-t0, 1))
    log(f"Phase 20 done ({time.time()-t0:.0f}s)")

    # ================================================================
    # PHASE 30: Patching (dose-response)
    # ================================================================
    log("=== Phase 30: Patching ===")
    t0 = time.time()
    dir_30 = ensure_dir(os.path.join(TASK_ROOT, "30_patching"))

    # Per-prompt GT scores for clean baseline
    clean_scores_per_prompt = compute_gt_score(clean_logits, model, start_years, last_pos)

    all_dose_rows = []
    for imp_name in ["signed", "abs"]:
        cc = cell_classifications[imp_name]
        ch = {c: list(heads) for c, heads in cc["heads"].items()}
        iv = imp_flat if imp_name == "signed" else imp_abs_flat

        for c in ["A", "B"]:
            ch[c].sort(key=lambda lh: iv[lh[0]*n_heads+lh[1]], reverse=True)
        for c in ["C", "D"]:
            ch[c].sort(key=lambda lh: pert_flat[lh[0]*n_heads+lh[1]])

        imp_ord = sorted(range(total_heads), key=lambda i: iv[i])
        imp_ord_lh = [(i // n_heads, i % n_heads) for i in imp_ord]

        c_size, d_size = len(ch["C"]), len(ch["D"])
        k_cap = min(c_size, d_size)

        groups = {"Cell_A": ch["A"], "Cell_B": ch["B"],
                  "Cell_C": ch["C"], "Cell_D": ch["D"][:k_cap],
                  "Cell_D_full": ch["D"], "imp_ordered": imp_ord_lh}

        log(f"  [{imp_name}] C={c_size}, D={d_size}, k_cap={k_cap}")

        for gname, glist in groups.items():
            for ratio in RATIOS_DOSE:
                k = max(1, math.ceil(total_heads * ratio))
                k_actual = min(k, len(glist))
                heads_to_patch = glist[:k_actual]
                if not heads_to_patch:
                    continue

                by_layer = {}
                for l, h in heads_to_patch:
                    by_layer.setdefault(l, []).append(h)
                hooks = []
                for l in sorted(by_layer):
                    hook_name = utils.get_act_name("z", l)
                    def make_hook(hs):
                        def fn(act, hook, heads=hs):
                            for h in heads:
                                act[:, :, h, :] = corrupt_cache[hook.name][:, :, h, :]
                            return act
                        return fn
                    hooks.append((hook_name, make_hook(by_layer[l])))

                with torch.no_grad():
                    patched_logits = model.run_with_hooks(clean_toks, fwd_hooks=hooks)

                patched_scores = compute_gt_score(patched_logits, model, start_years, last_pos)
                for pi in range(N_PROMPTS):
                    delta = clean_scores_per_prompt[pi] - patched_scores[pi]
                    all_dose_rows.append({
                        "importance_metric": imp_name, "group": gname,
                        "ratio": ratio, "k": k, "k_actual": k_actual,
                        "prompt_id": pi, "delta_gt_score_signed": float(delta),
                    })

        log(f"  [{imp_name}] done")

    dose_df = pd.DataFrame(all_dose_rows)
    dose_df.to_csv(os.path.join(dir_30, "dose_response.csv"), index=False)

    cd_rows = compute_cd_summary(dose_df, ["signed", "abs"])
    write_csv(os.path.join(dir_30, "cd_ratio_summary.csv"), cd_rows)

    for imp_name in ["signed", "abs"]:
        cd_10 = [r for r in cd_rows if r["ratio"] == 0.10
                 and r["importance_metric"] == imp_name and r["k_equal"]]
        if cd_10:
            log(f"C/D @10% [{imp_name}] Def B: {cd_10[0]['cd_ratio_def_b']:.3f}, "
                f"Def A: {cd_10[0]['cd_ratio_def_a']:.3f}")

    save_config(dir_30, n_ratios=len(RATIOS_DOSE), n_prompts=N_PROMPTS,
                elapsed_sec=round(time.time()-t0, 1))
    log(f"Phase 30 done ({time.time()-t0:.0f}s)")

    # ================================================================
    # PHASE 35: Circuit Recovery
    # ================================================================
    log("=== Phase 35: Circuit Recovery ===")
    t0 = time.time()
    dir_35 = ensure_dir(os.path.join(TASK_ROOT, "35_recoverability"))

    # AUROC
    try:
        from sklearn.metrics import roc_auc_score
        y_true = np.array([1 if (i//n_heads, i%n_heads) in GT_CIRCUIT_HEADS else 0
                           for i in range(total_heads)])
        auroc_signed = roc_auc_score(y_true, imp_flat)
        auroc_abs = roc_auc_score(y_true, imp_abs_flat)
        log(f"AUROC (signed): {auroc_signed:.3f}, AUROC (abs): {auroc_abs:.3f}")
    except ImportError:
        auroc_signed = auroc_abs = float("nan")

    # Threshold recovery
    recovery_rows = []
    for imp_name, iv in [("signed", np.abs(imp_flat)), ("abs", imp_abs_flat)]:
        cm = cell_classifications[imp_name]["map"]
        std_thresh = float(np.median(iv))
        d_indices = [i for i in range(total_heads) if cm[(i//n_heads, i%n_heads)] == "D"]
        d_thresh = float(np.median(iv[d_indices])) if d_indices else 0

        for rule, thresh in [("standard", std_thresh), ("cell_d", d_thresh)]:
            selected = {(i//n_heads, i%n_heads) for i in range(total_heads) if iv[i] >= thresh}
            recovered = selected & GT_CIRCUIT_HEADS
            recall = len(recovered) / N_GT_HEADS
            prec = len(recovered) / len(selected) if selected else 0
            recovery_rows.append({"importance_metric": imp_name, "threshold_rule": rule,
                                  "threshold": thresh, "n_selected": len(selected),
                                  "n_recovered": len(recovered), "recall": recall, "precision": prec})

    write_csv(os.path.join(dir_35, "threshold_recovery.csv"), recovery_rows)

    save_config(dir_35, n_gt_heads=N_GT_HEADS, auroc_signed=auroc_signed,
                auroc_abs=auroc_abs, elapsed_sec=round(time.time()-t0, 1))
    log(f"Phase 35 done ({time.time()-t0:.0f}s)")

    # ================================================================
    # PHASE 90: Post-hoc
    # ================================================================
    log("=== Phase 90: Post-hoc ===")
    t0 = time.time()
    dir_90 = ensure_dir(os.path.join(TASK_ROOT, "90_post_hoc"))

    from sklearn.linear_model import LinearRegression
    y = imp_abs_flat
    X_pert = pert_flat.reshape(-1, 1)
    r2_pert = LinearRegression().fit(X_pert, y).score(X_pert, y)

    summary = {"rho_signed": rho, "rho_abs": rho_abs,
               "auroc_signed": auroc_signed, "auroc_abs": auroc_abs,
               "r2_pert_predicts_abs_imp": r2_pert,
               "clean_gt_score": clean_score, "n_gt_heads": N_GT_HEADS}
    with open(os.path.join(dir_90, "posthoc_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    save_config(dir_90, elapsed_sec=round(time.time()-t0, 1))
    log(f"Phase 90 done ({time.time()-t0:.0f}s)")

    # ================================================================
    # Summary
    # ================================================================
    total_elapsed = time.time() - t0_total
    log(f"\n{'='*60}")
    log(f"Greater-Than Pipeline Complete -- {total_elapsed:.0f}s")
    log(f"{'='*60}")
    log(f"rho(signed, pert): {rho:.4f}")
    log(f"rho(abs, pert): {rho_abs:.4f}")
    log(f"AUROC (signed): {auroc_signed:.3f}")
    log(f"AUROC (abs): {auroc_abs:.3f}")
    for r in recovery_rows:
        log(f"Recovery [{r['importance_metric']}, {r['threshold_rule']}]: "
            f"{r['n_recovered']}/{N_GT_HEADS}")
    log(f"Output: {TASK_ROOT}")


if __name__ == "__main__":
    main()

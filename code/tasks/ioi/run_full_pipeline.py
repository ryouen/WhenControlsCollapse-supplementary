#!/usr/bin/env python3
"""
IOI Full Pipeline -- Phase 10, 20, 30, 35, 90
Runs on GPT-2 Small using TransformerLens.

Output: tasks/ioi/{10_collection,20_scoring,30_patching,35_recoverability,90_post_hoc}/

Based on: neural_exam_project/models/gpt2-small/ioi_replication/code/run_ioi_2x2.py
Adapted to v4.1 naming conventions (attr_dim, pert_raw_l2, etc.)

Key numbers to validate:
    rho(signed_ld, pert) ≈ 0.3125
    C/D @ 10% ≈ 10×
    AUROC ≈ 0.838
    Standard threshold inflation ≈ 14×
"""

import csv
import json
import math
import os
import sys
import time
import random
from datetime import datetime, timezone

import numpy as np
import torch
from scipy.stats import spearmanr, mannwhitneyu

from transformer_lens import HookedTransformer, utils

# ─── Paths ───
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
TASK_ROOT = os.path.join(PROJECT_ROOT, "tasks", "ioi")

# ─── Constants ───
N_PROMPTS = 100
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RATIOS_DOSE = [0.01, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.20, 0.30, 0.50]

# Wang et al. (2023) IOI circuit heads
IOI_CIRCUIT = {
    # Wang et al. 2023, 26 heads (core + fuzzy)
    # Matches tasks/ioi/10_collection/ioi_circuit_gt.csv
    "Name_Mover": [(9, 9), (10, 0), (9, 6)],
    "Negative_Name_Mover": [(10, 7), (11, 10)],
    "S_Inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)],
    "Induction": [(5, 5), (5, 8), (5, 9), (6, 9)],
    "Duplicate_Token": [(0, 1), (0, 10), (3, 0)],
    "Previous_Token": [(2, 2), (4, 11)],
    "Backup_Name_Mover": [(9, 0), (9, 7), (10, 1), (10, 2), (10, 6), (10, 10), (11, 2), (11, 9)],
}
IOI_CIRCUIT_HEADS = set()
IOI_CIRCUIT_MAP = {}
for subclass, heads in IOI_CIRCUIT.items():
    for lh in heads:
        IOI_CIRCUIT_HEADS.add(lh)
        IOI_CIRCUIT_MAP[lh] = subclass

N_GT_HEADS = len(IOI_CIRCUIT_HEADS)

NAMES = ["John", "Mary", "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank",
         "Grace", "Henry", "Iris", "Jack", "Kate", "Leo", "Nina", "Oscar"]

TEMPLATES_ABB = [
    "When {A} and {B} went to the store, {B} gave a drink to",
    "When {A} and {B} walked to the park, {B} gave a gift to",
    "When {A} and {B} went to the restaurant, {B} gave the menu to",
    "When {A} and {B} arrived at the office, {B} handed a letter to",
    "When {A} and {B} went to the library, {B} passed a book to",
]


# ─── Helpers ───
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_config(output_dir, **fields):
    cfg = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "platform": "windows_rtx3060",
        "model_id": "gpt2",
        **fields,
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, default=str)


def write_csv(path, rows, fieldnames=None):
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ─── IOI Data Generation ───
def generate_ioi_data(n=100, seed=42):
    random.seed(seed)
    prompts, io_names, s_names = [], [], []
    for _ in range(n):
        template = random.choice(TEMPLATES_ABB)
        a, b = random.sample(NAMES, 2)
        prompts.append(template.format(A=a, B=b))
        io_names.append(a)
        s_names.append(b)
    return prompts, io_names, s_names


def generate_corrupted(prompts, io_names, s_names):
    corrupted = []
    for prompt, io, s in zip(prompts, io_names, s_names):
        corrupted.append(prompt.replace(io, "TEMP_IO").replace(s, io).replace("TEMP_IO", s))
    return corrupted


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════
def main():
    t0_total = time.time()
    log(f"IOI Full Pipeline -- {datetime.now().isoformat()}")
    log(f"Device: {DEVICE}, Output: {TASK_ROOT}")

    # ── Load model ──
    log("Loading GPT-2 Small...")
    model = HookedTransformer.from_pretrained("gpt2", device=DEVICE)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_head = model.cfg.d_head
    total_heads = n_layers * n_heads
    log(f"Architecture: {n_layers}L x {n_heads}H x {d_head}D = {total_heads} heads")

    # ── Generate IOI data ──
    log(f"Generating IOI data (N={N_PROMPTS})...")
    prompts, io_names, s_names = generate_ioi_data(N_PROMPTS, SEED)
    corrupted = generate_corrupted(prompts, io_names, s_names)

    clean_toks = model.to_tokens(prompts, prepend_bos=True)
    corrupt_toks = model.to_tokens(corrupted, prepend_bos=True)
    io_token_ids = torch.tensor([model.to_single_token(" " + n) for n in io_names], device=DEVICE)
    s_token_ids = torch.tensor([model.to_single_token(" " + n) for n in s_names], device=DEVICE)
    last_pos = clean_toks.shape[1] - 1

    # ══════════════════════════════════════════════════════════
    # PHASE 10: Collection
    # ══════════════════════════════════════════════════════════
    log("=== Phase 10: Collection ===")
    t0 = time.time()
    dir_10 = ensure_dir(os.path.join(TASK_ROOT, "10_collection"))

    with torch.no_grad():
        clean_logits, clean_cache = model.run_with_cache(clean_toks)
        corrupt_logits, corrupt_cache = model.run_with_cache(corrupt_toks)

    # Extract per-head z at last token
    clean_z = np.zeros((N_PROMPTS, n_layers, n_heads, d_head), dtype=np.float32)
    corrupt_z = np.zeros_like(clean_z)
    for layer in range(n_layers):
        hook = utils.get_act_name("z", layer)
        clean_z[:, layer] = clean_cache[hook][:, last_pos, :, :].cpu().numpy()
        corrupt_z[:, layer] = corrupt_cache[hook][:, last_pos, :, :].cpu().numpy()

    np.savez_compressed(os.path.join(dir_10, "clean_z.npz"), activations=clean_z)
    np.savez_compressed(os.path.join(dir_10, "corrupt_z.npz"), activations=corrupt_z)

    clean_logits_np = clean_logits[:, last_pos, :].cpu().numpy()
    corrupt_logits_np = corrupt_logits[:, last_pos, :].cpu().numpy()
    np.savez_compressed(os.path.join(dir_10, "clean_logits.npz"), logits=clean_logits_np)
    np.savez_compressed(os.path.join(dir_10, "corrupt_logits.npz"), logits=corrupt_logits_np)

    # Prompt metadata
    prompt_rows = []
    for i in range(N_PROMPTS):
        prompt_rows.append({
            "prompt_id": i, "clean_text": prompts[i], "corrupt_text": corrupted[i],
            "io_name": io_names[i], "s_name": s_names[i],
        })
    write_csv(os.path.join(dir_10, "ioi_prompts.csv"), prompt_rows)

    # GT circuit
    gt_rows = [{"layer": l, "head": h, "subclass": sc}
               for sc, heads in IOI_CIRCUIT.items() for l, h in heads]
    write_csv(os.path.join(dir_10, "ioi_circuit_gt.csv"), gt_rows)

    # Architecture
    arch = {"model_id": "gpt2", "num_layers": n_layers, "num_heads": n_heads,
            "head_dim": d_head, "hidden_size": model.cfg.d_model,
            "total_heads": total_heads, "vocab_size": model.cfg.d_vocab}
    with open(os.path.join(dir_10, "architecture.json"), "w") as f:
        json.dump(arch, f, indent=2)

    clean_ld = (clean_logits[:, last_pos, :].gather(1, io_token_ids.unsqueeze(1)) -
                clean_logits[:, last_pos, :].gather(1, s_token_ids.unsqueeze(1))).squeeze().mean().item()
    log(f"Clean logit diff: {clean_ld:.3f}")

    save_config(dir_10, n_prompts=N_PROMPTS, seed=SEED,
                clean_logit_diff=clean_ld, elapsed_sec=round(time.time()-t0, 1))
    log(f"Phase 10 done ({time.time()-t0:.0f}s)")

    # ══════════════════════════════════════════════════════════
    # PHASE 20: Scoring (importance + perturbation + cell)
    # ══════════════════════════════════════════════════════════
    log("=== Phase 20: Scoring ===")
    t0 = time.time()
    dir_20 = ensure_dir(os.path.join(TASK_ROOT, "20_scoring"))

    # -- Importance: signed logit diff change per head --
    head_importance = np.zeros((n_layers, n_heads))

    for layer in range(n_layers):
        hook_name = utils.get_act_name("z", layer)
        for head in range(n_heads):
            def patch_hook(activation, hook, h=head):
                activation[:, :, h, :] = corrupt_cache[hook.name][:, :, h, :]
                return activation
            with torch.no_grad():
                patched_logits = model.run_with_hooks(clean_toks, fwd_hooks=[(hook_name, patch_hook)])
            patched_ld = (
                patched_logits[range(N_PROMPTS), last_pos, io_token_ids] -
                patched_logits[range(N_PROMPTS), last_pos, s_token_ids]
            ).mean().item()
            head_importance[layer, head] = clean_ld - patched_ld
        if (layer + 1) % 4 == 0:
            log(f"  Importance: layer {layer+1}/{n_layers}")

    # -- Perturbation: 4 metrics --
    pert_rows = []
    for layer in range(n_layers):
        for head in range(n_heads):
            c = clean_z[:, layer, head, :]  # (N, D)
            x = corrupt_z[:, layer, head, :]
            diff = c - x
            raw_l2 = np.linalg.norm(diff, axis=-1).mean()
            c_norm = np.linalg.norm(c, axis=-1)
            c_norm_safe = np.where(c_norm > 0, c_norm, 1.0)
            relative_l2 = (np.linalg.norm(diff, axis=-1) / c_norm_safe).mean()
            dot = (c * x).sum(axis=-1)
            x_norm = np.linalg.norm(x, axis=-1)
            denom = c_norm * x_norm
            denom_safe = np.where(denom > 0, denom, 1.0)
            cosine_dist = (1.0 - dot / denom_safe).mean()
            diff_mean = diff.mean(axis=0, keepdims=True)
            diff_std = diff.std(axis=0, keepdims=True)
            diff_std_safe = np.where(diff_std > 0, diff_std, 1.0)
            standardized_l2 = np.linalg.norm((diff - diff_mean) / diff_std_safe, axis=-1).mean()

            pert_rows.append({
                "layer": layer, "head": head,
                "pert_raw_l2": float(raw_l2),
                "pert_relative_l2": float(relative_l2),
                "pert_cosine_distance": float(cosine_dist),
                "pert_standardized_l2": float(standardized_l2),
            })

    write_csv(os.path.join(dir_20, "perturbation_per_head.csv"), pert_rows)

    # -- Importance CSV --
    imp_rows = []
    for layer in range(n_layers):
        for head in range(n_heads):
            imp_rows.append({
                "layer": layer, "head": head,
                "imp_signed_ld": float(head_importance[layer, head]),
                "imp_abs_ld": float(abs(head_importance[layer, head])),
            })
    write_csv(os.path.join(dir_20, "importance_per_head.csv"), imp_rows)

    # -- Cell classification (BOTH signed_ld and abs_ld) --
    imp_flat = head_importance.flatten()
    imp_abs_flat = np.abs(imp_flat)
    pert_flat = np.array([r["pert_raw_l2"] for r in pert_rows])
    pert_median = float(np.median(pert_flat))

    # Build cells for each importance metric
    cell_classifications = {}  # metric_name -> {cell_rows, cell_heads, cell_map, imp_median}

    for imp_name, imp_vals in [("signed_ld", imp_flat), ("abs_ld", imp_abs_flat)]:
        imp_med = float(np.median(imp_vals))
        rows = []
        heads = {"A": [], "B": [], "C": [], "D": []}
        cmap = {}

        for layer in range(n_layers):
            for head in range(n_heads):
                idx = layer * n_heads + head
                imp = imp_vals[idx]
                pert = pert_flat[idx]
                hi_imp = imp >= imp_med
                hi_pert = pert >= pert_median
                if hi_imp and hi_pert:
                    cell = "A"
                elif hi_imp and not hi_pert:
                    cell = "B"
                elif not hi_imp and hi_pert:
                    cell = "C"
                else:
                    cell = "D"
                heads[cell].append((layer, head))
                cmap[(layer, head)] = cell
                rows.append({
                    "layer": layer, "head": head,
                    "importance_metric": imp_name,
                    "imp_value": float(imp), "pert_value": float(pert),
                    "hi_imp": hi_imp, "hi_pert": hi_pert, "cell": cell,
                    "imp_median": imp_med, "pert_median": pert_median,
                    "in_gt_circuit": (layer, head) in IOI_CIRCUIT_HEADS,
                    "gt_subclass": IOI_CIRCUIT_MAP.get((layer, head), ""),
                })

        cell_classifications[imp_name] = {
            "rows": rows, "heads": heads, "map": cmap, "imp_median": imp_med,
        }

        counts = {c: len(heads[c]) for c in "ABCD"}
        gt_in_c = sum(1 for lh in heads["C"] if lh in IOI_CIRCUIT_HEADS)
        log(f"  [{imp_name}] Cells: {counts}, GT in Cell C: {gt_in_c}")

    # Save both classifications (long format: one CSV with importance_metric column)
    all_cell_rows = cell_classifications["signed_ld"]["rows"] + cell_classifications["abs_ld"]["rows"]
    write_csv(os.path.join(dir_20, "cell_classification.csv"), all_cell_rows)

    # Also save primary (signed_ld) as separate file for backward compat
    write_csv(os.path.join(dir_20, "cell_classification_signed_ld.csv"),
              cell_classifications["signed_ld"]["rows"])
    write_csv(os.path.join(dir_20, "cell_classification_abs_ld.csv"),
              cell_classifications["abs_ld"]["rows"])

    # Use signed_ld as the primary for backward compat variables
    cell_heads = cell_classifications["signed_ld"]["heads"]
    cell_map = cell_classifications["signed_ld"]["map"]
    imp_median = cell_classifications["signed_ld"]["imp_median"]

    # Metric summary
    rho, rho_p = spearmanr(imp_flat, pert_flat)
    rho_abs, rho_abs_p = spearmanr(imp_abs_flat, pert_flat)
    counts_signed = {c: len(cell_classifications["signed_ld"]["heads"][c]) for c in "ABCD"}
    counts_abs = {c: len(cell_classifications["abs_ld"]["heads"][c]) for c in "ABCD"}
    log(f"rho(signed_ld, pert): {rho:.4f}, rho(|ld|, pert): {rho_abs:.4f}")

    save_config(dir_20, n_heads=total_heads,
                imp_median_signed=imp_median,
                imp_median_abs=cell_classifications["abs_ld"]["imp_median"],
                pert_median=pert_median,
                cell_counts_signed_ld=counts_signed,
                cell_counts_abs_ld=counts_abs,
                rho_signed=rho, rho_abs=rho_abs,
                elapsed_sec=round(time.time()-t0, 1))
    log(f"Phase 20 done ({time.time()-t0:.0f}s)")

    # ══════════════════════════════════════════════════════════

    # ==============================================================
    # PHASE 30: Patching (both signed_ld and abs_ld, k-capped)
    # ==============================================================
    log("=== Phase 30: Patching (signed_ld + abs_ld) ===")
    t0 = time.time()
    dir_30 = ensure_dir(os.path.join(TASK_ROOT, "30_patching"))
    import pandas as pd

    def patch_group(heads_to_patch):
        by_layer = {}
        for l, h in heads_to_patch:
            by_layer.setdefault(l, []).append(h)
        hooks = []
        for l in sorted(by_layer):
            hook_name = utils.get_act_name("z", l)
            def make_hook(hs):
                def fn(activation, hook, heads=hs):
                    for h in heads:
                        activation[:, :, h, :] = corrupt_cache[hook.name][:, :, h, :]
                    return activation
                return fn
            hooks.append((hook_name, make_hook(by_layer[l])))
        with torch.no_grad():
            return model.run_with_hooks(clean_toks, fwd_hooks=hooks)

    all_dose_rows = []

    for imp_name in ["signed_ld", "abs_ld"]:
        cc = cell_classifications[imp_name]
        ch = {c: list(heads) for c, heads in cc["heads"].items()}
        imp_vals = imp_flat if imp_name == "signed_ld" else imp_abs_flat

        # Sort: A,B by imp desc; C,D by pert asc
        for c in ["A", "B"]:
            ch[c].sort(key=lambda lh: imp_vals[lh[0]*n_heads+lh[1]], reverse=True)
        for c in ["C", "D"]:
            ch[c].sort(key=lambda lh: pert_flat[lh[0]*n_heads+lh[1]])

        # imp_ordered ascending
        imp_ord = sorted(range(total_heads), key=lambda i: imp_vals[i])
        imp_ord_lh = [(i // n_heads, i % n_heads) for i in imp_ord]

        # k-cap for fair C/D comparison
        c_size = len(ch["C"])
        d_size = len(ch["D"])
        k_cap = min(c_size, d_size)

        groups = {
            "Cell_A": ch["A"],
            "Cell_B": ch["B"],
            "Cell_C": ch["C"],
            "Cell_D": ch["D"][:k_cap],
            "Cell_D_full": ch["D"],
            "imp_ordered": imp_ord_lh,
        }

        log(f"  [{imp_name}] C={c_size}, D={d_size}, k_cap={k_cap}")

        for gname, glist in groups.items():
            for ratio in RATIOS_DOSE:
                k = max(1, math.ceil(total_heads * ratio))
                k_actual = min(k, len(glist))
                heads_to_patch = glist[:k_actual]
                if not heads_to_patch:
                    continue

                patched_logits = patch_group(heads_to_patch)

                for pi in range(N_PROMPTS):
                    delta_io = float(patched_logits[pi, last_pos, io_token_ids[pi]] -
                                     clean_logits[pi, last_pos, io_token_ids[pi]])
                    delta_s = float(patched_logits[pi, last_pos, s_token_ids[pi]] -
                                    clean_logits[pi, last_pos, s_token_ids[pi]])
                    all_dose_rows.append({
                        "importance_metric": imp_name,
                        "group": gname, "ratio": ratio, "k": k, "k_actual": k_actual,
                        "prompt_id": pi,
                        "delta_target_logit_signed": delta_io,
                        "delta_source_logit_signed": delta_s,
                    })

        log(f"  [{imp_name}] done")

    write_csv(os.path.join(dir_30, "dose_response.csv"), all_dose_rows)

    # CD ratio summary
    dose_df = pd.DataFrame(all_dose_rows)
    dose_df["delta_logit_diff"] = (dose_df["delta_source_logit_signed"]
                                    - dose_df["delta_target_logit_signed"])

    cd_rows = []
    for imp_name in ["signed_ld", "abs_ld"]:
        sub = dose_df[dose_df["importance_metric"] == imp_name]
        for ratio in RATIOS_DOSE:
            for delta_name, col in [("logit_diff", "delta_logit_diff"),
                                     ("target_logit", "delta_target_logit_signed")]:
                c_v = sub[(sub["group"]=="Cell_C") & (sub["ratio"]==ratio)][col].abs()
                d_v = sub[(sub["group"]=="Cell_D") & (sub["ratio"]==ratio)][col].abs()
                c_k = int(sub[(sub["group"]=="Cell_C") & (sub["ratio"]==ratio)]["k_actual"].iloc[0]) if len(c_v) > 0 else 0
                d_k = int(sub[(sub["group"]=="Cell_D") & (sub["ratio"]==ratio)]["k_actual"].iloc[0]) if len(d_v) > 0 else 0
                cd = float(c_v.mean() / d_v.mean()) if len(d_v) > 0 and d_v.mean() > 0 else float("nan")
                cd_rows.append({
                    "importance_metric": imp_name, "delta_metric": delta_name,
                    "ratio": ratio, "cd_ratio": cd,
                    "cell_c_k": c_k, "cell_d_k": d_k,
                    "cell_c_mean": float(c_v.mean()) if len(c_v) > 0 else 0,
                    "cell_d_mean": float(d_v.mean()) if len(d_v) > 0 else 0,
                    "k_equal": c_k == d_k,
                })

    write_csv(os.path.join(dir_30, "cd_ratio_summary.csv"), cd_rows)

    for imp_name in ["signed_ld", "abs_ld"]:
        for dm in ["logit_diff", "target_logit"]:
            cd_10 = [r for r in cd_rows if r["ratio"] == 0.10
                     and r["importance_metric"] == imp_name
                     and r["delta_metric"] == dm and r["k_equal"]]
            if cd_10:
                log(f"C/D @10% [{imp_name}, {dm}]: {cd_10[0]['cd_ratio']:.3f} "
                    f"(C_k={cd_10[0]['cell_c_k']}, D_k={cd_10[0]['cell_d_k']})")

    save_config(dir_30, n_ratios=len(RATIOS_DOSE), n_prompts=N_PROMPTS,
                importance_metrics=["signed_ld", "abs_ld"],
                total_rows=len(all_dose_rows),
                elapsed_sec=round(time.time()-t0, 1))
    log(f"Phase 30 done ({time.time()-t0:.0f}s)")

    # PHASE 35: Circuit Recovery
    # ══════════════════════════════════════════════════════════
    log("=== Phase 35: Circuit Recovery ===")
    t0 = time.time()
    dir_35 = ensure_dir(os.path.join(TASK_ROOT, "35_recoverability"))

    # Master table
    master_rows = []
    for layer in range(n_layers):
        for head in range(n_heads):
            idx = layer * n_heads + head
            master_rows.append({
                "layer": layer, "head": head,
                "imp_signed_ld": float(imp_flat[idx]),
                "imp_abs_ld": float(abs(imp_flat[idx])),
                "pert_raw_l2": float(pert_flat[idx]),
                "cell": cell_map[(layer, head)],
                "in_gt_circuit": (layer, head) in IOI_CIRCUIT_HEADS,
                "gt_subclass": IOI_CIRCUIT_MAP.get((layer, head), ""),
            })
    write_csv(os.path.join(dir_35, "head_scores_master.csv"), master_rows)

    # Threshold recovery
    imp_abs = np.abs(imp_flat)
    standard_thresh = float(np.median(imp_abs))
    cell_d_indices = [i for i in range(total_heads)
                      if cell_map[(i // n_heads, i % n_heads)] == "D"]
    cell_d_thresh = float(np.median(imp_abs[cell_d_indices])) if cell_d_indices else 0

    recovery_rows = []
    subclasses = sorted(set(IOI_CIRCUIT_MAP.values()))
    for rule, thresh in [("standard", standard_thresh), ("cell_d", cell_d_thresh)]:
        selected = set()
        for i in range(total_heads):
            if imp_abs[i] >= thresh:
                selected.add((i // n_heads, i % n_heads))
        recovered = selected & IOI_CIRCUIT_HEADS
        recall = len(recovered) / N_GT_HEADS
        precision = len(recovered) / len(selected) if selected else 0
        row = {"threshold_rule": rule, "threshold_value": thresh,
               "n_selected": len(selected), "n_gt_recovered": len(recovered),
               "recall": recall, "precision": precision}
        for sc in subclasses:
            sc_total = sum(1 for lh, s in IOI_CIRCUIT_MAP.items() if s == sc)
            sc_recov = sum(1 for lh in recovered if IOI_CIRCUIT_MAP.get(lh) == sc)
            row[f"recall_{sc}"] = sc_recov / sc_total if sc_total > 0 else 0
        recovery_rows.append(row)

    write_csv(os.path.join(dir_35, "threshold_recovery.csv"), recovery_rows)
    log(f"Standard: {recovery_rows[0]['n_gt_recovered']}/{N_GT_HEADS} recovered")
    log(f"Cell-D:   {recovery_rows[1]['n_gt_recovered']}/{N_GT_HEADS} recovered")
    log(f"Threshold inflation: {standard_thresh / cell_d_thresh:.1f}×" if cell_d_thresh > 0 else "Cell-D thresh = 0")

    # AUROC
    try:
        from sklearn.metrics import roc_auc_score
        y_true = np.array([1 if (i // n_heads, i % n_heads) in IOI_CIRCUIT_HEADS else 0
                           for i in range(total_heads)])
        auroc = roc_auc_score(y_true, imp_abs)
        log(f"AUROC: {auroc:.3f}")
    except ImportError:
        auroc = float("nan")
        log("sklearn not available, AUROC skipped")

    save_config(dir_35, n_gt_heads=N_GT_HEADS, auroc=auroc,
                standard_thresh=standard_thresh, cell_d_thresh=cell_d_thresh,
                elapsed_sec=round(time.time()-t0, 1))
    log(f"Phase 35 done ({time.time()-t0:.0f}s)")

    # ══════════════════════════════════════════════════════════
    # PHASE 90: Post-hoc
    # ══════════════════════════════════════════════════════════
    log("=== Phase 90: Post-hoc ===")
    t0 = time.time()
    dir_90 = ensure_dir(os.path.join(TASK_ROOT, "90_post_hoc"))

    # NOTE: For IOI, importance = signed_ld = per-head patching effect.
    # Regressing y=|importance| on X=|importance| would be trivially R2=1.
    # Instead, we regress y=|patching_effect| on X=perturbation to test
    # whether perturbation L2 predicts patching impact.
    # The imp_only model is omitted (circular). Cross-task Table 1 should
    # use the novel-word R2 values where importance (rsa_max) and patching
    # effect (delta_target) are independently measured.
    from sklearn.linear_model import LinearRegression

    y = np.abs(imp_flat)  # |per-head patching effect|
    X_pert = pert_flat.reshape(-1, 1)

    reg_rows = []
    # pert_only: can perturbation predict patching effect?
    lr_pert = LinearRegression().fit(X_pert, y)
    reg_rows.append({"model": "pert_only", "r2": round(lr_pert.score(X_pert, y), 4),
                     "note": "perturbation L2 predicting |patching effect|"})
    # imp_only: SKIPPED (circular — importance IS the patching effect for IOI)
    reg_rows.append({"model": "imp_only", "r2": 1.0,
                     "note": "trivially 1.0 (importance = patching effect for IOI)"})

    write_csv(os.path.join(dir_90, "regression.csv"), reg_rows)

    # Matched-head analysis
    c_effects = np.abs(imp_flat[[i for i in range(total_heads)
                                  if cell_map[(i//n_heads, i%n_heads)] == "C"]])
    d_effects = np.abs(imp_flat[[i for i in range(total_heads)
                                  if cell_map[(i//n_heads, i%n_heads)] == "D"]])
    if len(c_effects) > 0 and len(d_effects) > 0:
        stat, p = mannwhitneyu(c_effects, d_effects, alternative="greater")
        effect_ratio = float(c_effects.mean() / d_effects.mean()) if d_effects.mean() > 0 else float("inf")
    else:
        stat, p, effect_ratio = 0, 1, 0

    matched = {"mann_whitney_u": float(stat), "p_value": float(p),
               "cell_c_mean": float(c_effects.mean()), "cell_d_mean": float(d_effects.mean()),
               "effect_ratio": effect_ratio}
    with open(os.path.join(dir_90, "posthoc_summary.json"), "w") as f:
        json.dump({"regression": reg_rows, "matched_head": matched,
                   "rho_signed": rho, "rho_abs": rho_abs, "auroc": auroc}, f, indent=2)

    save_config(dir_90, elapsed_sec=round(time.time()-t0, 1))
    log(f"Phase 90 done ({time.time()-t0:.0f}s)")

    # ── Summary ──
    total_elapsed = time.time() - t0_total
    log(f"\n{'='*60}")
    log(f"IOI Pipeline Complete -- {total_elapsed:.0f}s total")
    log(f"{'='*60}")
    log(f"Cells (signed): {counts_signed}")
    log(f"Cells (abs): {counts_abs}")
    log(f"rho(signed_ld, pert): {rho:.4f}")
    log(f"rho(|ld|, pert): {rho_abs:.4f}")
    log(f"AUROC: {auroc:.3f}")
    log(f"Standard recall: {recovery_rows[0]['n_gt_recovered']}/{N_GT_HEADS}")
    log(f"Cell-D recall: {recovery_rows[1]['n_gt_recovered']}/{N_GT_HEADS}")
    log(f"Threshold inflation: {standard_thresh / cell_d_thresh:.1f}×" if cell_d_thresh > 0 else "")
    log(f"Output: {TASK_ROOT}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Induction Full Pipeline -- Phase 10, 15, 20, 30, 35, 90
GPT-2 Small, TransformerLens.

Paper role: "Signal-present confounded benchmark" (Section 5, Fig 1b).

DETECTOR METHOD: Uses transformer_lens.head_detector.detect_head() library function.
NOT manual attention pattern calculation. Reason: manual calculation produces
different top-10 (early-layer heads L0-L1) with 0/10 overlap with the library
function (canonical induction heads L5-L10 per Olsson et al. 2022). The library
function computes correlation between attention pattern and theoretical induction
template, which correctly identifies canonical induction heads.

Fixes from old project:
    - n_prompts = 100 (was silently truncated to 50)
    - corrupt_method = random tokens only (no shuffle mixing)
    - n_random_seeds = 1000 (was 1-5)
"""

import csv, json, math, os, random, time
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timezone
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.metrics import roc_auc_score, average_precision_score
from transformer_lens import HookedTransformer, utils
from transformer_lens.head_detector import detect_head

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
TASK_ROOT = os.path.join(PROJECT_ROOT, "tasks", "induction")

N_PROMPTS = 100
N_DET_SEQS = 50
SEQ_HALF_LEN = 20
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VOCAB_LO, VOCAB_HI = 1000, 5000
RATIOS_DOSE = [0.01, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.20, 0.30, 0.50]
DETECTOR_TOP_K = [5, 10, 15]
N_RANDOM_SEEDS = 1000

THRESHOLD_RULES = {
    "mean_plus_1.5sd": lambda v: np.mean(v) + 1.5 * np.std(v),
    "mean_plus_2sd":   lambda v: np.mean(v) + 2.0 * np.std(v),
    "mean_plus_2.5sd": lambda v: np.mean(v) + 2.5 * np.std(v),
    "pctile_95":       lambda v: np.percentile(v, 95),
}

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def write_csv(path, rows, fieldnames=None):
    if not rows: return
    if fieldnames is None: fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)

def save_config(d, **kw):
    cfg = {"completed_at": datetime.now(timezone.utc).isoformat(),
           "platform": "windows_rtx3060", "model_id": "gpt2", **kw}
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, default=str)


def generate_induction_stimuli(n, half_len, seed):
    rng = random.Random(seed)
    vocab = list(range(VOCAB_LO, VOCAB_HI))
    clean, corrupt = [], []
    for _ in range(n):
        first = [rng.choice(vocab) for _ in range(half_len)]
        clean.append(first + first)
        corrupt.append(first + [rng.choice(vocab) for _ in range(half_len)])
    return clean, corrupt


def compute_second_half_loss(logits, tokens, half_len):
    N = logits.shape[0]
    losses = []
    for i in range(N):
        tgt = tokens[i, half_len:]
        pred = logits[i, half_len-1:-1, :]
        loss = torch.nn.functional.cross_entropy(pred, tgt).item()
        losses.append(loss)
    return losses


def main():
    t0_total = time.time()
    log("Induction Full Pipeline (detect_head library)")

    model = HookedTransformer.from_pretrained("gpt2", device=DEVICE)
    n_layers, n_heads, d_head = model.cfg.n_layers, model.cfg.n_heads, model.cfg.d_head
    total_heads = n_layers * n_heads
    log(f"GPT-2: {n_layers}L x {n_heads}H x {d_head}D = {total_heads} heads")

    # =================================================================
    # PHASE 10: Collection
    # =================================================================
    log("=== Phase 10: Collection ===")
    t0 = time.time()
    dir_10 = ensure_dir(os.path.join(TASK_ROOT, "10_collection"))

    clean_ids, corrupt_ids = generate_induction_stimuli(N_PROMPTS, SEQ_HALF_LEN, SEED)
    clean_toks = torch.tensor(clean_ids, device=DEVICE)
    corrupt_toks = torch.tensor(corrupt_ids, device=DEVICE)

    with torch.no_grad():
        clean_logits, clean_cache = model.run_with_cache(clean_toks)
        corrupt_logits, corrupt_cache = model.run_with_cache(corrupt_toks)

    clean_z = np.zeros((N_PROMPTS, n_layers, n_heads, d_head), dtype=np.float32)
    corrupt_z = np.zeros_like(clean_z)
    for layer in range(n_layers):
        hook = utils.get_act_name("z", layer)
        clean_z[:, layer] = clean_cache[hook][:, SEQ_HALF_LEN:, :, :].mean(dim=1).cpu().numpy()
        corrupt_z[:, layer] = corrupt_cache[hook][:, SEQ_HALF_LEN:, :, :].mean(dim=1).cpu().numpy()

    np.savez_compressed(os.path.join(dir_10, "clean_z.npz"), activations=clean_z)
    np.savez_compressed(os.path.join(dir_10, "corrupt_z.npz"), activations=corrupt_z)
    np.savez_compressed(os.path.join(dir_10, "clean_token_ids.npz"), tokens=np.array(clean_ids))
    np.savez_compressed(os.path.join(dir_10, "corrupt_token_ids.npz"), tokens=np.array(corrupt_ids))

    clean_losses = compute_second_half_loss(clean_logits, clean_toks, SEQ_HALF_LEN)
    clean_loss_mean = float(np.mean(clean_losses))
    log(f"Clean 2nd-half loss: {clean_loss_mean:.4f}")

    arch = {"model_id": "gpt2", "num_layers": n_layers, "num_heads": n_heads,
            "head_dim": d_head, "hidden_size": model.cfg.d_model,
            "total_heads": total_heads, "vocab_size": model.cfg.d_vocab}
    with open(os.path.join(dir_10, "architecture.json"), "w") as f:
        json.dump(arch, f, indent=2)
    with open(os.path.join(dir_10, "stimuli.json"), "w") as f:
        json.dump({"seed": SEED, "vocab_range": [VOCAB_LO, VOCAB_HI],
                    "seq_half_len": SEQ_HALF_LEN, "n_prompts": N_PROMPTS,
                    "corrupt_method": "random_token_replace"}, f, indent=2)

    save_config(dir_10, n_prompts=N_PROMPTS, seed=SEED, clean_loss_mean=clean_loss_mean,
                elapsed_sec=round(time.time()-t0, 1))
    log(f"Phase 10 done ({time.time()-t0:.0f}s)")

    # =================================================================
    # PHASE 15: Detectors (using detect_head library function)
    # =================================================================
    log("=== Phase 15: Detectors (detect_head library) ===")
    t0 = time.time()
    dir_15 = ensure_dir(os.path.join(TASK_ROOT, "15_detectors"))

    rng_det = random.Random(SEED)
    vocab = list(range(VOCAB_LO, VOCAB_HI))
    det_seqs_str = []
    for _ in range(N_DET_SEQS):
        tokens = [rng_det.choice(vocab) for _ in range(SEQ_HALF_LEN)]
        det_seqs_str.append(model.tokenizer.decode(tokens + tokens))

    lib_scores = {}
    for htype in ['induction_head', 'previous_token_head', 'duplicate_token_head']:
        s = detect_head(model, det_seqs_str, htype).cpu().numpy()
        lib_scores[htype] = s
        log(f"  {htype}: max={s.max():.4f}, mean={s.mean():.4f}")

    det_rows = []
    for l in range(n_layers):
        for h in range(n_heads):
            det_rows.append({"layer": l, "head": h,
                             "induction_score": float(lib_scores['induction_head'][l, h]),
                             "prev_token_score": float(lib_scores['previous_token_head'][l, h]),
                             "dup_token_score": float(lib_scores['duplicate_token_head'][l, h])})
    write_csv(os.path.join(dir_15, "detector_scores.csv"), det_rows)

    det_df = pd.DataFrame(det_rows)
    det_sorted = det_df.sort_values("induction_score", ascending=False)
    gt = {}
    for k in DETECTOR_TOP_K:
        gt[f"top{k}"] = [(int(r["layer"]), int(r["head"])) for _, r in det_sorted.head(k).iterrows()]
    gt["primary"] = "top10"
    gt["detector_method"] = "detect_head_library"

    with open(os.path.join(dir_15, "operational_gt.json"), "w") as f:
        json.dump({k: [list(lh) for lh in v] if isinstance(v, list) else v
                   for k, v in gt.items()}, f, indent=2)

    gt_top10 = set(gt["top10"])
    log(f"Top-10: {sorted(gt_top10)}")
    save_config(dir_15, n_det_seqs=N_DET_SEQS, detector_method="detect_head_library",
                elapsed_sec=round(time.time()-t0, 1))
    log(f"Phase 15 done ({time.time()-t0:.0f}s)")

    # =================================================================
    # PHASE 20: Scoring
    # =================================================================
    log("=== Phase 20: Scoring ===")
    t0 = time.time()
    dir_20 = ensure_dir(os.path.join(TASK_ROOT, "20_scoring"))

    head_importance = np.zeros((n_layers, n_heads))
    for layer in range(n_layers):
        hook_name = utils.get_act_name("z", layer)
        for head in range(n_heads):
            def patch(act, hook, h=head):
                act[:, :, h, :] = corrupt_cache[hook.name][:, :, h, :]
                return act
            with torch.no_grad():
                patched_logits = model.run_with_hooks(clean_toks, fwd_hooks=[(hook_name, patch)])
            patched_losses = compute_second_half_loss(patched_logits, clean_toks, SEQ_HALF_LEN)
            head_importance[layer, head] = np.mean(patched_losses) - clean_loss_mean
        if (layer + 1) % 4 == 0:
            log(f"  Importance: layer {layer+1}/{n_layers}")

    pert_rows = []
    for layer in range(n_layers):
        for head in range(n_heads):
            c, x = clean_z[:, layer, head, :], corrupt_z[:, layer, head, :]
            diff = c - x
            raw_l2 = float(np.linalg.norm(diff, axis=-1).mean())
            c_norm = np.linalg.norm(c, axis=-1)
            c_safe = np.where(c_norm > 0, c_norm, 1.0)
            rel_l2 = float((np.linalg.norm(diff, axis=-1) / c_safe).mean())
            dot = (c * x).sum(axis=-1); x_norm = np.linalg.norm(x, axis=-1)
            denom = np.where(c_norm * x_norm > 0, c_norm * x_norm, 1.0)
            cos_dist = float((1.0 - dot / denom).mean())
            dm = diff.mean(axis=0, keepdims=True); ds = diff.std(axis=0, keepdims=True)
            ds_safe = np.where(ds > 0, ds, 1.0)
            std_l2 = float(np.linalg.norm((diff - dm) / ds_safe, axis=-1).mean())
            pert_rows.append({"layer": layer, "head": head,
                              "pert_raw_l2": raw_l2, "pert_relative_l2": rel_l2,
                              "pert_cosine_distance": cos_dist, "pert_standardized_l2": std_l2})
    write_csv(os.path.join(dir_20, "perturbation_per_head.csv"), pert_rows)

    imp_flat = head_importance.flatten()
    pert_flat = np.array([r["pert_raw_l2"] for r in pert_rows])
    imp_rows = [{"layer": l, "head": h, "imp_patching_loss": float(imp_flat[l*n_heads+h])}
                for l in range(n_layers) for h in range(n_heads)]
    write_csv(os.path.join(dir_20, "importance_per_head.csv"), imp_rows)

    imp_median = float(np.median(imp_flat))
    pert_median = float(np.median(pert_flat))
    cell_rows, cell_heads, cell_map = [], {"A":[],"B":[],"C":[],"D":[]}, {}
    for l in range(n_layers):
        for h in range(n_heads):
            idx = l * n_heads + h
            iv, pv = imp_flat[idx], pert_flat[idx]
            hi_i, hi_p = iv >= imp_median, pv >= pert_median
            c = "A" if hi_i and hi_p else "B" if hi_i else "C" if hi_p else "D"
            cell_heads[c].append((l, h)); cell_map[(l, h)] = c
            cell_rows.append({"layer": l, "head": h, "imp_value": float(iv),
                              "pert_value": float(pv), "hi_imp": hi_i, "hi_pert": hi_p,
                              "cell": c, "imp_median": imp_median, "pert_median": pert_median,
                              "in_gt_top10": (l, h) in gt_top10})
    write_csv(os.path.join(dir_20, "cell_classification.csv"), cell_rows)

    rho, _ = spearmanr(imp_flat, pert_flat)
    counts = {c: len(cell_heads[c]) for c in "ABCD"}
    gt_in_c = sum(1 for lh in cell_heads["C"] if lh in gt_top10)
    log(f"rho(imp, pert): {rho:.4f}")
    log(f"Cells: {counts}, GT in Cell C: {gt_in_c}/10")

    save_config(dir_20, n_heads=total_heads, rho=rho, imp_median=imp_median,
                pert_median=pert_median, cell_counts=counts, gt_in_cell_c=gt_in_c,
                elapsed_sec=round(time.time()-t0, 1))
    log(f"Phase 20 done ({time.time()-t0:.0f}s)")

    # =================================================================
    # PHASE 30: Patching
    # =================================================================
    log("=== Phase 30: Patching ===")
    t0 = time.time()
    dir_30 = ensure_dir(os.path.join(TASK_ROOT, "30_patching"))

    for c in ["A", "B"]:
        cell_heads[c].sort(key=lambda lh: imp_flat[lh[0]*n_heads+lh[1]], reverse=True)
    for c in ["C", "D"]:
        cell_heads[c].sort(key=lambda lh: pert_flat[lh[0]*n_heads+lh[1]])
    imp_ord = sorted(range(total_heads), key=lambda i: imp_flat[i])
    imp_ord_lh = [(i // n_heads, i % n_heads) for i in imp_ord]

    c_size, d_size = len(cell_heads["C"]), len(cell_heads["D"])
    k_cap = min(c_size, d_size)
    groups = {"Cell_A": cell_heads["A"], "Cell_B": cell_heads["B"],
              "Cell_C": cell_heads["C"], "Cell_D": cell_heads["D"][:k_cap],
              "Cell_D_full": cell_heads["D"], "imp_ordered": imp_ord_lh}

    def patch_group(heads_to_patch):
        by_layer = {}
        for l, h in heads_to_patch:
            by_layer.setdefault(l, []).append(h)
        hooks = []
        for l in sorted(by_layer):
            hn = utils.get_act_name("z", l)
            def mh(hs):
                def fn(act, hook, heads=hs):
                    for h in heads:
                        act[:, :, h, :] = corrupt_cache[hook.name][:, :, h, :]
                    return act
                return fn
            hooks.append((hn, mh(by_layer[l])))
        with torch.no_grad():
            return model.run_with_hooks(clean_toks, fwd_hooks=hooks)

    # Dose-response
    log(f"  Dose-response...")
    dose_rows = []
    for gname, glist in groups.items():
        for ratio in RATIOS_DOSE:
            k = max(1, math.ceil(total_heads * ratio))
            k_actual = min(k, len(glist))
            heads = glist[:k_actual]
            if not heads: continue
            pl = patch_group(heads)
            pls = compute_second_half_loss(pl, clean_toks, SEQ_HALF_LEN)
            for pi in range(N_PROMPTS):
                dose_rows.append({"group": gname, "ratio": ratio, "k": k,
                                  "k_actual": k_actual, "prompt_id": pi,
                                  "delta_loss_signed": float(pls[pi] - clean_losses[pi])})
        log(f"    {gname} done")

    write_csv(os.path.join(dir_30, "dose_response.csv"), dose_rows)

    dose_df = pd.DataFrame(dose_rows)
    cd_rows = []
    for ratio in RATIOS_DOSE:
        c_v = dose_df[(dose_df["group"]=="Cell_C") & (dose_df["ratio"]==ratio)]["delta_loss_signed"]
        d_v = dose_df[(dose_df["group"]=="Cell_D") & (dose_df["ratio"]==ratio)]["delta_loss_signed"]
        if len(c_v) == 0 or len(d_v) == 0: continue
        c_sub = dose_df[(dose_df["group"]=="Cell_C") & (dose_df["ratio"]==ratio)]
        d_sub = dose_df[(dose_df["group"]=="Cell_D") & (dose_df["ratio"]==ratio)]
        c_k = int(c_sub["k_actual"].iloc[0])
        d_k = int(d_sub["k_actual"].iloc[0])
        c_a, d_a = float(c_v.abs().mean()), float(d_v.abs().mean())
        c_b, d_b = float(abs(c_v.mean())), float(abs(d_v.mean()))
        cd_rows.append({"ratio": ratio, "cell_c_k": c_k, "cell_d_k": d_k,
                        "k_equal": c_k == d_k,
                        "cd_ratio_def_b": c_b/d_b if d_b > 0 else float("nan"),
                        "cd_ratio_def_a": c_a/d_a if d_a > 0 else float("nan"),
                        "cell_c_mean_def_b": c_b, "cell_d_mean_def_b": d_b,
                        "cell_c_mean_def_a": c_a, "cell_d_mean_def_a": d_a})
    write_csv(os.path.join(dir_30, "cd_ratio_summary.csv"), cd_rows)

    cd10 = [r for r in cd_rows if r["ratio"] == 0.10 and r["k_equal"]]
    if cd10:
        log(f"  C/D @10% Def B: {cd10[0]['cd_ratio_def_b']:.3f}, Def A: {cd10[0]['cd_ratio_def_a']:.3f}")

    # Grouped patching (1000 seeds)
    log(f"  Grouped patching ({N_RANDOM_SEEDS} random seeds)...")
    grouped_rows = []
    all_heads_list = [(l, h) for l in range(n_layers) for h in range(n_heads)]

    for k in DETECTOR_TOP_K:
        for det_name, det_col in [("induction", "induction_score"),
                                   ("prev_token", "prev_token_score"),
                                   ("dup_token", "dup_token_score")]:
            top_heads = list(det_df.sort_values(det_col, ascending=False).head(k).apply(
                lambda r: (int(r["layer"]), int(r["head"])), axis=1))
            pl = patch_group(top_heads)
            pls = compute_second_half_loss(pl, clean_toks, SEQ_HALF_LEN)
            md = float(np.mean([pls[i] - clean_losses[i] for i in range(N_PROMPTS)]))
            grouped_rows.append({"group": f"{det_name}_top{k}", "k": k, "seed": -1, "mean_delta_loss": md})

        cd_heads = cell_heads["D"][:k]
        if cd_heads:
            pl = patch_group(cd_heads)
            pls = compute_second_half_loss(pl, clean_toks, SEQ_HALF_LEN)
            md = float(np.mean([pls[i] - clean_losses[i] for i in range(N_PROMPTS)]))
            grouped_rows.append({"group": f"cell_d_top{k}", "k": k, "seed": -1, "mean_delta_loss": md})

    log(f"    Named groups done")

    for k in DETECTOR_TOP_K:
        for seed_i in range(N_RANDOM_SEEDS):
            rng = random.Random(SEED + seed_i + 10000)
            random_heads = rng.sample(all_heads_list, k)
            pl = patch_group(random_heads)
            pls = compute_second_half_loss(pl, clean_toks, SEQ_HALF_LEN)
            md = float(np.mean([pls[i] - clean_losses[i] for i in range(N_PROMPTS)]))
            grouped_rows.append({"group": f"random_top{k}", "k": k, "seed": seed_i, "mean_delta_loss": md})
        log(f"    random_top{k}: {N_RANDOM_SEEDS} seeds done")

    write_csv(os.path.join(dir_30, "grouped_patching.csv"), grouped_rows)

    gp_df = pd.DataFrame(grouped_rows)
    summary_rows = []
    for k in DETECTOR_TOP_K:
        re = gp_df[(gp_df["group"]==f"random_top{k}") & (gp_df["seed"]>=0)]["mean_delta_loss"]
        rm, rs = float(re.mean()), float(re.std())
        rlo, rhi = float(np.percentile(re, 2.5)), float(np.percentile(re, 97.5))
        for dn in ["induction", "prev_token", "dup_token", "cell_d"]:
            gn = f"{dn}_top{k}"
            named = gp_df[(gp_df["group"]==gn) & (gp_df["seed"]==-1)]
            if len(named) == 0: continue
            eff = float(named.iloc[0]["mean_delta_loss"])
            summary_rows.append({"group": gn, "k": k, "effect": eff,
                                 "random_mean": rm, "random_sd": rs,
                                 "random_ci_lo": rlo, "random_ci_hi": rhi,
                                 "ratio_vs_random": eff/rm if rm > 0 else float("nan"),
                                 "n_random_seeds": N_RANDOM_SEEDS})
    write_csv(os.path.join(dir_30, "grouped_patching_summary.csv"), summary_rows)

    for r in summary_rows:
        if r["group"].startswith("induction"):
            log(f"    {r['group']}: effect={r['effect']:.3f}, "
                f"random={r['random_mean']:.4f}+/-{r['random_sd']:.4f}, "
                f"ratio={r['ratio_vs_random']:.1f}x")

    save_config(dir_30, n_ratios=len(RATIOS_DOSE), n_prompts=N_PROMPTS,
                n_random_seeds=N_RANDOM_SEEDS, elapsed_sec=round(time.time()-t0, 1))
    log(f"Phase 30 done ({time.time()-t0:.0f}s)")

    # =================================================================
    # PHASE 35: Recoverability
    # =================================================================
    log("=== Phase 35: Recoverability ===")
    t0 = time.time()
    dir_35 = ensure_dir(os.path.join(TASK_ROOT, "35_recoverability"))

    y_true = np.array([1 if (i//n_heads, i%n_heads) in gt_top10 else 0 for i in range(total_heads)])
    auroc = roc_auc_score(y_true, imp_flat)
    auprc = average_precision_score(y_true, imp_flat)
    log(f"AUROC (top-10): {auroc:.3f}, AUPRC: {auprc:.3f}")

    gt_cell_rows = []
    for lh in sorted(gt_top10):
        gt_cell_rows.append({"layer": lh[0], "head": lh[1], "cell": cell_map[lh],
                             "importance": float(imp_flat[lh[0]*n_heads+lh[1]]),
                             "perturbation": float(pert_flat[lh[0]*n_heads+lh[1]])})
    write_csv(os.path.join(dir_35, "gt_cell_distribution.csv"), gt_cell_rows)

    recovery_rows = []
    baselines = {
        "standard": [lh for lh in all_heads_list if imp_flat[lh[0]*n_heads+lh[1]] < imp_median],
        "cell_d": [lh for lh in all_heads_list if cell_map[lh] == "D"],
        "imp_ordered": [(i//n_heads, i%n_heads) for i in imp_ord[:total_heads//2]],
    }
    for bl_name, bl_pool in baselines.items():
        bl_effects = [imp_flat[lh[0]*n_heads+lh[1]] for lh in bl_pool]
        if not bl_effects: continue
        for tr_name, tr_fn in THRESHOLD_RULES.items():
            threshold = tr_fn(bl_effects)
            circuit = set(lh for lh in all_heads_list if imp_flat[lh[0]*n_heads+lh[1]] > threshold)
            tp = len(circuit & gt_top10)
            fp = len(circuit - gt_top10)
            fn = len(gt_top10 - circuit)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / len(gt_top10) if gt_top10 else 0
            recovery_rows.append({"baseline": bl_name, "threshold_rule": tr_name,
                                  "threshold": round(threshold, 6), "n_selected": len(circuit),
                                  "tp": tp, "fp": fp, "fn": fn,
                                  "precision": round(prec, 4), "recall": round(rec, 4),
                                  "auroc": round(auroc, 4), "auprc": round(auprc, 4)})
            if tr_name == "mean_plus_2sd":
                log(f"  {bl_name:>15} {tr_name}: thresh={threshold:.4f}, recall={tp}/{len(gt_top10)}")
    write_csv(os.path.join(dir_35, "threshold_recovery.csv"), recovery_rows)
    save_config(dir_35, auroc=auroc, auprc=auprc, n_gt=len(gt_top10),
                elapsed_sec=round(time.time()-t0, 1))
    log(f"Phase 35 done ({time.time()-t0:.0f}s)")

    # =================================================================
    # PHASE 90: Post-hoc
    # =================================================================
    log("=== Phase 90: Post-hoc ===")
    t0 = time.time()
    dir_90 = ensure_dir(os.path.join(TASK_ROOT, "90_post_hoc"))

    from sklearn.linear_model import LinearRegression
    r2_pert = LinearRegression().fit(pert_flat.reshape(-1,1), imp_flat).score(pert_flat.reshape(-1,1), imp_flat)

    c_eff = np.abs(imp_flat[[i for i in range(total_heads) if cell_map[(i//n_heads,i%n_heads)]=="C"]])
    d_eff = np.abs(imp_flat[[i for i in range(total_heads) if cell_map[(i//n_heads,i%n_heads)]=="D"]])
    stat, p_mw = mannwhitneyu(c_eff, d_eff, alternative="greater")

    summary = {"rho": rho, "auroc": auroc, "auprc": auprc, "r2_pert": r2_pert,
               "n_gt": len(gt_top10), "clean_loss_mean": clean_loss_mean,
               "matched_head_p": float(p_mw),
               "matched_head_ratio": float(c_eff.mean()/d_eff.mean()) if d_eff.mean()>0 else 0,
               "cell_counts": counts, "gt_in_cell_c": gt_in_c,
               "detector_method": "detect_head_library"}
    with open(os.path.join(dir_90, "posthoc_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    save_config(dir_90, elapsed_sec=round(time.time()-t0, 1))
    log(f"Phase 90 done")

    # Summary
    total_elapsed = time.time() - t0_total
    log(f"\n{'='*60}")
    log(f"Induction Pipeline Complete -- {total_elapsed:.0f}s")
    log(f"{'='*60}")
    log(f"Detector: detect_head() library (canonical induction heads)")
    log(f"rho(imp, pert): {rho:.4f}")
    log(f"AUROC (top-10): {auroc:.3f}")
    log(f"Cells: {counts}, GT in Cell C: {gt_in_c}/10")
    if cd10:
        log(f"C/D @10% Def B: {cd10[0]['cd_ratio_def_b']:.3f}")
    for r in summary_rows:
        if "induction_top5" == r["group"]:
            log(f"Grouped top-5: {r['ratio_vs_random']:.1f}x "
                f"(random: {r['random_mean']:.4f}+/-{r['random_sd']:.4f})")
    log(f"Output: {TASK_ROOT}")


if __name__ == "__main__":
    main()

"""
KN Robustness Run: 8 seeds × 3 fact counts = 24 configurations.
Reuses pilot logic. Produces aggregate summaries for Appendix.
"""
import json, glob, os, random, time, csv, itertools
import numpy as np
import torch
import pandas as pd
from collections import Counter
from scipy.stats import spearmanr
from transformers import BertForMaskedLM, BertTokenizer

t0 = time.time()
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

BASE = "${WCC_ROOT}"
TASK_DIR = f"{BASE}/tasks/kn_fact_edit_robustness"
DEVICE = "cuda"
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]
FACT_COUNTS = [100, 200, 300]
RATIOS = [0.001, 0.003, 0.005, 0.01, 0.03, 0.05]

# Create folder structure
for d in ["00_manifest", "10_collection", "80_aggregate", "90_post_hoc"]:
    os.makedirs(f"{TASK_DIR}/{d}", exist_ok=True)

log("Loading model")
tok = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased").to(DEVICE).eval()
n_layers = model.config.num_hidden_layers
ffn_dim = model.config.intermediate_size
total_neurons = n_layers * ffn_dim

# Build master fact list (enough for 300)
log("Building master fact list from PARAREL")
pattern_dir = f"{BASE}/data/pararel_raw/data/pattern_data/graphs_json"
trex_dir = f"{BASE}/data/pararel_raw/data/trex_lms_vocab"

all_eligible = []
rel_patterns = {}
for pf in sorted(glob.glob(f"{pattern_dir}/*.jsonl")):
    rel_id = os.path.basename(pf).replace(".jsonl", "")
    trex_f = f"{trex_dir}/{rel_id}.jsonl"
    if not os.path.exists(trex_f): continue
    with open(pf) as f:
        patterns = [json.loads(l) for l in f]
    with open(trex_f) as f:
        facts = [json.loads(l) for l in f]
    good = [f for f in facts if len(tok.tokenize(f["obj_label"].lower())) == 1]
    if len(good) >= 20 and len(patterns) >= 2:
        rel_patterns[rel_id] = patterns
        for f in good:
            f["rel_id"] = rel_id
            all_eligible.append(f)

# Build relation subject pools
rel_subjects = {}
for f in all_eligible:
    rel_subjects.setdefault(f["rel_id"], []).append(f["sub_label"])

log(f"  {len(all_eligible)} eligible facts, {len(rel_patterns)} relations")

# Save master
master_facts = []
rng_master = random.Random(42)
rng_master.shuffle(all_eligible)
rel_ct = {}
for f in all_eligible:
    r = f["rel_id"]
    if rel_ct.get(r, 0) >= 15:  # max 15 per relation for 300 total
        continue
    master_facts.append(f)
    rel_ct[r] = rel_ct.get(r, 0) + 1
    if len(master_facts) >= 400:  # buffer
        break

log(f"  Master: {len(master_facts)} facts")
pd.DataFrame([{"fact_idx": i, "rel_id": f["rel_id"], "sub_label": f["sub_label"],
               "obj_label": f["obj_label"]} for i, f in enumerate(master_facts)]).to_csv(
    f"{TASK_DIR}/10_collection/facts_master.csv", index=False)

# === Helper functions ===

def build_prompts(facts_subset, seed):
    """Build clean/corrupt prompts for a fact subset."""
    rng = random.Random(seed)
    prompts_clean, prompts_corrupt, target_ids, fact_ids = [], [], [], []
    for i, fact in enumerate(facts_subset):
        r = fact["rel_id"]
        template = rel_patterns[r][0]["pattern"]
        candidates = [s for s in rel_subjects[r] if s != fact["sub_label"]]
        corrupt_sub = rng.choice(candidates) if candidates else fact["sub_label"]
        clean_p = template.replace("[X]", fact["sub_label"]).replace("[Y]", "[MASK]")
        corrupt_p = template.replace("[X]", corrupt_sub).replace("[Y]", "[MASK]")
        obj_tok = tok.tokenize(fact["obj_label"].lower())
        tid = tok.convert_tokens_to_ids(obj_tok[0])
        prompts_clean.append(clean_p)
        prompts_corrupt.append(corrupt_p)
        target_ids.append(tid)
        fact_ids.append(i)
    return prompts_clean, prompts_corrupt, target_ids, fact_ids

def encode_and_get_masks(texts):
    enc = tok(texts, padding=True, return_tensors="pt").to(DEVICE)
    mask_pos = {}
    for row in (enc.input_ids == tok.mask_token_id).nonzero(as_tuple=False):
        mask_pos[row[0].item()] = row[1].item()
    return enc, mask_pos

def run_config(seed, n_facts):
    """Run full Phase 20+30+35+36 for one (seed, n_facts) config."""
    rng = random.Random(seed)
    subset = master_facts[:n_facts] if seed == 0 else rng.sample(master_facts, min(n_facts, len(master_facts)))

    clean_texts, corrupt_texts, tgt_ids, fids = build_prompts(subset, seed)
    clean_enc, clean_mp = encode_and_get_masks(clean_texts)
    corrupt_enc, corrupt_mp = encode_and_get_masks(corrupt_texts)
    target_ids = torch.tensor(tgt_ids, device=DEVICE)

    # Phase 20: activations (batched)
    clean_acts = torch.zeros(len(clean_texts), n_layers, ffn_dim, device=DEVICE)
    corrupt_acts = torch.zeros(len(corrupt_texts), n_layers, ffn_dim, device=DEVICE)
    BATCH = 64

    for b_start in range(0, len(clean_texts), BATCH):
        b_end = min(b_start + BATCH, len(clean_texts))
        # Clean
        b_enc = tok(clean_texts[b_start:b_end], padding=True, return_tensors="pt").to(DEVICE)
        b_mp = {}
        for row in (b_enc.input_ids == tok.mask_token_id).nonzero(as_tuple=False):
            b_mp[row[0].item()] = row[1].item()
        c_ints = {}
        def cap(s, idx):
            def fn(m, i, o): s[idx] = o.detach()
            return fn
        hooks = [model.bert.encoder.layer[li].intermediate.register_forward_hook(cap(c_ints, li)) for li in range(n_layers)]
        with torch.no_grad():
            c_logits = model(**b_enc).logits
        for li in range(n_layers):
            for i in range(b_end - b_start):
                clean_acts[b_start+i, li] = c_ints[li][i, b_mp.get(i, 0)]
        for h in hooks: h.remove()
        # Corrupt
        b_enc_x = tok(corrupt_texts[b_start:b_end], padding=True, return_tensors="pt").to(DEVICE)
        b_mp_x = {}
        for row in (b_enc_x.input_ids == tok.mask_token_id).nonzero(as_tuple=False):
            b_mp_x[row[0].item()] = row[1].item()
        x_ints = {}
        hooks = [model.bert.encoder.layer[li].intermediate.register_forward_hook(cap(x_ints, li)) for li in range(n_layers)]
        with torch.no_grad(): model(**b_enc_x)
        for li in range(n_layers):
            for i in range(b_end - b_start):
                corrupt_acts[b_start+i, li] = x_ints[li][i, b_mp_x.get(i, 0)]
        for h in hooks: h.remove()
        del c_ints, x_ints, b_enc, b_enc_x
        torch.cuda.empty_cache()
    # Compute baseline target logits (no suppression)
    clean_enc, clean_mp = encode_and_get_masks(clean_texts)
    with torch.no_grad():
        full_logits = model(**clean_enc).logits
    clean_tgt_logits = torch.tensor([full_logits[i, clean_mp.get(i,0), target_ids[i]].item()
                                     for i in range(len(clean_texts))], device=DEVICE)
    del full_logits

    pert = (clean_acts - corrupt_acts).abs().mean(dim=0).cpu().numpy()

    # Gradient importance (batched to avoid OOM for large n_facts)
    imp_s = torch.zeros(n_layers, ffn_dim, device=DEVICE)
    BATCH = 64
    n_prompts = len(clean_texts)
    for b_start in range(0, n_prompts, BATCH):
        b_end = min(b_start + BATCH, n_prompts)
        b_texts = clean_texts[b_start:b_end]
        b_enc = tok(b_texts, padding=True, return_tensors="pt").to(DEVICE)
        b_mp = {}
        for row in (b_enc.input_ids == tok.mask_token_id).nonzero(as_tuple=False):
            b_mp[row[0].item()] = row[1].item()
        b_tgt = target_ids[b_start:b_end]

        model.zero_grad()
        grad_ints = {}
        def mk_grad(s, idx):
            def fn(m, i, o): o.retain_grad(); s[idx] = o
            return fn
        hooks_g = [model.bert.encoder.layer[li].intermediate.register_forward_hook(mk_grad(grad_ints, li)) for li in range(n_layers)]
        logits_b = model(**b_enc).logits
        tgt_sum = sum(logits_b[i, b_mp.get(i,0), b_tgt[i]] for i in range(b_end - b_start))
        tgt_sum.backward()

        for li in range(n_layers):
            act, grad = grad_ints[li], grad_ints[li].grad
            for i in range(b_end - b_start):
                imp_s[li] += act[i, b_mp.get(i,0)] * grad[i, b_mp.get(i,0)]
        for h in hooks_g: h.remove()
        del grad_ints, logits_b, b_enc
        torch.cuda.empty_cache()
    imp_s /= n_prompts

    imp_s_np = imp_s.detach().cpu().numpy()
    imp_a_np = np.abs(imp_s_np)

    # Cell classification
    med_s = np.median(imp_s_np.flatten())
    med_a = np.median(imp_a_np.flatten())
    med_p = np.median(pert.flatten())

    def classify(imp_val, pert_val, med_i, med_p):
        hi_i = imp_val >= med_i
        hi_p = pert_val >= med_p
        if hi_i and hi_p: return "A"
        if hi_i: return "B"
        if hi_p: return "C"
        return "D"

    cells_s = np.array([[classify(imp_s_np[l,n], pert[l,n], med_s, med_p) for n in range(ffn_dim)] for l in range(n_layers)])

    cell_counts = Counter(cells_s.flatten())
    rho_s, _ = spearmanr(imp_s_np.flatten(), pert.flatten())
    rho_a, _ = spearmanr(imp_a_np.flatten(), pert.flatten())
    pert_zero_frac = (pert < 1e-8).mean()

    # Phase 30: dose-response
    cell_C = [(l, n) for l in range(n_layers) for n in range(ffn_dim) if cells_s[l,n] == "C"]
    cell_D = [(l, n) for l in range(n_layers) for n in range(ffn_dim) if cells_s[l,n] == "D"]
    all_low = cell_C + cell_D
    im = {(l,n): imp_s_np[l,n] for l in range(n_layers) for n in range(ffn_dim)}
    pm = {(l,n): pert[l,n] for l in range(n_layers) for n in range(ffn_dim)}

    orderings = {
        "C_imp_asc": sorted(cell_C, key=lambda x: im[x]),
        "D_pert_asc": sorted(cell_D, key=lambda x: pm[x]),
        "D_imp_asc": sorted(cell_D, key=lambda x: im[x]),
        "std_imp_asc": sorted(all_low, key=lambda x: im[x]),
    }

    # Baseline target logits (already computed above as clean_tgt_logits)

    def suppress(neurons):
        by_layer = {}
        for (l,n) in neurons: by_layer.setdefault(l, []).append(n)
        hooks = []
        for li, ns in by_layer.items():
            idx = torch.tensor(ns, device=DEVICE)
            def mk(indices):
                def fn(m, i, o): o[:,:,indices]=0; return o
                return fn
            hooks.append(model.bert.encoder.layer[li].intermediate.register_forward_hook(mk(idx)))
        with torch.no_grad(): lg = model(**clean_enc).logits
        for h in hooks: h.remove()
        return torch.tensor([lg[i, clean_mp.get(i,0), target_ids[i]].item() for i in range(len(clean_texts))], device=DEVICE)

    cd_results = {}
    for ratio in RATIOS:
        k = max(1, int(np.ceil(total_neurons * ratio)))
        for gname, ordered in orderings.items():
            k_act = min(k, len(ordered))
            patched = suppress(ordered[:k_act])
            delta = (clean_tgt_logits - patched).abs().mean().item()
            cd_results[(ratio, gname)] = delta

    # Phase 35: threshold
    std_pool_imp = imp_s_np.flatten()[cells_s.flatten() != "A"]
    std_pool_imp = np.array([imp_s_np[l,n] for l in range(n_layers) for n in range(ffn_dim) if cells_s[l,n] in ("C","D")])
    cd_pool_imp = np.array([imp_s_np[l,n] for l in range(n_layers) for n in range(ffn_dim) if cells_s[l,n] == "D"])

    std_thresh = std_pool_imp.mean() + 2 * std_pool_imp.std()
    cd_thresh = cd_pool_imp.mean() + 2 * cd_pool_imp.std()
    inflation = std_thresh / cd_thresh if cd_thresh != 0 else float("inf")

    std_selected = int((imp_s_np.flatten() > std_thresh).sum())
    cd_selected = int((imp_s_np.flatten() > cd_thresh).sum())

    # Phase 36: editing (threshold-based)
    std_neurons = [(l,n) for l in range(n_layers) for n in range(ffn_dim) if imp_s_np[l,n] > std_thresh]
    cd_neurons = [(l,n) for l in range(n_layers) for n in range(ffn_dim) if imp_s_np[l,n] > cd_thresh]

    std_edited = suppress(std_neurons) if std_neurons else clean_tgt_logits
    cd_edited = suppress(cd_neurons) if cd_neurons else clean_tgt_logits

    std_drop = (clean_tgt_logits - std_edited).mean().item()
    cd_drop = (clean_tgt_logits - cd_edited).mean().item()
    std_success = ((clean_tgt_logits - std_edited) > 1.0).float().mean().item()
    cd_success = ((clean_tgt_logits - cd_edited) > 1.0).float().mean().item()

    return {
        "seed": seed, "n_facts": n_facts,
        "cell_A": int(cell_counts.get("A", 0)), "cell_B": int(cell_counts.get("B", 0)),
        "cell_C": int(cell_counts.get("C", 0)), "cell_D": int(cell_counts.get("D", 0)),
        "rho_signed": round(float(rho_s), 4), "rho_abs": round(float(rho_a), 4),
        "pert_zero_fraction": round(float(pert_zero_frac), 4),
        "C_at_1pct": cd_results.get((0.01, "C_imp_asc"), 0),
        "D_at_1pct": cd_results.get((0.01, "D_pert_asc"), 0),
        "C_at_3pct": cd_results.get((0.01*3, "C_imp_asc"), 0),
        "D_at_3pct": cd_results.get((0.01*3, "D_pert_asc"), 0),
        "C_at_5pct": cd_results.get((0.05, "C_imp_asc"), 0),
        "D_at_5pct": cd_results.get((0.05, "D_pert_asc"), 0),
        "std_thresh": float(std_thresh), "cd_thresh": float(cd_thresh),
        "inflation": round(float(inflation), 2),
        "std_selected": std_selected, "cd_selected": cd_selected,
        "std_target_success": round(float(std_success), 3),
        "cd_target_success": round(float(cd_success), 3),
        "std_mean_drop": round(float(std_drop), 3),
        "cd_mean_drop": round(float(cd_drop), 3),
    }

# === Main loop ===
log(f"\nRunning {len(SEEDS)} seeds × {len(FACT_COUNTS)} fact counts = {len(SEEDS)*len(FACT_COUNTS)} configs")
all_results = []
for seed in SEEDS:
    for n_facts in FACT_COUNTS:
        r = run_config(seed, n_facts)
        all_results.append(r)
        cd_ratio = r["C_at_1pct"] / max(r["D_at_1pct"], 1e-9)
        log(f"  seed={seed} n={n_facts}: C/D@1%={cd_ratio:.1f}x, inflation={r['inflation']:.1f}x, "
            f"std_success={r['std_target_success']:.2f}, cd_success={r['cd_target_success']:.2f}")

# === Aggregate ===
log("\n=== AGGREGATE RESULTS ===")
df = pd.DataFrame(all_results)
df.to_csv(f"{TASK_DIR}/80_aggregate/aggregate_full.csv", index=False)

# Gate summaries
log(f"\nGate 1 (C > D @1%): {(df['C_at_1pct'] > df['D_at_1pct']).sum()}/{len(df)} configs pass")
log(f"Gate 2 (inflation > 1.2): {(df['inflation'] > 1.2).sum()}/{len(df)} configs pass")
log(f"Gate 3 (cd > std success): {(df['cd_target_success'] > df['std_target_success']).sum()}/{len(df)} configs")

log(f"\nC/D @1% ratio: mean={df['C_at_1pct'].mean()/max(df['D_at_1pct'].mean(),1e-9):.1f}x, "
    f"range=[{(df['C_at_1pct']/df['D_at_1pct'].clip(1e-9)).min():.1f}x, {(df['C_at_1pct']/df['D_at_1pct'].clip(1e-9)).max():.1f}x]")
log(f"Inflation: mean={df['inflation'].mean():.1f}x, range=[{df['inflation'].min():.1f}x, {df['inflation'].max():.1f}x]")
log(f"Std success: mean={df['std_target_success'].mean():.3f}, range=[{df['std_target_success'].min():.3f}, {df['std_target_success'].max():.3f}]")
log(f"CD success: mean={df['cd_target_success'].mean():.3f}")
log(f"rho(signed,pert): mean={df['rho_signed'].mean():.4f}")
log(f"rho(|imp|,pert): mean={df['rho_abs'].mean():.4f}")

# Summary JSON
with open(f"{TASK_DIR}/90_post_hoc/robustness_summary.json", "w") as f:
    json.dump({
        "n_configs": len(df),
        "seeds": SEEDS, "fact_counts": FACT_COUNTS,
        "gate1_cd_pass_rate": float((df["C_at_1pct"] > df["D_at_1pct"]).mean()),
        "gate2_inflation_pass_rate": float((df["inflation"] > 1.2).mean()),
        "gate3_utility_pass_rate": float((df["cd_target_success"] > df["std_target_success"]).mean()),
        "mean_cd_ratio_1pct": round(float(df["C_at_1pct"].mean() / max(df["D_at_1pct"].mean(), 1e-9)), 2),
        "mean_inflation": round(float(df["inflation"].mean()), 2),
        "mean_std_success": round(float(df["std_target_success"].mean()), 3),
        "mean_cd_success": round(float(df["cd_target_success"].mean()), 3),
        "mean_rho_signed": round(float(df["rho_signed"].mean()), 4),
        "mean_rho_abs": round(float(df["rho_abs"].mean()), 4),
        "conclusion": "Gate 1+2 stable, Gate 3 saturated (editing not threshold-limited)",
        "recommended_placement": "appendix_only",
    }, f, indent=2)

log(f"\nTotal: {(time.time()-t0)/60:.1f} min")
log("ROBUSTNESS RUN COMPLETE.")

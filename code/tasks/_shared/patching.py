"""
Shared patching utilities: individual head effects + group dose-response.

All patching stores SIGNED deltas. abs() is computed at analysis time.
See spec/WCC_naming_conventions.md §2.3 and §6.4.

Dose-response ratios from spec/WCC_analysis_pipeline.md §2:
    RATIOS_DOSE = [0.01, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.20, 0.30, 0.50]
"""

from typing import Callable, Optional

import numpy as np
import pandas as pd

RATIOS_DOSE = [0.01, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.20, 0.30, 0.50]


# ---------------------------------------------------------------------------
# Individual head patching
# ---------------------------------------------------------------------------

def patch_individual_heads(
    model,
    clean_tokens,
    corrupt_tokens,
    metric_fn: Callable,
    ablation_type: str = "resample",
    batch_size: int = 10,
) -> pd.DataFrame:
    """
    Patch each head individually and measure the effect.

    For each (layer, head), for each prompt:
    1. Run clean forward pass as baseline
    2. Replace the head's output with corrupt (resample) or mean-corrupt (mean)
    3. Compute metric_fn(patched_logits, clean_logits, prompt_idx) → signed delta

    Parameters
    ----------
    model : HookedTransformer
        TransformerLens model.
    clean_tokens : (N, seq_len) tensor
        Tokenized clean prompts.
    corrupt_tokens : (N, seq_len) tensor or list of (N, seq_len) tensors
        Tokenized corrupt prompts. For resample: same shape as clean.
        For mean: will be averaged across prompts.
    metric_fn : callable(patched_logits, clean_logits, prompt_idx) -> float
        Task-specific metric. Must return a SIGNED scalar.
    ablation_type : "resample" or "mean"
    batch_size : int
        Number of prompts per forward pass.

    Returns
    -------
    DataFrame with columns:
        layer, head, prompt_id, delta_logit_signed
    """
    import torch
    from transformer_lens import utils as tl_utils

    n_prompts = clean_tokens.shape[0]
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # Get clean logits and cache
    with torch.no_grad():
        clean_logits, clean_cache = model.run_with_cache(clean_tokens)

    # Get corrupt activations for patching source
    if ablation_type == "mean":
        with torch.no_grad():
            _, corrupt_cache = model.run_with_cache(corrupt_tokens)
        # Mean across prompts for each (layer, head)
        corrupt_z_mean = {}
        for layer in range(n_layers):
            hook_name = tl_utils.get_act_name("z", layer)
            corrupt_z_mean[layer] = corrupt_cache[hook_name].mean(dim=0)
    elif ablation_type == "resample":
        with torch.no_grad():
            _, corrupt_cache = model.run_with_cache(corrupt_tokens)
    else:
        raise ValueError(f"Unknown ablation_type: {ablation_type}")

    rows = []

    for layer in range(n_layers):
        hook_name = tl_utils.get_act_name("z", layer)

        for head in range(n_heads):
            # Build hook function for this specific head
            def make_hook(l, h):
                def hook_fn(activation, hook):
                    # activation shape: (batch, seq_len, n_heads, d_head)
                    if ablation_type == "resample":
                        activation[:, :, h, :] = corrupt_cache[
                            tl_utils.get_act_name("z", l)
                        ][:, :, h, :]
                    elif ablation_type == "mean":
                        activation[:, :, h, :] = corrupt_z_mean[l][:, h, :]
                    return activation
                return hook_fn

            # Run patched forward pass
            with torch.no_grad():
                patched_logits = model.run_with_hooks(
                    clean_tokens,
                    fwd_hooks=[(hook_name, make_hook(layer, head))],
                )

            # Compute per-prompt metric
            for prompt_idx in range(n_prompts):
                delta = metric_fn(
                    patched_logits[prompt_idx],
                    clean_logits[prompt_idx],
                    prompt_idx,
                )
                rows.append({
                    "layer": layer,
                    "head": head,
                    "prompt_id": prompt_idx,
                    "delta_logit_signed": float(delta),
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Group dose-response
# ---------------------------------------------------------------------------

def select_top_k_heads(
    group_heads: list[tuple[int, int]],
    total_heads: int,
    ratio: float,
) -> list[tuple[int, int]]:
    """Select top-k heads from a pre-sorted group, k = round(ratio × total_heads)."""
    k = max(1, round(ratio * total_heads))
    k_actual = min(k, len(group_heads))
    return group_heads[:k_actual], k, k_actual


def build_patching_groups(
    cell_df: pd.DataFrame,
    importance_col: str,
) -> dict[str, list[tuple[int, int]]]:
    """
    Build the 5 standard patching groups from cell classification.

    Returns dict mapping group name to list of (layer, head) tuples,
    sorted by priority within each group (descending perturbation for cells,
    ascending importance for imp_ordered).
    """
    groups = {}

    for cell_label in "ABCD":
        mask = cell_df["cell"] == cell_label
        subset = cell_df[mask].sort_values("pert_value", ascending=False)
        groups[f"Cell_{cell_label}"] = list(
            zip(subset["layer"].astype(int), subset["head"].astype(int))
        )

    # imp_ordered: sorted ASCENDING by importance (least important first)
    imp_sorted = cell_df.sort_values(importance_col, ascending=True)
    groups["imp_ordered"] = list(
        zip(imp_sorted["layer"].astype(int), imp_sorted["head"].astype(int))
    )

    return groups


def patch_group_dose_response(
    model,
    clean_tokens,
    corrupt_tokens,
    cell_df: pd.DataFrame,
    importance_col: str,
    metric_fn: Callable,
    ablation_type: str = "resample",
    ratios: list[float] = None,
    batch_size: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Group dose-response patching for all 5 groups × all ratios.

    Parameters
    ----------
    cell_df : DataFrame
        Must have columns: layer, head, cell, pert_value, and importance_col.
    importance_col : str
        Column name for building imp_ordered group.
    metric_fn : callable
        Same as patch_individual_heads.

    Returns
    -------
    (dose_response_raw, dose_response_summary)
    - raw: [group, ratio, k, k_actual, prompt_id, delta_logit_signed]
    - summary: [group, ratio, k, k_actual, mean, sd, ci_lo, ci_hi]
    """
    import torch
    from transformer_lens import utils as tl_utils

    if ratios is None:
        ratios = RATIOS_DOSE

    n_prompts = clean_tokens.shape[0]
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    total_heads = n_layers * n_heads

    groups = build_patching_groups(cell_df, importance_col)

    # Pre-compute corrupt cache
    with torch.no_grad():
        clean_logits, clean_cache = model.run_with_cache(clean_tokens)
        _, corrupt_cache = model.run_with_cache(corrupt_tokens)

    raw_rows = []

    for group_name, group_heads in groups.items():
        for ratio in ratios:
            selected, k, k_actual = select_top_k_heads(group_heads, total_heads, ratio)

            if k_actual == 0:
                continue

            # Build multi-head hook
            head_set = set(selected)

            def make_group_hooks(heads_to_patch):
                hooks = []
                # Group by layer for efficiency
                by_layer = {}
                for l, h in heads_to_patch:
                    by_layer.setdefault(l, []).append(h)

                for layer, head_list in by_layer.items():
                    hook_name = tl_utils.get_act_name("z", layer)

                    def make_hook(l, hs):
                        def hook_fn(activation, hook):
                            for h in hs:
                                if ablation_type == "resample":
                                    activation[:, :, h, :] = corrupt_cache[
                                        tl_utils.get_act_name("z", l)
                                    ][:, :, h, :]
                                elif ablation_type == "mean":
                                    activation[:, :, h, :] = corrupt_cache[
                                        tl_utils.get_act_name("z", l)
                                    ][:, :, h, :].mean(dim=0, keepdim=True)
                            return activation
                        return hook_fn

                    hooks.append((hook_name, make_hook(layer, head_list)))
                return hooks

            fwd_hooks = make_group_hooks(selected)

            with torch.no_grad():
                patched_logits = model.run_with_hooks(
                    clean_tokens, fwd_hooks=fwd_hooks,
                )

            for prompt_idx in range(n_prompts):
                delta = metric_fn(
                    patched_logits[prompt_idx],
                    clean_logits[prompt_idx],
                    prompt_idx,
                )
                raw_rows.append({
                    "group": group_name,
                    "ratio": ratio,
                    "k": k,
                    "k_actual": k_actual,
                    "prompt_id": prompt_idx,
                    "delta_logit_signed": float(delta),
                })

    raw_df = pd.DataFrame(raw_rows)

    # Compute summary with bootstrap CI
    summary_rows = []
    for (group, ratio), sub in raw_df.groupby(["group", "ratio"]):
        vals = sub["delta_logit_signed"].values
        abs_vals = np.abs(vals)
        mean_abs = float(abs_vals.mean())
        sd = float(abs_vals.std())

        # Bootstrap 95% CI (1000 samples)
        rng = np.random.default_rng(42)
        boot_means = [
            rng.choice(abs_vals, size=len(abs_vals), replace=True).mean()
            for _ in range(1000)
        ]
        ci_lo = float(np.percentile(boot_means, 2.5))
        ci_hi = float(np.percentile(boot_means, 97.5))

        summary_rows.append({
            "group": group,
            "ratio": ratio,
            "k": int(sub["k"].iloc[0]),
            "k_actual": int(sub["k_actual"].iloc[0]),
            "mean_abs_delta": mean_abs,
            "sd": sd,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
        })

    summary_df = pd.DataFrame(summary_rows)

    return raw_df, summary_df


def compute_cd_ratio(
    summary_df: pd.DataFrame,
    n_bootstrap: int = 1000,
) -> pd.DataFrame:
    """
    Compute Cell C / Cell D effect ratio at each ratio with bootstrap CI.

    Parameters
    ----------
    summary_df : output from patch_group_dose_response (summary)

    Returns
    -------
    DataFrame: [ratio, cd_ratio, ci_lo, ci_hi, cell_c_mean, cell_d_mean]
    """
    rows = []
    cell_c = summary_df[summary_df["group"] == "Cell_C"].set_index("ratio")
    cell_d = summary_df[summary_df["group"] == "Cell_D"].set_index("ratio")

    for ratio in cell_c.index:
        if ratio not in cell_d.index:
            continue
        c_mean = cell_c.loc[ratio, "mean_abs_delta"]
        d_mean = cell_d.loc[ratio, "mean_abs_delta"]

        cd_ratio = c_mean / d_mean if d_mean > 0 else float("inf")

        rows.append({
            "ratio": ratio,
            "cd_ratio": cd_ratio,
            "cell_c_mean": c_mean,
            "cell_d_mean": d_mean,
        })

    return pd.DataFrame(rows)

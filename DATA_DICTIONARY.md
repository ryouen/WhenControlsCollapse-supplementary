# DATA_DICTIONARY — When Controls Collapse Supplementary CSVs

This document describes every CSV shipped under `supplementary/data/`. For each file: source pipeline, columns, units, and which paper Appendix table / figure it backs.

Models referenced as `{m}` ∈ `{llama-3.1-8b-instruct, gemma-2-9b-it, qwen-2.5-7b-instruct, qwen-2.5-14b-instruct, gemma-3-27b-it, olmo-2-13b-instruct, gpt-j-6b-fp32, llama-3.1-70b-instruct-4bit}` (8 models; 70B is NF4 stress arm).

## 1. Novel-word per-model — `data/novel_word/{m}/`

### `10_collection/`

- **`word_pairs.csv`** — 10 novel-word pairs per model (algorithmically generated from `WORD_SEED = 2026`; per-model independent generation, see Appendix A §A.1).
  Columns: `pair_id, word_a, word_b, len_a, len_b, freq_a, freq_b, levenshtein, bpe_a, bpe_b`.
  Reproducibility: deterministic given seed + model tokenizer.

- **`trial_definitions.csv`** — 80 trials (10 pair × 4 crel × 2 attr_dim).
  Columns: `trial_id, pair_id, crel, attr_dim, attribute, target_word, incorrect_word, prompt`.
  Backs Appendix A §A.2 trial grid.

- **`model_responses.csv`** — clean-prompt model output per trial (no patching).
  Columns: `trial_id, target_logit, incorrect_logit, gap, top1_token, top1_logit, argmax_correct, dplus`.
  Backs Appendix A.5 D+ validity (D+ = `target_logit > incorrect_logit`).

- **`answer_token_diagnostic.csv`** — first-subtoken id sanity-check per trial.

### `11_position_bias/` — Appendix A §A.5 position-swap audit

- `position_bias_responses.csv`: per-trial response under {word_a, word_b} order swap.
- `position_bias_trials.csv`: position-swap trial definitions.

### `20_scoring/` — per-head importance and perturbation (1024 rows for Llama-8B = 1024 heads, etc.)

- **`rsa_per_head.csv`** — backs Appendix A §A.6 / B §B.3 regression.
  Columns: `layer, head, rsa_crel, rsa_attr, rsa_max, rsa_max_before_clamp, rsa_crel_pval, rsa_attr_pval`.
  `rsa_max = max(rsa_crel, rsa_attr, 0)` (non-negative by construction).
  Permutation p-values: `N_PERMS = 10 000`, `PERM_SEED = 42`.

- **`perturbation_per_head.csv`** — backs Appendix A §A.7 / B §B.2 within-stratum.
  Columns: `layer, head, perturbation_L2`.
  Per-head mean of `‖a_clean − a_source‖₂` over 120 same-pair-source tuples (6 crel pairs × 10 pair_id × 2 attr_dim).

- **`cell_classification.csv`** — backs Appendix A §A.9 cell definitions.
  Columns: `layer, head, cell, rsa_max, perturbation_L2, importance_quartile`.
  `cell ∈ {hihp, hilp, C, D}` (low-importance pair labelled `C` and `D`).
  `importance_quartile ∈ {Q1, Q2, Q3, Q4}` is used by the within-stratum analysis (Appendix B §B.2).

- **`median_thresholds.json`** — model-specific `(rsa_median, pert_median)` pair used for the 2×2 split.

- **`perhead_regression.csv`** — per-model joint OLS regression (z-scored): `mean_abs_delta ~ rsa_max + perturbation_L2 + layer`. Columns: `model, n_heads, beta_imp, beta_pert, beta_layer, p_imp, p_pert, p_layer, R2_full, R2_imp_only, R2_pert_only, R2_layer_only, partial_R2_pert`. Produced by `code/shared/compute_perhead_regression.py`.

  **Mapping to Appendix B Table B4:** the paper Table B4 column labeled `partial_R²_pert` displays the per-model **`R2_pert_only`** values (single-predictor R² of perturbation, range 0.072–0.586 across the 8 models, matching the appendix prose "pert-only ranging 0.072–0.586"). The CSV's `partial_R2_pert` column gives the standard *partial-R² of perturbation* (coefficient of partial determination, `(SSR_reduced − SSR_full) / SSR_reduced`, range 0.07–0.60). Both are shipped so reviewers can verify either definition; for the printed Table B4 values, read `R2_pert_only`.

- **`behavioral_crel_breakdown.csv`** — per-model 6-row CSV (one row per crel_pair: `SAME-OPP, SAME-MORE, SAME-LESS, OPP-MORE, OPP-LESS, MORE-LESS`) with columns `crel_pair, n_C_trials, n_D_trials, c_mean_abs_delta, d_mean_abs_delta, c_minus_d`. Backs Appendix B §B.5 ("Cell C per-crel breakdown") and Table B10 (per-crel-pair C-D gaps). The MORE-LESS row carries the largest C−D gap in 6 of 8 models; on Gemma-2-9B and GPT-J the OPP-MORE row is largest instead, consistent with the bolding in Table B10. Produced by `code/shared/compute_behavioral_crel_breakdown.py`.

### `30_patching/` — group patching outcomes

- **`individual_head_effects.csv`** — per-head per-trial single-head patch effect.
  Columns: `layer, head, trial_id, crel_pair_id, pair_id, attr_dim, target_logit_clean, target_logit_patched, delta_target_logit_signed`.
  Largest file (~3–22 MB per model). Used by causal-importance reanalysis (Appendix B §B.10) to compute `causal_imp`.

- **`dose_response_v2.csv`** — group patching mean `|Δ target logit|` per (cell, ordering, ratio).
  Columns: `cell, ordering, ratio, k, mean_abs_delta_target_logit, mean_target_rank, n_trials, ...`.
  Backs **body Table 2** (in-sample C/D) and **body Figure 3** (dose-response curves).

### `50_prospect/` — held-out prospective evaluation, 10 splits (seeds 43–52)

- **`prospect_dose_response_v2.csv`** — backs **body Table 3** (prospective C−D, A−B with sign-consistency).
  Columns: `seed, cell, ordering, ratio, k, mean_abs_delta_target_logit, ...`.

- **`per_seed_cell_classification_prospect.csv`** — per-seed cell membership re-derivation.
- **`trials_all_prospect.csv`** — Prospect 80-pair held-out trial set.
- **`word_pairs_prospect.csv`** — Prospect 80 word pairs (`WORD_SEED = 2027`).

### `causal_importance_reanalysis_v2_session_consistent/` — Appendix B §B.10

- **`per_head_causal_imp.csv`** — single-head causal patch effect per head.
  Columns: `layer, head, causal_imp`.
  `causal_imp = mean |Δ target logit| over 120 tuples` (single-head patching).

- **`cell_classification_causal.csv`** — re-classification of heads under (`causal_imp`, `perturbation_L2`).
  Columns: `layer, head, causal_imp, perturbation_L2, hi_imp, hi_pert, cell_new_abs` (cell_new_abs ∈ {`hihp_p`, `hilp_p`, `C_p`, `D_p`}). The column name `cell_new_abs` flags this as the absolute-importance reclassification of §B.10.

- **`dose_response_v2_causal.csv`** — group patching with `C_p_imp_asc` / `D_p_imp_asc` orderings.
  Backs **body Table 6** (C′ vs D′ at r=0.10) and Appendix B Tables B11–B12.

## 2. Factual recall — `data/factual_recall/`

### `00_data/` (shared across all 8 models)

- **`trial_definitions.csv`** — 68 trials × cross-tokenizer intersection filter.
  Columns: `trial_id, relation_id, target, competitor, prompt`.
  Backs Appendix D §D.2.

- **`competitor_map.csv`** — per-target competitor selection.
- **`relation_manifest.csv`** — 34 Wikidata relations × 2 trials each.

### Per-model `{m}/` (8 models)

- **`10_collection/clean_logits.csv`** — clean-prompt target/competitor logits per trial.

- **`20_scoring/importance_per_head.csv`** — per-head FR importance = single-head patch effect on target logit, averaged over 68 trials.
- **`20_scoring/perturbation_per_head.csv`** — per-head FR perturbation_L2 (same convention as novel-word).
- **`20_scoring/cell_classification.csv`** — FR-specific 2×2 cell labels.
- **`20_scoring/importance_per_trial_head.csv`** — per-(head, trial) raw patch effects (largest FR file, used for selector robustness).
- **`20_scoring/perturbation_per_trial_head.csv`** — per-(head, trial) perturbation_L2.

- **`30_patching/behavioral_gamma.csv`** — per-trial FR group-patching outcomes.
- **`30_patching/behavioral_gamma_summary.csv`** — model-level summary.
  Columns: `cell, ordering, ratio, argmax_stability, target_top1_stability, mean_target_rank, mean_margin, mean_abs_delta, n_trials`.
- **`cross_model/fr_dose_response_per_model.csv`** (under `data/factual_recall/cross_model/`) — concatenation of per-model `behavioral_gamma_summary.csv` with a `model` column prepended (8 models × 5 ratios × 7 orderings = 280 rows). Cited in Appendix C §C.3 as the FR dose-response source. Produced by `code/shared/compute_fr_dose_response_per_model.py`.
  Backs **body Table 7** (FR C/D + Gini).

## 3. IOI — `data/ioi/` (Appendix E)

- **`10_collection/ioi_prompts.csv`** — IOI test prompts.
- **`10_collection/ioi_circuit_gt.csv`** — Wang et al. (2022) ground-truth circuit (26 heads).
- **`10_collection/abc_prompts.csv`** — ABC distractor prompts (counterfactual control).
- **`20_scoring/cell_classification.csv`** — signed `ΔLD` 2×2 classification.
- **`20_scoring/cell_classification_abs.csv`** — absolute `|ΔLD|` 2×2 classification.
- **`20_scoring/cell_classification_signed_ld.csv`**, **`cell_classification_abs_ld.csv`** — metric-robustness variants.
- **`20_scoring/importance_per_head.csv`** — IOI importance = `|logit(IO) − logit(S)|`.
- **`20_scoring/perturbation_per_head.csv`** — IOI perturbation_L2.
- **`30_patching/cd_ratio_summary.csv`** — IOI C/D ratio per ordering / ratio.
- **`30_patching/dose_response_v2.csv`** — IOI dose-response.
  Backs Appendix E Tables E1–E5 (NNM exclusion, Backup NM exclusion).
- **`30_patching/e4_bootstrap_ci.csv`** — Appendix §E.4 effect-size summary for the C-ex-NNM vs D head comparison (n_C=26, n_D=44). Columns: `mean_diff` = +0.0589 logit with 95% bootstrap CI [-0.0000, 0.1425]; `cohens_d` (pooled-SD, ddof=1) = 0.4822 with 95% CI [-0.0001, 0.8300] — matches body §4.2 d≈0.48; `cliffs_delta` = +0.2570 with 95% CI [-0.0245, 0.5192]. 10 000 two-sample resamples, `seed=42`. Produced by `code/shared/bootstrap_e4_ci.py`.

## 4. ACDC — `data/acdc_ioi_cross_model/` (Appendix F.5)

Edge-level analysis on 4 model variants: `gpt-j-6b, gemma-2-9b-it, qwen-2.5-7b-instruct, qwen-2.5-14b-instruct`.

- **`10_collection/prompts_master.csv`** — IOI prompt corpus shared across all 4 models.
- Per-model **`30_patching/{m}/cd_ratio_summary.csv`** — edge-count per ordering.
- Per-model **`30_patching/{m}/dose_response_v2.csv`** — edge-level dose-response.
- **`50_aggregate/appendix_ready_tables.csv`** — directly backs **Appendix F Table F4**: per-(model, condition) `selected_edges`, `faithfulness`, `sparsity`, `threshold`. The edge-count inflation factor 4.4×–5.8× cited in F.5 = `selected_edges(condition='celld') / selected_edges(condition='standard')` per model.
- **`50_aggregate/aggregate_policy_summary.csv`** — supporting policy-level summary.

## 5. Boundary tasks — `data/{induction,greater_than,kn_fact_edit_robustness}/`

### Induction (GPT-2 Small, Olsson et al. 2022) — Appendix F.2

- `15_detectors/detector_scores.csv` — per-head induction score.
- `20_scoring/{cell_classification,cell_classification_abs,importance_per_head,perturbation_per_head}.csv`
- `30_patching/{cd_ratio_summary,dose_response_v2,grouped_patching_summary}.csv`

  **Backing aggregation for Table F1.** The paper Table F1 `|Δloss|` values at r = 0.10 (C = 1.0210, D = 1.1266) are the **prompt-mean of `|delta_loss_signed|` from `dose_response_v2.csv`** at `ratio = 0.10`, NOT the per-head pre-aggregated cell means in `cd_ratio_summary.csv` (which is on a different scale: per-head means averaged across heads first, then across prompts). Reviewers reproducing Table F1 should query `dose_response_v2.csv` rows where `ratio == 0.10` and compute `mean(abs(delta_loss_signed))` per `group ∈ {C_imp_asc, D_imp_asc}`.

### Greater-Than (GPT-2 Small, Hanna et al. 2023) — Appendix F.3

- `15_detectors/`, `20_scoring/`, `30_patching/{cd_ratio_summary,dose_response_v2}.csv`

  **Backing aggregation for Table F2.** Same rule as Induction Table F1: paper values C = 0.00671, D = 0.00342 at r = 0.10 come from `dose_response_v2.csv` (prompt-mean of `|delta_score_signed|`), not from `cd_ratio_summary.csv` (per-head pre-aggregated). Note: in the paper text, the metric is named `|Δ score|` — the Greater-Than score of Hanna et al. — to keep the GT abbreviation reserved for "ground truth".

### Knowledge Neurons (BERT-base, Dai et al. 2022) — Appendix F.4

- `10_collection/facts_master.csv` — 24 fact configurations.
- `80_aggregate/aggregate_full.csv` — pooled C/D ratio at 1% suppression and downstream erasure outcome.
  Backs Appendix F Table F3.

## 6. Cross-model aggregations — `data/cross_model/` and `data/cross_model_per_model/`

### `cross_model/` (project-root level)

- **`causal_importance_phase2_dose_response_long.csv`** — Phase 2 v2 reanalysis long-format (24 rows × 7 cols).
  Backs Appendix B Table B11.

- **`causal_importance_phase2_summary.csv`** / **`mwu.csv`** / **`paired_tests.csv`** —
  significance tests for B.10 (MWU, paired Wilcoxon, sign test).
  Backs Appendix B Table B12.

- **`diagonal_dose_response_novel_word.csv`** / **`_summary.csv`** — diagonal-rank ordering sensitivity (Appendix C §C.4 alternative selectors).

- **`dose_response_v2_summary.csv`** — cross-model dose-response summary at the diagnostic ratio.

- **`matched_head_def_a_rsa_quartile_hipert_vs_lopert.csv`** / **`matched_head_def_b_pert_quartile_hirsa_vs_lorsa.csv`** —
  within-stratum perturbation comparison (Appendix B §B.2 Definition A and Definition B; the within-stratum 32-row Table B2 is rebuilt from these). Produced by `code/shared/run_matched_head_both_definitions.py`.

- **`layer_bias_analysis.csv`** — per-model Spearman ρ(layer, perturbation_L2) and within-model layer-quartile statistics. Backs Appendix B §B.2.1 (Qwen layer-perturbation entanglement). Produced by `code/shared/compute_layer_perturbation_correlation.py`.

### `cross_model_per_model/` (mirrors `models/cross_model/`)

**Paper-cited aggregates (4 files; regenerated from current per-model sources 2026-05-01)**:

- **`signed_delta_per_cell.csv`** — backs **Appendix B Table B1**. 8 models × 4 cells = 32 rows. Columns: `model, cell ∈ {A,B,C,D}, n_heads, mean_signed, mean_abs, pct_down_pt1, pct_up_pt1, pct_near0_pt1, tilt_down_minus_up`. `mean_signed` = mean(`delta_target_logit_signed`) over (head, tuple) pairs in cell, where `delta_target_logit_signed = patched − clean` (Appendix B §B.1). `tilt_down_minus_up` = (% with Δ < −0.1) − (% with Δ > +0.1), in pp.
- **`strict_argmax_flip_per_model.csv`** — strict (token-id-level, no normalization) argmax-flip rate per (model, group, ratio). 8 models × 5 groups × ~5 ratios = 156 rows. Columns: `model, group, ratio, strict_flip_rate, n`. Companion to body Table 4 and Appendix B Table B6 (which use capitalization-normalized flips).
- **`normalized_argmax_per_model.csv`** — per-model task-validation accuracy (clean prompts, no patching). Columns: `model, n=80, strict_argmax_pct, normalized_first_token_argmax_pct, gain_pp, d_plus_pct`. Backs Appendix A §A.5 (Table A3) D+ validity and the body §2.1 "100% capitalization-normalized first-token argmax" claim for the canonical-instruct core.
- **`causal_imp_crossfit.csv`** — Phase-1-vs-Phase-2 cross-fit per-head sensitivity check on the causal-importance reanalysis (8 models × 5 cycles). Columns include `model, source ∈ {v2_session_consistent, v3_post_v7_5_0_rerun, v4_eligible_denominator}, n_C_p, n_D_p, k, mean_abs_delta_C_p_full, mean_abs_delta_D_p_full, mean_abs_delta_C_p_xfit, mean_abs_delta_D_p_xfit, gap_full, gap_xfit`. Per-model `source` reflects the latest reanalysis variant (per Appendix G §G.1 / `metadata/run_summary.txt`). Backs Appendix B Table B15.

**Auxiliary cross-model summaries** (retained for consistency checks; canonical table values are documented in the corresponding per-model CSVs and Appendix tables above):

- **`8model_comparison.csv`** — auxiliary cross-model summary; canonical values come from per-model `30_patching/dose_response_v2.csv` (Appendix G §G.4 / Table G1).
- **`behavioral_flip_rates_per_model.csv`** — auxiliary per-model top-1 flip aggregate (7-model only; 70B not included).
- **`cell_counts_per_model.csv`** — auxiliary per-model |A|/|B|/|C|/|D| (7-model only); canonical counts are in Table G2.
- **`main_dose_response_gap_per_model.csv`** — auxiliary C−D summary (7-model only).
- **`prospect_gap_per_model.csv`** — auxiliary prospect gap summary (Prospect not run on 70B by design).
- **`body_table7_gini_fr_vs_nw.csv`** — auxiliary Table 7 Gini comparison; canonical values come from a per-head Gini recompute on the eligibility-filtered head set (Gemma-3 = 1885, Qwen-7B = 783, others architectural).
- **`saturation_onsets.csv`** — per-model `r_sat = min(|C|, |D|) / eligible_heads`. Matches Appendix G §G.3 Table G2.
- **`within_stratum_28rows.csv`** — flattened table source for Appendix B Table B2 (the `28rows` filename refers to the native-precision seven-model × 4-quartile = 28-comparison family α / 28 = 0.001786; the 70B stress-arm rows are appended in the Appendix table for the full 8-model × 4-quartile = 32-row presentation).
- **`causal_importance_reanalysis_v2_summary.csv`** — auxiliary Phase-2 summary; canonical headline values are in Table B11/B12, with the cross-fit values in `causal_imp_crossfit.csv`.

## Naming conventions and units

- All `mean_abs_delta_target_logit`, `delta_target_logit_signed`, `target_logit`, `incorrect_logit`, `competitor_logit`, `gap`, `causal_imp` values are in **logit units** (natural log of pre-softmax score).
- `perturbation_L2`: L2 norm in head-activation space (`o_proj.input` slice for the target head, see Appendix A §A.7); units are model- and layer-dependent.
- `ratio` columns are **patch budgets** as fractions of total heads (`r = k / N`).
- `seed` columns identify the calibration/evaluation random split (Prospect uses seeds 43–52 for the 10-split protocol).

## Reproduction provenance

All values can be regenerated from the model checkpoints listed in `metadata/checkpoint_revisions.txt` via the analysis scripts in `code/` and the Colab notebooks in `notebooks/`. The production runs spanning 2026-04-13 to 2026-04-30 are documented in `metadata/run_summary.txt` (8 originals plus the 2026-04-29/30 paper-Main re-run on `transformers 5.0.0` for four models and the 2026-04-30 Phase 2 v3 reanalysis); environment versions are pinned in `metadata/environment.yml` (three blocks: `wcc-paper-main-7std`, `wcc-paper-main-70b`, `wcc-phase2-v2-reanalysis`). The Phase 2 v2 / v3 reanalysis is session-consistent — see Appendix G §G.5 for the OLMo-13B 6.0-logit cross-session divergence rationale.

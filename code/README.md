# Code — analysis scripts and pipelines

This directory holds the analysis scripts that produced the per-model and
cross-model CSVs under `data/`. The four production pipelines (one per task)
are the canonical entry points:

| Pipeline | Where | Purpose |
|---|---|---|
| Novel-word (8 models) | `notebooks/unified_pipeline_*.ipynb` (Colab) | RSA + perturbation + cell classification + group patching + Prospect 10-seed |
| Causal-importance reanalysis (8 models) | `notebooks/causal_importance_reanalysis_*.ipynb` | Body §3.6 / Appendix B.10 |
| Factual recall (8 models) | `notebooks/factual_recall_*.ipynb` | Body Table 7 / Appendix D |
| ACDC cross-model (4 model variants) | `notebooks/acdc_ioi_cross_model.ipynb` | Appendix F.5 / Table F4 |
| IOI / Induction / Greater-Than | `code/tasks/{ioi,induction,greater_than}/run_full_pipeline.py` | Boundary benchmarks (Appendix E, F.1–F.3) |
| KN (knowledge neurons) | `code/tasks/kn_fact_edit/run_robustness.py` | 24-config robustness sweep (Appendix F.4) |

The cross-model post-processing scripts under `code/shared/` reproduce the
cross-model summary CSVs cited in the appendices:

| Script | Produces | Backs |
|---|---|---|
| `shared/compute_layer_perturbation_correlation.py` | `cross_model/layer_bias_analysis.csv` | Appendix B §B.2.1 (Qwen layer–perturbation entanglement) |
| `shared/compute_perhead_regression.py` | `data/novel_word/{m}/20_scoring/perhead_regression.csv` (×8 models) | Appendix B §B.3 / Table B4 (joint OLS regression) |
| `shared/bootstrap_e4_ci.py` | `data/ioi/30_patching/e4_bootstrap_ci.csv` | Appendix E §E.4 (mean-diff and Cliff's-delta bootstrap CI) |
| `shared/compute_saturation_onsets.py` | `cross_model_per_model/saturation_onsets.csv` | Appendix G §G.3 / Table G2 (per-model `r_sat`) |
| `shared/compute_fr_dose_response_per_model.py` | `factual_recall/cross_model/fr_dose_response_per_model.csv` | Appendix C §C.3 (FR dose-response across selectors) |
| `shared/compute_behavioral_crel_breakdown.py` | `data/novel_word/{m}/20_scoring/behavioral_crel_breakdown.csv` | Appendix B §B.5 / Table B10 (per-crel-pair C−D gap) |

`code/shared/` also contains three legacy helpers that read from the pre-archive `models/{m}/...` layout (used by the original Colab pipeline), not from this archive's `data/...` layout: `run_matched_head_both_definitions.py`, `run_cell_cd_test_all_models.py`, `rebuild_cross_model_dose_summaries.py`. Reviewers should not invoke these — the CSVs they would emit (`cross_model/matched_head_def_{a,b}_*.csv`, `cross_model/dose_response_v2_summary.csv`) are already shipped under `data/cross_model/` and can be read directly to verify Appendix B §B.2 (within-stratum, both definitions) and the cross-model dose-response summary referenced in body §3.5.

The IOI, Induction, and Greater-Than pipelines under `code/tasks/{ioi, induction, greater_than}/` use `task_config.py` + `run_full_pipeline.py` and share helpers under `code/tasks/_shared/` (collection, scoring, patching, post_hoc, io_utils, config_writer). The KN (knowledge neurons) pipeline under `code/tasks/kn_fact_edit/` uses a single `run_robustness.py` that sweeps 24 fact-edit configurations and writes `data/kn_fact_edit_robustness/80_aggregate/aggregate_full.csv` directly (no separate `task_config.py`).

## Reproduction map (Appendix table → command)

```bash
# Set the project root (the unzipped supplementary directory)
export WCC_ROOT=$(pwd)
export HF_TOKEN=<YOUR_HF_TOKEN>   # supply your own HuggingFace token

# Body Table 2 / Figure 3 — Novel-word in-sample C vs D, dose-response
#   Source CSV (8 models): data/novel_word/{m}/30_patching/dose_response_v2.csv
#   Producer: notebooks/unified_pipeline_*.ipynb (run on Colab; 4 model-specific
#   notebooks plus model-loop variants for the remaining models)

# Body Table 3 — Prospective 10-seed
#   Source CSV: data/novel_word/{m}/50_prospect/prospect_dose_response_v2.csv
#   Producer: notebooks/prospect_*.ipynb

# Body Table 4 — Greedy top-1 flip rate (capitalization-normalized)
#   Source CSV: data/final_tables/body_table4_behavioral_flip.csv
#               + per-model data/novel_word/{m}/38_behavioral/flip_rates_normalized.csv
#               (cross-model summary: data/cross_model_per_model/normalized_flip_per_model.csv)
#   Producer: notebooks/unified_pipeline_*.ipynb Phase 38 (behavioral derivation),
#             plus paper/_ai_workspace/compute_normalized_flip_rates.py for cap-normalization

# Body Table 5 — std vs D
#   Source CSV: data/novel_word/{m}/30_patching/dose_response_v2.csv (std_imp_asc rows)

# Body Table 6 / Appendix B Tables B11–B15 — Causal-importance reanalysis
#   Source CSV: data/novel_word/{m}/causal_importance_reanalysis/
#                 dose_response_v2_causal.csv
#               + data/cross_model/causal_importance_phase2_*.csv
#   Producer: notebooks/causal_importance_reanalysis_*.ipynb

# Body Table 7 / Appendix D — Factual recall
#   Source CSV: data/factual_recall/{m}/30_patching/behavioral_gamma_summary.csv
#               + data/cross_model_per_model/body_table7_gini_fr_vs_nw.csv
#   Producer: notebooks/factual_recall_*.ipynb
#   Aggregator: python code/shared/compute_fr_dose_response_per_model.py

# Body Table 8 / Appendix F — Boundary benchmarks
python code/tasks/induction/run_full_pipeline.py
python code/tasks/greater_than/run_full_pipeline.py
python code/tasks/ioi/run_full_pipeline.py
python code/tasks/kn_fact_edit/run_robustness.py
# ACDC: notebooks/acdc_ioi_cross_model.ipynb produces data/acdc_ioi_cross_model/50_aggregate/
#       (appendix_ready_tables.csv backs Table F4 directly)

# Appendix B.2.1 (Qwen layer–perturbation correlation, Table B3)
python code/shared/compute_layer_perturbation_correlation.py

# Appendix B.3 (Table B4 joint regression)
python code/shared/compute_perhead_regression.py
# (per-model OLS, z-scored, partial R^2 of perturbation after controlling for
#  importance and layer; output: 8 per-model perhead_regression.csv files)

# Appendix B.5 (Table B10 per-crel-pair C-D gap)
python code/shared/compute_behavioral_crel_breakdown.py

# Appendix E.4 (bootstrap CI for mean diff and Cliff's delta)
python code/shared/bootstrap_e4_ci.py

# Appendix G.3 / Table G2 (per-model r_sat)
python code/shared/compute_saturation_onsets.py
```

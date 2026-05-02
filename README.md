# When Controls Collapse — Supplementary Material (Appendix G.6)

The shipped bundle contains the canonical review-time artifacts used to verify the
submitted paper tables and figures. Internal exploratory scripts and intermediate
development artifacts are not included. The canonical reproduction path uses the
notebooks under `notebooks/`, the scripts under `code/`, and the CSV artifacts under
`data/`. The four metadata files below pin the byte-level reproduction state referenced
in Appendix G.6.

## Files

### `checkpoint_revisions.txt`
Human-readable table: 8 models × {HF repo_id, commit SHA, total bytes, file count, dtype, run-date trace}.
The commit SHA is the head of `main` at the **earliest** production-run date that loaded the model.
At the final paper-preparation snapshot recorded in this archive, all 8 models' `main` heads
remained unchanged from these SHAs, so reviewers reproducing at the snapshot's time obtain
identical bytes by either (a) `revision=<sha>` or (b) plain `revision="main"`.

### `checkpoint_manifest.json`
Programmatic per-file manifest: for every file in every model snapshot, gives `path`, `size`,
`lfs_sha256` (for LFS-backed weights/tokenizers), and `blob_id` (git OID for non-LFS small
files like `config.json`). 128 files across 8 models, 289 GB total. Reviewers can verify
byte-level integrity of any downloaded snapshot against this manifest.

All 8 models' Hugging Face `main` heads at the listed SHAs remain unchanged across
the production-run window, so the manifest is identical to a re-fetch at any of the
10 production-run dates.

### `run_summary.txt`
Maps each production run to its date, hardware, library environment, and data
location. Each row pins down which `environment.yml` block applies and where the
deliverable CSVs are stored relative to the supplementary archive.

### `environment.yml`
Three distinct conda-style environment blocks (`wcc-paper-main-7std`, `wcc-paper-main-70b`,
`wcc-phase2-v2-reanalysis`) plus a local-auxiliary note for the GPT-2-small / BERT-base experiments. Versions are captured directly
from production-notebook stdout logs preserved in the notebook output cells:
- 7-standard paper-Main: `Transformers=5.0.0` (Cell 3 log line)
- 70B paper-Main: `Successfully installed bitsandbytes-0.49.2 transformers-5.6.0` (Cell 1 install log)
- Prospect 70B: explicit `Uninstalling transformers-5.0.0 → installed transformers-5.6.0` (Cell 1)
- Phase 2 v2: `Torch: 2.10.0+cu128, nnsight 0.6.3` + transformers resolved via `pip install -U` to 5.6.2 at 04-26~27 (PyPI release table cross-validated)
- The four-model novel-word re-run (Llama-8B, Gemma-2-9B, Gemma-3-27B, 70B) reuses `wcc-paper-main-7std` / `-70b` blocks with Cell 1 explicitly pinned to `transformers==5.0.0` (3 native-precision models) or `==5.6.0` (70B) to neutralize Colab default drift.
- The Phase 2 v3 reanalysis reuses `wcc-phase2-v2-reanalysis` byte-for-byte.

## Reproducing a single model run

```python
from huggingface_hub import snapshot_download
import torch
from nnsight import LanguageModel

# 1. Pin the exact commit used in the paper (from checkpoint_revisions.txt):
local = snapshot_download(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    revision="0e9e39f249a16976918f6564b8830bc894c89659",
)

# 2. Match the transformers version to the run environment.
#    Look up `run_summary.txt` for the run you want to reproduce; cross-reference
#    `environment.yml` for the exact pip pin. E.g.:
#    - paper-Main 7-standard:    `pip install transformers==5.0.0`
#    - paper-Main 70B-4bit:      `pip install transformers==5.6.0 bitsandbytes==0.49.2`
#    - Phase 2 v2 reanalysis:    `pip install transformers==5.6.2`

# 3. Load with nnsight 0.6.3:
model = LanguageModel(local, dtype=torch.bfloat16, device_map="auto")
```

## Code-bundle scope (`supplementary/code/`)

The shipped `code/` tree contains the production scripts that produced every CSV
in `supplementary/data/` and every table / figure in the paper. Exploratory and
one-shot helper scripts are not shipped, to keep the bundle minimal.

### Production pipelines

| Pipeline | Script | Used by |
|---|---|---|
| Novel-Word Phase 10–40 (7 standard models) | `notebooks/unified_pipeline_*.ipynb` | All 7 standard models |
| Novel-Word Phase 10–40 (Llama-70B-4bit) | `notebooks/unified_pipeline_llama-3.1-70b-instruct-4bit.ipynb` | 70B paper-Main |
| Novel-Word Prospect (Phase 50) | `notebooks/prospect_*.ipynb` | All 8 models |
| Phase 2 v2 reanalysis (group-patch causal-imp) | `notebooks/causal_importance_reanalysis_*.ipynb` | All 8 models |
| Behavioral aggregates | `notebooks/behavioral_*.ipynb` | All 8 models |
| ACDC cross-model | `notebooks/acdc_ioi_cross_model.ipynb` | 4 TransformerLens-loadable models (gpt-j-6b, gemma-2-9b-it, qwen-2.5-7b-instruct, qwen-2.5-14b-instruct) |
| Factual-recall (Appendix D) | `notebooks/factual_recall_*.ipynb` | All 8 models |
| Cross-model aggregation | `code/cross_model/aggregate_v7_5.py` | Selected cross-model summary CSVs (cell counts, dose-response gap, prospect gap, behavioral); paper-table values can also be verified directly from the per-model CSVs in `data/` |
| Boundary tasks (Appendix F) | `code/tasks/{ioi,greater_than,induction,kn_fact_edit}/run_*.py` | Appendix E/F tables |

### What is NOT shipped (and why)

- **Alternative GPT-J execution paths**: the canonical GPT-J production data under `data/novel_word/gpt-j-6b-fp32/` was produced by `notebooks/unified_pipeline_gpt-j-6b-fp32.ipynb` on Colab A100-SXM4-80GB. Alternative loading paths are not part of the canonical reproduction path.
- **NDIF remote-inference scripts**: the canonical 70B reproduction is `notebooks/unified_pipeline_llama-3.1-70b-instruct-4bit.ipynb`.
- **Auxiliary helper scripts** that are not part of the canonical reproduction path; the production pipeline notebooks under `notebooks/` plus `code/cross_model/aggregate_v7_5.py` are the canonical entry points.
- **IOI alternative variants** (`tasks/ioi/run_v14_*.py`, `run_pythia_v13*.py`, `run_sae_*.py`): exploratory variants not used in the submitted Appendix E. The canonical IOI script is `tasks/ioi/run_full_pipeline.py`.
- **KN pilot scripts** (`tasks/kn_fact_edit/run_pilot_phase{10_20_30,35_36}.py`, `run_sparse_regime.py`): not part of the canonical reproduction path. The canonical KN script (Appendix F.4 24-config robustness sweep) is `tasks/kn_fact_edit/run_robustness.py`.

### Raw activation tensors (not bundled)

Raw activation `.npz` artifacts (per-model `activation_vectors_*.npz` and
`clean_logits_*.npz` at ~0.3–1 GB per model per phase, totalling ~4.9 GB
across 8 models × 2 phases) are **not included in this bundle**. The production
Colab notebooks under `notebooks/` regenerate them deterministically from the
pinned model checkpoints (see `metadata/checkpoint_revisions.txt`); the per-model
CSVs that paper tables actually cite (`cell_classification.csv`,
`individual_head_effects.csv`, `dose_response_v2.csv`, etc.) are shipped here
in full and are the canonical artifacts for verifying paper claims.

### Verifying paper-cited table values (no script execution needed)

The `data/` directory contains the **canonical per-model CSVs** that back every
paper table; per-CSV schema, units, and the Appendix table each CSV backs are
catalogued in `DATA_DICTIONARY.md`. Aggregation logic (within-stratum 28 rows,
joint OLS regression, percentile sweeps, FR vs NW Gini, signed-vs-absolute Δ
target logit, etc.) is described in prose in the cited Appendix sections.

The audit / table-generation scripts that produced the Appendix table cells
from the per-model CSVs are not shipped (see "What is NOT shipped" above);
the per-model CSVs themselves are the canonical artifacts. Reviewers verifying
paper claims can therefore proceed in either of two ways:

1. **Direct comparison (recommended for spot-checks)**: locate the relevant
   `data/...` CSV via `DATA_DICTIONARY.md` and compare the row(s) used in the
   Appendix table directly. No code execution needed.
2. **Full re-derivation (recommended for end-to-end reproduction)**: run the
   production pipeline (Colab notebooks under `notebooks/` plus the cross-model
   aggregator `code/cross_model/aggregate_v7_5.py`) starting from the pinned
   model checkpoints (`metadata/checkpoint_revisions.txt`).

### Cross-model aggregator usage

```bash
export WCC_ROOT=/path/to/unzipped/supplementary
python supplementary/code/cross_model/aggregate_v7_5.py
```

Reads from `$WCC_ROOT/data/novel_word/{model}/{20_scoring,30_patching,38_behavioral,50_prospect}/...`
and writes 4 cross-model tables under `$WCC_ROOT/data/cross_model_per_model/`.

## Critical caveat: OLMo-2-13B Phase 2 group patching

For the Phase 2 group-patching reanalysis of OLMo-2-13B (Appendix G §G.5), the `transformers`
version of activation extraction MUST match the version of group-patching forward passes.
Cross-session mixing of `transformers 5.0.0` (paper-Main extraction) with `transformers 5.6.x`
(Phase 2 reanalysis) produces a **6.0-logit** clean-logit divergence on OLMo (other 7 models
unaffected). The `wcc-phase2-v2-reanalysis` environment is **session-consistent** by design:
both extraction and patching happen in a single Colab kernel using the same `transformers 5.6.2`,
which restores byte-level reproducibility for OLMo. This is the v2 notebook redesign documented
in Appendix G §G.5 and summarized in `metadata/run_summary.txt`.

## Caveat for size-limited supplementary submission

Total checkpoint size is 289 GB (Llama-70B alone is 39.5 GB, GPT-J-6B fp32 is 73 GB), which
exceeds typical NeurIPS supplementary archive size caps. We therefore do NOT bundle the
model weights here. Instead, the (model_id, revision_sha) pair in `checkpoint_revisions.txt`
together with the per-file LFS sha256 in `checkpoint_manifest.json` gives a stable byte-level
identifier from which any reviewer can reconstruct the exact bytes via Hugging Face Hub's
content-addressed download. Hugging Face retains commit history indefinitely, so this
identifier remains valid post-publication.

## License

The analysis code under `code/` and the Colab notebooks under `notebooks/` are released
under the **MIT License** (see `LICENSE`). Pre-trained model weights are NOT bundled here; each
model retains its upstream license as published on the Hugging Face Hub (see
`metadata/checkpoint_revisions.txt` for the model_id → repo mapping).

The copyright holder line in `LICENSE` is currently anonymized (`[Anonymized for double-blind
review]`) and will be replaced with author / affiliation information at camera-ready time.
The MIT terms themselves are unchanged.

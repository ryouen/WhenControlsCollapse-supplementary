"""
Shared utilities for WCC auxiliary task pipelines (IOI, ICL, GT, Induction).

All 4 tasks run on GPT-2 Small (12L × 12H × 64d = 144 heads) and share:
- Perturbation metric computation (4 metrics)
- Median-split cell classification (A/B/C/D)
- Individual + group dose-response patching
- Config and I/O helpers

See spec/WCC_naming_conventions.md §6-7 for canonical column names.
See spec/WCC_analysis_pipeline.md §1.1 for the Phase number registry.
"""

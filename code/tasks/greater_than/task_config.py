"""
Greater-Than task configuration.

Source: Hanna et al. 2023 — 5-head ground-truth circuit on GPT-2 Small.
Paper role: Null-difference control — confound ABSENT in localized circuit.

Key numbers to reproduce:
    rho(importance, perturbation) = +0.1747 (weak positive)
    C/D @ 10% (raw_l2, resample) = 3.02×
    AUROC vs GT = 0.4295 (BELOW chance — anti-correlated)
    All 5 GT heads land in Cell A
    GT is "null-difference control", NOT "circuit recovered"

CRITICAL narrative: AUROC < 0.5 means per-head resample patching is
anti-correlated with circuit membership. The 5 GT heads have individual
patching effects smaller than many non-circuit heads because year comparison
is nonlinear and coordinated.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GTConfig:
    task_id: str = "greater_than"
    model_id: str = "gpt2"
    model_short: str = "gpt2-small"

    n_prompts: int = 100  # P2/O1 canonical (NOT A3's 200)
    n_layers: int = 12
    n_heads: int = 12
    d_head: int = 64
    total_heads: int = 144

    primary_importance: str = "imp_gt_score_drop"
    importance_metrics: list = field(default_factory=lambda: ["imp_gt_score_drop"])
    primary_perturbation: str = "pert_raw_l2"
    ablation_methods: list = field(default_factory=lambda: ["resample"])

    # Template: P2/O1 single template (NOT A3 5-template)
    template: str = "The war lasted from the year {year1} to the year {year2}"
    year_range: tuple = (1700, 1999)
    corruption_method: str = "year_delta"

    # Ground truth circuit (Hanna et al. 2023) — 5 heads
    gt_circuit: list = field(default_factory=lambda: [
        (7, 10, "GT"), (7, 11, "GT"), (8, 10, "GT"),
        (8, 11, "GT"), (9, 1, "GT"),
    ])

    capture_residual: bool = False  # GT has no readout decomposition

    output_root: Path = Path("tasks/greater_than")

    @property
    def gt_heads(self) -> set:
        return {(l, h) for l, h, _ in self.gt_circuit}

    @property
    def gt_n_heads(self) -> int:
        return len(self.gt_circuit)


def gt_metric_fn(patched_logits, clean_logits, prompt_idx, correct_year_tokens):
    """
    GT metric: change in correct-year logit sum.

    For GT, the "correct" output is the set of year tokens > year1.
    """
    tokens = correct_year_tokens[prompt_idx]
    patched_sum = patched_logits[tokens].sum()
    clean_sum = clean_logits[tokens].sum()
    return float(patched_sum - clean_sum)

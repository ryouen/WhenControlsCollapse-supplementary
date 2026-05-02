"""
Induction task configuration.

No published canonical circuit — GT established operationally via detectors.
Paper role: "Signal-present confounded benchmark" (Fig 1b).

Key numbers to reproduce:
    rho(importance, perturbation) = 0.3084
    AUROC top-10 = 0.599
    C/D @ 10% = 2.08×
    Grouped patching top5 = 69× vs random (HEADLINE)
    SAE feature overlap: induction vs random = 0/10 at all 4 layers

FIXED from old project:
    - n_prompts: 100 (was silently truncated to 50)
    - corrupt_method: random_token_replace (was mixed with shuffle)
    - n_random_seeds for controls: 30 (was 1-5)
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class InductionConfig:
    task_id: str = "induction"
    model_id: str = "gpt2"
    model_short: str = "gpt2-small"

    n_prompts: int = 100  # FIXED: was silently truncated to 50 in old project
    n_layers: int = 12
    n_heads: int = 12
    d_head: int = 64
    total_heads: int = 144

    primary_importance: str = "imp_patching_loss"
    importance_metrics: list = field(default_factory=lambda: ["imp_patching_loss"])
    primary_perturbation: str = "pert_raw_l2"
    ablation_methods: list = field(default_factory=lambda: ["resample"])

    # Corrupt construction: random token replacement (NOT shuffle)
    corrupt_method: str = "random_token_replace"
    seq_half_len: int = 20   # Matches old project (SEQ_HALF_LEN=20 in all phases)
    seed: int = 42

    gt_circuit: None = None  # Operational GT defined in Phase 15

    # Detector setup for Phase 15
    detectors: list = field(default_factory=lambda: [
        "induction_score", "prev_token_score", "dup_token_score",
    ])
    detector_n_seqs: int = 50   # Matches old project (N_DET_SEQS=50)
    detector_top_k: list = field(default_factory=lambda: [5, 10, 15])
    detector_primary: str = "top10"

    # Grouped patching
    n_random_seeds: int = 30  # FIXED: was 1-5 in old project

    capture_residual: bool = False  # Induction has no readout decomposition

    output_root: Path = Path("tasks/induction")


def induction_metric_fn_loss(patched_logits, clean_logits, prompt_idx,
                              clean_tokens, seq_half_len):
    """
    Induction metric: loss increase on second-half tokens.

    delta = loss_patched(second_half) - loss_clean(second_half)
    Signed: positive means patching increased loss (disrupted induction).
    """
    import torch
    import torch.nn.functional as F

    # Second half targets (the repeated tokens)
    targets = clean_tokens[prompt_idx, seq_half_len:]  # (seq_half_len,)
    patched_loss = F.cross_entropy(patched_logits[seq_half_len - 1:-1], targets)
    clean_loss = F.cross_entropy(clean_logits[seq_half_len - 1:-1], targets)
    return float(patched_loss - clean_loss)

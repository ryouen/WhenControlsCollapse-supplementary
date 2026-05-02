"""
IOI (Indirect Object Identification) task configuration.

Source: Wang et al. 2023 — 26-head ground-truth circuit on GPT-2 Small.
Paper role: Headline metric-dependent confound + circuit recovery (Section 4).

Key numbers to reproduce:
    rho(signed_ld, pert) = 0.3125
    rho(|ld|, pert) = 0.5136
    AUROC = 0.838
    Standard threshold = 1.323; Cell-D threshold = 0.092 → 14.3× inflation
    Standard recall = 4/26; Cell-D recall = 19/26 → 15 heads rescued
"""

from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class IOIConfig:
    task_id: str = "ioi"
    model_id: str = "gpt2"
    model_short: str = "gpt2-small"

    n_prompts: int = 100
    n_layers: int = 12
    n_heads: int = 12
    d_head: int = 64
    total_heads: int = 144

    # Importance: 5 metrics, primary = signed_ld
    primary_importance: str = "imp_signed_ld"
    importance_metrics: list = field(default_factory=lambda: [
        "imp_signed_ld", "imp_abs_ld", "imp_eap", "imp_gradient", "imp_dla",
    ])

    # Perturbation: 4 metrics, primary = pert_raw_l2
    primary_perturbation: str = "pert_raw_l2"

    # Ablation: both resample and mean
    ablation_methods: list = field(default_factory=lambda: ["resample", "mean"])

    # Ground truth circuit (Wang et al. 2023) — 26 heads
    # Format: (layer, head, subclass)
    gt_circuit: list = field(default_factory=lambda: [
        # Name Movers (3)
        (9, 9, "Name_Mover"), (10, 0, "Name_Mover"), (9, 6, "Name_Mover"),
        # Negative Name Movers (2)
        (10, 7, "Negative_Name_Mover"), (11, 10, "Negative_Name_Mover"),
        # S-Inhibition (4)
        (7, 3, "S_Inhibition"), (7, 9, "S_Inhibition"),
        (8, 6, "S_Inhibition"), (8, 10, "S_Inhibition"),
        # Induction (4)
        (5, 5, "Induction"), (5, 8, "Induction"),
        (5, 9, "Induction"), (6, 9, "Induction"),
        # Duplicate Token (3)
        (0, 1, "Duplicate_Token"), (0, 10, "Duplicate_Token"),
        (3, 0, "Duplicate_Token"),
        # Previous Token (2)
        (2, 2, "Previous_Token"), (4, 11, "Previous_Token"),
        # Backup Name Movers (8)
        (9, 0, "Backup_Name_Mover"), (9, 7, "Backup_Name_Mover"),
        (10, 1, "Backup_Name_Mover"), (10, 2, "Backup_Name_Mover"),
        (10, 6, "Backup_Name_Mover"), (10, 10, "Backup_Name_Mover"),
        (11, 2, "Backup_Name_Mover"), (11, 9, "Backup_Name_Mover"),
    ])

    capture_residual: bool = True  # for Phase 40 readout (future)

    output_root: Path = Path("tasks/ioi")

    @property
    def gt_heads(self) -> set:
        """Set of (layer, head) in the ground truth circuit."""
        return {(l, h) for l, h, _ in self.gt_circuit}

    @property
    def gt_n_heads(self) -> int:
        return len(self.gt_circuit)


def generate_ioi_prompts(n_prompts: int = 100, seed: int = 42):
    """
    Generate IOI prompts using TransformerLens's IOI dataset.

    Returns: (clean_tokens, corrupt_tokens, prompt_metadata)
    where prompt_metadata is a list of dicts with io_token, s_token, etc.
    """
    from transformer_lens.utils import get_act_name
    # TransformerLens provides an IOI dataset generator
    # This is a skeleton — actual implementation uses ioi_dataset.py from TL
    raise NotImplementedError(
        "Implement using transformer_lens IOI dataset generator. "
        "See neural_exam_project/models/gpt2-small/ioi_replication/code/ "
        "for the original implementation."
    )


def ioi_metric_fn(patched_logits, clean_logits, prompt_idx, io_tokens, s_tokens):
    """
    IOI metric: signed logit difference delta.

    delta = (patched[IO] - patched[S]) - (clean[IO] - clean[S])

    This is the CHANGE in logit difference caused by patching.
    Signed: positive means patching increased IO preference.
    """
    patched_ld = patched_logits[io_tokens[prompt_idx]] - patched_logits[s_tokens[prompt_idx]]
    clean_ld = clean_logits[io_tokens[prompt_idx]] - clean_logits[s_tokens[prompt_idx]]
    return float(patched_ld - clean_ld)

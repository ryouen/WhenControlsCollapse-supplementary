"""
Shared activation extraction for auxiliary tasks (Phase 10).

Extracts per-head output vectors at the final token position for both
clean and corrupt prompts. All 4 small tasks use GPT-2 Small via
TransformerLens.

Output shape: (N, L, H, D) where
    N = n_prompts, L = n_layers, H = n_heads, D = d_head

See spec/WCC_naming_conventions.md §3.1 for file naming.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .config_writer import phase_timer
from .io_utils import save_csv, save_npz, save_json


@dataclass
class ArchitectureInfo:
    """Model architecture metadata for architecture.json."""
    model_id: str
    num_layers: int
    num_heads: int
    head_dim: int
    hidden_size: int
    total_heads: int
    vocab_size: int
    tokenizer: str

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "hidden_size": self.hidden_size,
            "total_heads": self.total_heads,
            "vocab_size": self.vocab_size,
            "tokenizer": self.tokenizer,
        }


def get_architecture(model) -> ArchitectureInfo:
    """Extract architecture info from a TransformerLens HookedTransformer."""
    cfg = model.cfg
    return ArchitectureInfo(
        model_id=cfg.model_name,
        num_layers=cfg.n_layers,
        num_heads=cfg.n_heads,
        head_dim=cfg.d_head,
        hidden_size=cfg.d_model,
        total_heads=cfg.n_layers * cfg.n_heads,
        vocab_size=cfg.d_vocab,
        tokenizer=cfg.tokenizer_name or cfg.model_name,
    )


def extract_head_outputs(
    model,
    clean_tokens: torch.Tensor,
    corrupt_tokens: torch.Tensor,
    output_dir: Path,
    capture_residual: bool = False,
    task_id: str = "",
) -> dict:
    """
    Extract per-head activation vectors for clean and corrupt inputs.

    Parameters
    ----------
    model : HookedTransformer
    clean_tokens : (N, seq_len) int tensor
    corrupt_tokens : (N, seq_len) int tensor
    output_dir : Path to 10_collection/ directory
    capture_residual : if True, also save final-layer residual stream
    task_id : for logging

    Returns
    -------
    dict with shapes, timing, and path metadata
    """
    from transformer_lens import utils as tl_utils

    arch = get_architecture(model)
    n_prompts = clean_tokens.shape[0]
    output_dir.mkdir(parents=True, exist_ok=True)

    with phase_timer(output_dir, model_id=arch.model_id,
                     task_id=task_id, n_prompts=n_prompts) as cfg:

        # --- Clean forward pass ---
        with torch.no_grad():
            clean_logits, clean_cache = model.run_with_cache(clean_tokens)

        # Extract per-head z at final token position
        clean_z = _extract_z_final_token(
            clean_cache, arch.num_layers, arch.num_heads, clean_tokens
        )  # (N, L, H, D)

        # --- Corrupt forward pass ---
        with torch.no_grad():
            corrupt_logits, corrupt_cache = model.run_with_cache(corrupt_tokens)

        corrupt_z = _extract_z_final_token(
            corrupt_cache, arch.num_layers, arch.num_heads, corrupt_tokens
        )

        # Save activations
        save_npz({"activations": clean_z}, output_dir / "clean_z.npz")
        save_npz({"activations": corrupt_z}, output_dir / "corrupt_z.npz")

        # Save logits (final token only)
        clean_logits_np = clean_logits[:, -1, :].cpu().numpy()  # (N, V)
        corrupt_logits_np = corrupt_logits[:, -1, :].cpu().numpy()
        save_npz({"logits": clean_logits_np}, output_dir / "clean_logits.npz")
        save_npz({"logits": corrupt_logits_np}, output_dir / "corrupt_logits.npz")

        # Save residual stream if requested (for Phase 40 readout)
        if capture_residual:
            resid_hook = tl_utils.get_act_name("resid_post", arch.num_layers - 1)
            clean_resid = clean_cache[resid_hook][:, -1, :].cpu().numpy()
            corrupt_resid = corrupt_cache[resid_hook][:, -1, :].cpu().numpy()
            save_npz({"residual": clean_resid}, output_dir / "clean_resid_final.npz")
            save_npz({"residual": corrupt_resid}, output_dir / "corrupt_resid_final.npz")

        # Save architecture
        save_json(arch.to_dict(), output_dir / "architecture.json")

        cfg["clean_z_shape"] = list(clean_z.shape)
        cfg["corrupt_z_shape"] = list(corrupt_z.shape)
        cfg["clean_logits_shape"] = list(clean_logits_np.shape)
        cfg["capture_residual"] = capture_residual

    return {
        "n_prompts": n_prompts,
        "clean_z_shape": clean_z.shape,
        "architecture": arch.to_dict(),
    }


def _extract_z_final_token(
    cache,
    n_layers: int,
    n_heads: int,
    tokens: torch.Tensor,
) -> np.ndarray:
    """
    Extract per-head z vectors at the final non-padding token.

    Returns (N, L, H, D) numpy array in float32.
    """
    from transformer_lens import utils as tl_utils

    layers = []
    for layer in range(n_layers):
        hook_name = tl_utils.get_act_name("z", layer)
        z = cache[hook_name]  # (N, seq_len, n_heads, d_head)
        # Take final token position
        z_final = z[:, -1, :, :]  # (N, H, D)
        layers.append(z_final.cpu().numpy())

    # Stack: (L, N, H, D) → transpose to (N, L, H, D)
    stacked = np.stack(layers, axis=0)  # (L, N, H, D)
    return stacked.transpose(1, 0, 2, 3).astype(np.float32)  # (N, L, H, D)

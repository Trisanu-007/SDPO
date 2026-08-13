# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Attention map extraction utilities for SDPO teacher/student analysis.

This module provides a context manager (AttentionMapCollector) that hooks into the
last N attention layers of a HuggingFace transformer model during a forward pass,
captures the per-head attention weights, and saves them to disk as .npz files.

Saved file layout (one file per forward call):
    <save_dir>/step_{step:06d}_{role}.npz

    Keys inside the npz:
        "layer_{layer_idx}"  ->  np.ndarray of shape (batch, n_heads, seq_len, seq_len)
                                  (averaged over the batch dimension before saving to
                                   keep file sizes manageable)

Usage example (inside update_policy):
    collector = AttentionMapCollector(
        model=self.actor_module,
        config=attn_map_cfg,
        step=self._attn_map_step,
        role="student",
    )
    with collector:
        outputs = self._forward_micro_batch(model_inputs, ...)
    collector.save()          # writes .npz to disk on rank 0 only

Notes:
- Requires output_attentions=True which disables FlashAttention / fused kernels.
- Only rank 0 writes files to avoid duplicate output across DDP/FSDP workers.
- The model is temporarily put in output_attentions=True mode via monkey-patching
  the config; all other forward kwargs remain unchanged.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: find the list of attention submodules in a HuggingFace model
# ---------------------------------------------------------------------------

def _find_attention_layers(model: nn.Module) -> list[nn.Module]:
    """Return all submodules whose class name contains 'Attention'.

    Works for standard HuggingFace architectures (Qwen2, LlamaAttention, …).
    The list is returned in the order they appear during a depth-first traversal,
    which corresponds to layer order for typical transformers.
    """
    attn_layers: list[nn.Module] = []
    for name, module in model.named_modules():
        cls_name = type(module).__name__
        # Match HF attention layer names; exclude output projection wrappers
        if "Attention" in cls_name and "Output" not in cls_name and "FlashAttention" not in cls_name:
            attn_layers.append((name, module))
    return attn_layers


def _unwrap_model(model: nn.Module) -> nn.Module:
    """Strip FSDP / DDP wrappers to reach the raw HuggingFace model."""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    while isinstance(model, (FSDP, torch.nn.parallel.DistributedDataParallel)):
        model = model.module
    # Handle FSDP2 / FSDPModule wrappers from verl
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

class AttentionMapCollector:
    """Context manager that captures attention weights from the last N layers.

    Args:
        model: The nn.Module (actor or teacher module) to hook into.
        save_dir: Directory where .npz files will be written.
        num_layers_from_end: Number of transformer layers (counting from the last)
            whose attention weights to capture. Default: 4.
        step: The current training step counter (used in file names).
        role: A label string – "student" or "teacher" – used in file names.
        rank: The current distributed rank. Only rank 0 writes files.
    """

    def __init__(
        self,
        model: nn.Module,
        save_dir: str,
        num_layers_from_end: int = 4,
        step: int = 0,
        role: str = "student",
        rank: int = 0,
        attention_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
    ) -> None:
        self.model = model
        self.save_dir = Path(save_dir)
        self.num_layers_from_end = num_layers_from_end
        self.step = step
        self.role = role
        self.rank = rank
        # attention_mask: (batch, seq_len) with 1=valid, 0=padding
        # Used to slice out only the real (non-padding) token rows/columns before saving.
        self.attention_mask = attention_mask
        self.input_ids = input_ids

        # Collected attention tensors: {layer_name: list[Tensor]}
        self._captured: dict[str, list[torch.Tensor]] = {}
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._orig_output_attentions: Optional[bool] = None
        self._hooked_layers: list[tuple[str, nn.Module]] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_raw_model_config(self) -> Optional[object]:
        """Try to get the HuggingFace config from the (possibly wrapped) model."""
        raw = _unwrap_model(self.model)
        return getattr(raw, "config", None)

    def _patch_output_attentions(self, value: bool) -> Optional[bool]:
        """Temporarily set model.config.output_attentions to *value*.

        Returns the original value so it can be restored.
        """
        cfg = self._get_raw_model_config()
        if cfg is None:
            return None
        original = getattr(cfg, "output_attentions", False)
        cfg.output_attentions = value
        return original

    def _build_hook(self, layer_name: str):
        """Return a forward hook function that captures attn_weights."""
        captured = self._captured

        def hook(module, inputs, output):
            # HuggingFace attention layers return a tuple:
            #   (attn_output, attn_weights [optional], past_key_value [optional])
            # attn_weights has shape (batch, n_heads, seq_len, seq_len) when present.
            attn_weights = None
            if isinstance(output, tuple):
                for item in output:
                    if (
                        isinstance(item, torch.Tensor)
                        and item.ndim == 4
                        and item.shape[-2] == item.shape[-1]  # square: (bsz, heads, seq, seq)
                    ):
                        attn_weights = item
                        break
            if attn_weights is not None:
                if layer_name not in captured:
                    captured[layer_name] = []
                captured[layer_name].append(attn_weights.detach().float().cpu())

        return hook

    # ------------------------------------------------------------------
    # Context manager entry / exit
    # ------------------------------------------------------------------

    def __enter__(self):
        # 1. Locate attention layers
        all_attn_layers = _find_attention_layers(self.model)
        if not all_attn_layers:
            logger.warning(
                "[AttentionMapCollector] No attention layers found in model. "
                "Attention maps will not be captured."
            )
            return self

        target_layers = all_attn_layers[-self.num_layers_from_end:]
        self._hooked_layers = target_layers

        # 2. Patch config to enable output_attentions
        self._orig_output_attentions = self._patch_output_attentions(True)

        # 3. Register forward hooks
        for name, module in target_layers:
            hook = self._build_hook(name)
            handle = module.register_forward_hook(hook)
            self._hooks.append(handle)

        logger.info(
            "[AttentionMapCollector] Hooked %d attention layers for role='%s', step=%d",
            len(target_layers),
            self.role,
            self.step,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Remove all hooks
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

        # Restore original output_attentions setting
        if self._orig_output_attentions is not None:
            self._patch_output_attentions(self._orig_output_attentions)

        return False  # do not suppress exceptions

    # ------------------------------------------------------------------
    # Save to disk
    # ------------------------------------------------------------------

    def save(self) -> Optional[Path]:
        """Save captured attention maps to an .npz file.

        Only rank 0 writes to disk. Returns the path that was written (or None).

        File name format:
            <save_dir>/step_{step:06d}_{role}.npz

        Array format per key "layer_<name>":
            shape: (n_heads, n_valid, n_valid)  — non-padding tokens only,
                   averaged over batch dimension.
            If no attention_mask was supplied, shape is (n_heads, seq_len, seq_len).
        """
        if not self._captured:
            logger.warning(
                "[AttentionMapCollector] No attention maps captured for role='%s', step=%d. "
                "Make sure output_attentions is supported by the model architecture.",
                self.role,
                self.step,
            )
            return None

        # Only rank 0 writes files
        is_rank0 = (not dist.is_available()) or (not dist.is_initialized()) or (dist.get_rank() == 0)
        if not is_rank0:
            return None

        self.save_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.save_dir / f"step_{self.step:06d}_{self.role}.npz"

        # Compute valid (non-padding) token count from attention_mask.
        # attention_mask: (batch, seq_len), 1=valid, 0=padding.
        # We take the minimum valid length across the batch so every batch item
        # can be sliced to the same shape before averaging.
        valid_len: Optional[int] = None
        if self.attention_mask is not None:
            mask_cpu = self.attention_mask.cpu()
            # Sum of 1s per batch item gives the valid token count.
            valid_lengths = mask_cpu.sum(dim=-1).long()  # (batch,)
            valid_len = int(valid_lengths.min().item())
            if valid_len <= 0:
                valid_len = None  # fallback: keep full sequence

        arrays: dict[str, np.ndarray] = {}
        for layer_name, tensors in self._captured.items():
            # tensors: list of (batch, n_heads, seq_len, seq_len) tensors
            stacked = torch.cat(tensors, dim=0)  # (total_batch, n_heads, seq, seq)
            if valid_len is not None:
                # Slice to valid token rows and columns only.
                # This works for right-padded sequences where valid tokens are
                # at positions [0 : valid_len].
                stacked = stacked[:, :, :valid_len, :valid_len]
            averaged = stacked.mean(dim=0).numpy()  # (n_heads, n_valid, n_valid)
            safe_key = "layer_" + layer_name.replace(".", "_")
            arrays[safe_key] = averaged

        if self.input_ids is not None:
            input_ids_cpu = self.input_ids.cpu()
            if valid_len is not None:
                input_ids_cpu = input_ids_cpu[:, :valid_len]
            arrays["input_ids"] = input_ids_cpu.numpy()

        np.savez_compressed(str(out_path), **arrays)
        first_shape = next(iter(arrays.values())).shape
        logger.info(
            "[AttentionMapCollector] Saved %d layer maps to %s  shape=%s%s",
            len(arrays),
            out_path,
            first_shape,
            f"  (sliced to {valid_len} valid tokens)" if valid_len else "",
        )
        return out_path


# ---------------------------------------------------------------------------
# Convenience: build collector from OmegaConf / SimpleNamespace config
# ---------------------------------------------------------------------------

def make_attention_collector(
    model: nn.Module,
    attn_map_cfg,
    step: int,
    role: str,
    attention_mask: Optional[torch.Tensor] = None,
    input_ids: Optional[torch.Tensor] = None,
) -> "AttentionMapCollector":
    """Construct an AttentionMapCollector from a config object.

    The config object is expected to have the following fields (all optional
    with sensible defaults):
        save_dir: str               – directory to save .npz files
        num_layers_from_end: int    – how many last layers to capture (default 4)

    Args:
        model: The nn.Module to hook into.
        attn_map_cfg: A config object (OmegaConf DictConfig or SimpleNamespace).
        step: Current step counter.
        role: "student" or "teacher".
        attention_mask: Optional (batch, seq_len) bool/int tensor. When provided,
            the saved attention maps are sliced to only the non-padding token
            rows and columns, significantly reducing file size.
        input_ids: Optional (batch, seq_len) tensor to save alongside maps for visualization.

    Returns:
        An AttentionMapCollector instance (not yet entered as context manager).
    """
    save_dir = attn_map_cfg.get("save_dir", "./attention_maps") if isinstance(attn_map_cfg, dict) else getattr(attn_map_cfg, "save_dir", "./attention_maps")
    num_layers = attn_map_cfg.get("num_layers_from_end", 4) if isinstance(attn_map_cfg, dict) else getattr(attn_map_cfg, "num_layers_from_end", 4)
    rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
    return AttentionMapCollector(
        model=model,
        save_dir=save_dir,
        num_layers_from_end=num_layers,
        step=step,
        role=role,
        rank=rank,
        attention_mask=attention_mask,
        input_ids=input_ids,
    )


# ---------------------------------------------------------------------------
# Standalone visualization helper (for offline use after training)
# ---------------------------------------------------------------------------

def visualize_attention_maps(npz_path: str, output_dir: Optional[str] = None, show: bool = False):
    """Load a saved .npz file and produce one heatmap PNG per layer.

    Args:
        npz_path:   Path to the .npz file produced by AttentionMapCollector.save().
        output_dir: Directory to write PNG files. Defaults to the same directory
                    as npz_path.
        show:       Whether to call plt.show() (useful for interactive notebooks).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for visualization. Install with: pip install matplotlib")

    npz_path = Path(npz_path)
    if output_dir is None:
        output_dir = npz_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(str(npz_path))
    stem = npz_path.stem  # e.g. "step_000001_teacher"

    for layer_key in data.files:
        attn = data[layer_key]  # (n_heads, seq_len, seq_len)
        n_heads = attn.shape[0]

        # -- Mean over heads --
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(attn.mean(axis=0), cmap="viridis", aspect="auto")
        ax.set_title(f"{stem} | {layer_key} | mean over {n_heads} heads")
        ax.set_xlabel("Key position")
        ax.set_ylabel("Query position")
        plt.colorbar(im, ax=ax)
        out_file = output_dir / f"{stem}_{layer_key}_mean.png"
        plt.savefig(str(out_file), dpi=120, bbox_inches="tight")
        plt.close(fig)

        # -- Per-head grid (up to 16 heads) --
        n_show = min(n_heads, 16)
        ncols = 4
        nrows = (n_show + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
        axes_flat = np.array(axes).flatten()
        for h in range(n_show):
            ax = axes_flat[h]
            ax.imshow(attn[h], cmap="viridis", aspect="auto")
            ax.set_title(f"head {h}", fontsize=8)
            ax.axis("off")
        for h in range(n_show, len(axes_flat)):
            axes_flat[h].axis("off")
        fig.suptitle(f"{stem} | {layer_key} | per head", fontsize=10)
        plt.tight_layout()
        out_file = output_dir / f"{stem}_{layer_key}_per_head.png"
        plt.savefig(str(out_file), dpi=100, bbox_inches="tight")
        plt.close(fig)

        logger.info("Saved visualization: %s", out_file)

    if show:
        plt.show()

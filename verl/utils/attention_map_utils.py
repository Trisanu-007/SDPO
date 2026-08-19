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

This module provides AttentionMapCollector, which uses TransformerLens
``run_with_cache`` to capture two complementary activations at layers
28, 29, 30, and 31 during a single forward pass:

  * ``hook_z``       – per-head value-weighted output, shape
                        (batch, seq_len, n_heads, d_head).  Useful for
                        probing what information each head writes.
  * ``hook_pattern`` – softmaxed attention weights, shape
                        (batch, n_heads, seq_len, seq_len).  Required by
                        ``circuitsvis`` for interactive attention visualisation.

Saved file layout (one file per call to ``run_and_capture``):
    <save_dir>/step_{step:06d}_{role}.npz

    Keys inside the npz:
        "layer_{idx}"         -> hook_z,   shape (n_heads, n_valid, d_head)
        "pattern_layer_{idx}" -> hook_pattern, shape (n_heads, n_valid, n_valid)
        "input_ids"           -> token ids, shape (batch, n_valid)  [optional]

    All arrays are averaged over the batch dimension before saving.
    The seq_len axis is sliced to n_valid non-padding tokens when
    an ``attention_mask`` is provided.

Usage example (inside update_policy):

    collector = AttentionMapCollector(
        save_dir=attn_map_cfg.save_dir,
        target_layers=[28, 29, 30, 31],
        step=self._attn_map_step,
        role="student",
    )
    output = collector.run_and_capture(
        model=self.actor_module,          # TransformerLens model / bridge
        input=tokens,
        return_type="logits",
        attention_mask=attention_mask,
        input_ids=input_ids,
    )
    collector.save()                      # writes .npz to disk on rank 0 only

    # Offline circuitsvis visualisation (e.g. in a notebook):
    visualize_attention_maps_circuitsvis(
        npz_path="./attention_maps/step_000001_student.npz",
        model=model,                      # for model.to_str_tokens()
    )

Notes:
- The model must expose a ``run_with_cache`` method compatible with the
  TransformerBridge / HookedRootModule API.
- Only rank 0 writes files to avoid duplicate output across DDP/FSDP workers.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Layer indices to capture (fixed: last 4 layers of a 32-layer model)
# ---------------------------------------------------------------------------
_DEFAULT_TARGET_LAYERS: list[int] = [28, 29, 30, 31]


def _layer_hook_name(layer_idx: int) -> str:
    """Return the TransformerLens activation name for hook_z at *layer_idx*."""
    return f"blocks.{layer_idx}.attn.hook_z"


def _layer_pattern_name(layer_idx: int) -> str:
    """Return the TransformerLens activation name for hook_pattern at *layer_idx*.

    ``hook_pattern`` contains the softmax-normalised attention weights with shape
    (batch, n_heads, seq_len, seq_len) — exactly what circuitsvis expects.
    """
    return f"blocks.{layer_idx}.attn.hook_pattern"


def _build_names_filter(target_layers: list[int]) -> list[str]:
    """Return hook names for both hook_z and hook_pattern at every target layer.

    Both activations are requested in a single ``run_with_cache`` call so that
    only one forward pass is needed.
    """
    names: list[str] = []
    for i in target_layers:
        names.append(_layer_hook_name(i))     # value-weighted output
        names.append(_layer_pattern_name(i))  # attention weight matrix
    return names


# ---------------------------------------------------------------------------
# Model unwrapping and TransformerBridge helpers
# ---------------------------------------------------------------------------

def _unwrap_model(model: Any) -> Any:
    """Unwrap FSDP, DDP, or torch.compile wrappers to access the raw HuggingFace model."""
    raw_model = model
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        while isinstance(raw_model, (FSDP, torch.nn.parallel.DistributedDataParallel)):
            raw_model = raw_model.module
    except Exception:
        pass
    # Unwrap torch.compile wrapper
    if hasattr(raw_model, "_orig_mod"):
        raw_model = raw_model._orig_mod
    # Do NOT follow .module further here: HuggingFace PreTrainedModel objects have
    # internal submodules named .module (e.g. LlamaModel inside LlamaForCausalLM).
    # Following .module past the FSDP/DDP layer would descend into model internals.
    return raw_model


def _get_model_bridge(model: Any) -> Any:
    """Ensure the model exposes ``run_with_cache`` and ``to_str_tokens``.

    If the model does not directly expose ``run_with_cache``, unwrap FSDP/DDP
    wrappers and attach a ``TransformerBridge`` around the existing single model instance.
    This guarantees that the exact same model parameters used in training are used for
    attention pattern capture.
    """
    if hasattr(model, "run_with_cache"):
        return model

    raw_model = _unwrap_model(model)
    if hasattr(raw_model, "run_with_cache"):
        return raw_model

    # Check if a TransformerBridge was already attached to the raw model instance
    if hasattr(raw_model, "_tl_bridge") and hasattr(raw_model._tl_bridge, "run_with_cache"):
        return raw_model._tl_bridge

    # Wrap the existing single model instance using TransformerBridge
    try:
        from transformer_lens.model_bridge import TransformerBridge
        logger.info("[AttentionMapCollector] Wrapping model instance with TransformerBridge...")
        bridge = TransformerBridge(raw_model)
        raw_model._tl_bridge = bridge
        return bridge
    except Exception as exc:
        logger.warning(
            "[AttentionMapCollector] Could not wrap model with TransformerBridge: %s", exc
        )
        return model


# ---------------------------------------------------------------------------
# Main collector class
# ---------------------------------------------------------------------------

class AttentionMapCollector:
    """Captures attention outputs via TransformerLens ``run_with_cache``.

    Instead of registering PyTorch forward hooks, this class delegates the
    entire forward pass to ``model.run_with_cache``, asking TransformerLens
    to cache only the ``hook_z`` activations at the chosen layers.  The
    cached tensors are stored internally and flushed to disk by ``save()``.

    Args:
        save_dir: Directory where .npz files will be written.
        target_layers: List of layer indices whose activations to capture.
            Defaults to [28, 29, 30, 31].
        step: The current training step counter (used in file names).
        role: A label string – "student" or "teacher" – used in file names.
        rank: The current distributed rank. Only rank 0 writes files.
    """

    def __init__(
        self,
        save_dir: str,
        target_layers: Optional[list[int]] = None,
        step: int = 0,
        role: str = "student",
        rank: int = 0,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.target_layers: list[int] = target_layers if target_layers is not None else list(_DEFAULT_TARGET_LAYERS)
        self.step = step
        self.role = role
        self.rank = rank

        # Populated by run_and_capture():
        #   hook_z     – {layer_idx: Tensor (batch, seq_len, n_heads, d_head)}
        #   hook_pattern – {layer_idx: Tensor (batch, n_heads, seq_len, seq_len)}
        self._captured: dict[int, torch.Tensor] = {}
        self._captured_patterns: dict[int, torch.Tensor] = {}

        # Optionally set before / during run_and_capture:
        self._attention_mask: Optional[torch.Tensor] = None
        self._input_ids: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_and_capture(
        self,
        model: Any,
        input: Any,
        return_type: str = "logits",
        attention_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        remove_batch_dim: bool = False,
        reset_hooks_end: bool = True,
        **kwargs,
    ) -> Any:
        """Run ``model.run_with_cache`` and store ``hook_z`` activations.

        Args:
            model: A TransformerLens model or TransformerBridge that exposes
                ``run_with_cache``.
            input: Tokens / string(s) to pass as the first positional argument
                to ``run_with_cache``.
            return_type: Forwarded to ``run_with_cache`` (e.g. "logits" or
                "loss").
            attention_mask: Optional (batch, seq_len) tensor used when saving
                to trim padding.
            input_ids: Optional (batch, seq_len) tensor saved alongside maps
                for offline visualization.
            remove_batch_dim: Passed through to ``run_with_cache``.
            reset_hooks_end: Passed through to ``run_with_cache``.
            **kwargs: Any additional keyword arguments forwarded verbatim to
                ``run_with_cache``.

        Returns:
            The model output (logits, loss, etc.) returned by
            ``run_with_cache``.
        """
        self._attention_mask = attention_mask
        self._input_ids = input_ids
        self._captured = {}
        self._captured_patterns = {}

        # Request both hook_z (value output) and hook_pattern (attention weights)
        # for every target layer in a single forward pass.
        names_filter = _build_names_filter(self.target_layers)

        logger.info(
            "[AttentionMapCollector] Calling run_with_cache with names_filter=%s  "
            "role='%s', step=%d",
            names_filter,
            self.role,
            self.step,
        )

        model_bridge = _get_model_bridge(model)

        # TransformerBridge.run_with_cache signature:
        #   run_with_cache(input, return_cache_object=False, ..., names_filter=..., **kwargs)
        # When return_cache_object=False it returns (output, dict[str, Tensor]).
        result = model_bridge.run_with_cache(
            input,
            return_cache_object=False,
            remove_batch_dim=remove_batch_dim,
            names_filter=names_filter,
            reset_hooks_end=reset_hooks_end,
            return_type=return_type,
            **kwargs,
        )

        # Unpack (output, cache_dict)
        if isinstance(result, tuple) and len(result) == 2:
            output, cache = result
        else:
            # Fallback: model returned only the output (no cache dict)
            logger.warning(
                "[AttentionMapCollector] run_with_cache did not return a (output, cache) "
                "tuple; got %s. No activations captured.",
                type(result),
            )
            return result

        # Store hook_z and hook_pattern tensors by layer index
        for layer_idx in self.target_layers:
            # --- hook_z: (batch, seq_len, n_heads, d_head) ---
            hook_name = _layer_hook_name(layer_idx)
            if hook_name in cache:
                self._captured[layer_idx] = cache[hook_name].detach().float().cpu()
            else:
                logger.warning(
                    "[AttentionMapCollector] hook_z '%s' not found in cache. "
                    "Available keys: %s",
                    hook_name,
                    list(cache.keys()),
                )

            # --- hook_pattern: (batch, n_heads, seq_len, seq_len) ---
            pattern_name = _layer_pattern_name(layer_idx)
            if pattern_name in cache:
                self._captured_patterns[layer_idx] = cache[pattern_name].detach().float().cpu()
            else:
                logger.warning(
                    "[AttentionMapCollector] hook_pattern '%s' not found in cache.",
                    pattern_name,
                )

        logger.info(
            "[AttentionMapCollector] Captured hook_z for %d / %d layers: %s  "
            "| hook_pattern for %d / %d layers: %s",
            len(self._captured),
            len(self.target_layers),
            sorted(self._captured.keys()),
            len(self._captured_patterns),
            len(self.target_layers),
            sorted(self._captured_patterns.keys()),
        )

        return output

    # ------------------------------------------------------------------
    # Save to disk
    # ------------------------------------------------------------------

    def save(self) -> Optional[Path]:
        """Save captured attention maps to an .npz file.

        Only rank 0 writes to disk. Returns the path that was written (or None).

        File name format:
            <save_dir>/step_{step:06d}_{role}.npz

        Array format per key "layer_<idx>":
            shape: (n_heads, n_valid, d_head)  — non-padding positions only,
                   averaged over batch dimension.
            If no attention_mask was supplied, shape is (n_heads, seq_len, d_head).

        hook_z layout (TransformerLens):
            (batch, seq_len, n_heads, d_head)
        We permute to (batch, n_heads, seq_len, d_head) then average over batch
        and slice the seq_len axis to ``n_valid``.
        """
        if not self._captured:
            logger.warning(
                "[AttentionMapCollector] No attention maps captured for role='%s', step=%d. "
                "Make sure run_and_capture() was called before save().",
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

        # Determine valid (non-padding) token count from attention_mask.
        # attention_mask: (batch, seq_len), 1=valid, 0=padding.
        valid_len: Optional[int] = None
        if self._attention_mask is not None:
            mask_cpu = self._attention_mask.cpu()
            valid_lengths = mask_cpu.sum(dim=-1).long()  # (batch,)
            valid_len = int(valid_lengths.min().item())
            if valid_len <= 0:
                valid_len = None  # fallback: keep full sequence

        arrays: dict[str, np.ndarray] = {}

        # --- hook_z arrays: (n_heads, n_valid, d_head) ---
        for layer_idx, tensor in self._captured.items():
            # tensor shape: (batch, seq_len, n_heads, d_head)
            # Permute to (batch, n_heads, seq_len, d_head)
            tensor = tensor.permute(0, 2, 1, 3)
            if valid_len is not None:
                tensor = tensor[:, :, :valid_len, :]  # (batch, n_heads, n_valid, d_head)
            averaged = tensor.mean(dim=0).numpy()     # (n_heads, n_valid, d_head)
            arrays[f"layer_{layer_idx}"] = averaged

        # --- hook_pattern arrays: (n_heads, n_valid, n_valid) ---
        # These are the softmax attention weights needed by circuitsvis.
        for layer_idx, pattern in self._captured_patterns.items():
            # pattern shape: (batch, n_heads, seq_len, seq_len)
            if valid_len is not None:
                pattern = pattern[:, :, :valid_len, :valid_len]  # trim padding
            averaged_pat = pattern.mean(dim=0).numpy()  # (n_heads, n_valid, n_valid)
            arrays[f"pattern_layer_{layer_idx}"] = averaged_pat

        if self._input_ids is not None:
            input_ids_cpu = self._input_ids.cpu()
            if valid_len is not None:
                input_ids_cpu = input_ids_cpu[:, :valid_len]
            arrays["input_ids"] = input_ids_cpu.numpy()

        np.savez_compressed(str(out_path), **arrays)
        first_key = next(iter(arrays))
        first_shape = arrays[first_key].shape
        logger.info(
            "[AttentionMapCollector] Saved %d hook_z + %d hook_pattern arrays to %s  "
            "first_shape=%s%s",
            len(self._captured),
            len(self._captured_patterns),
            out_path,
            first_shape,
            f"  (sliced to {valid_len} valid tokens)" if valid_len else "",
        )
        return out_path


# ---------------------------------------------------------------------------
# Convenience: build collector from OmegaConf / SimpleNamespace config
# ---------------------------------------------------------------------------

def make_attention_collector(
    attn_map_cfg,
    step: int,
    role: str,
) -> "AttentionMapCollector":
    """Construct an AttentionMapCollector from a config object.

    The config object is expected to have the following fields (all optional
    with sensible defaults):
        save_dir: str            – directory to save .npz files (default "./attention_maps")
        target_layers: list[int] – layer indices to capture (default [28, 29, 30, 31])

    Args:
        attn_map_cfg: A config object (OmegaConf DictConfig, dict, or SimpleNamespace).
        step: Current step counter.
        role: "student" or "teacher".

    Returns:
        An AttentionMapCollector instance ready for use with ``run_and_capture``.
    """
    if isinstance(attn_map_cfg, dict):
        save_dir = attn_map_cfg.get("save_dir", "./attention_maps")
        target_layers = attn_map_cfg.get("target_layers", _DEFAULT_TARGET_LAYERS)
    else:
        save_dir = getattr(attn_map_cfg, "save_dir", "./attention_maps")
        target_layers = getattr(attn_map_cfg, "target_layers", _DEFAULT_TARGET_LAYERS)

    rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
    return AttentionMapCollector(
        save_dir=save_dir,
        target_layers=list(target_layers),
        step=step,
        role=role,
        rank=rank,
    )


# ---------------------------------------------------------------------------
# Standalone visualization helper (for offline use after training)
# ---------------------------------------------------------------------------

def visualize_attention_maps(npz_path: str, output_dir: Optional[str] = None, show: bool = False):
    """Load a saved .npz file and produce matplotlib heatmap PNGs for hook_z arrays.

    Only processes ``layer_{idx}`` keys (hook_z, shape ``(n_heads, seq_len, d_head)``).
    ``pattern_layer_{idx}`` keys (hook_pattern) are skipped here — use
    :func:`visualize_attention_maps_circuitsvis` for interactive attention pattern
    visualisation.

    Each layer produces two files:
        - ``{stem}_{layer_key}_mean_over_heads.png``  — mean over heads (seq × d_head)
        - ``{stem}_{layer_key}_per_head.png``          — grid, one subplot per head

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
        # Skip non-hook_z arrays: input_ids and pattern_layer_* (attention weights)
        # pattern_layer_* are visualised via visualize_attention_maps_circuitsvis.
        if layer_key == "input_ids" or layer_key.startswith("pattern_layer_"):
            continue
        arr = data[layer_key]  # hook_z: (n_heads, n_valid, d_head)
        n_heads = arr.shape[0]

        # -- Mean over heads: (n_valid, d_head) --
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(arr.mean(axis=0), cmap="viridis", aspect="auto")
        ax.set_title(f"{stem} | {layer_key} (hook_z) | mean over {n_heads} heads")
        ax.set_xlabel("d_head dimension")
        ax.set_ylabel("Sequence position")
        plt.colorbar(im, ax=ax)
        out_file = output_dir / f"{stem}_{layer_key}_mean_over_heads.png"
        plt.savefig(str(out_file), dpi=120, bbox_inches="tight")
        plt.close(fig)

        # -- Per-head grid (up to 16 heads): each subplot is (n_valid, d_head) --
        n_show = min(n_heads, 16)
        ncols = 4
        nrows = (n_show + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
        axes_flat = np.array(axes).flatten()
        for h in range(n_show):
            ax = axes_flat[h]
            ax.imshow(arr[h], cmap="viridis", aspect="auto")
            ax.set_title(f"head {h}", fontsize=8)
            ax.axis("off")
        for h in range(n_show, len(axes_flat)):
            axes_flat[h].axis("off")
        fig.suptitle(f"{stem} | {layer_key} (hook_z) | per head", fontsize=10)
        plt.tight_layout()
        out_file = output_dir / f"{stem}_{layer_key}_per_head.png"
        plt.savefig(str(out_file), dpi=100, bbox_inches="tight")
        plt.close(fig)

        logger.info("Saved visualization: %s", out_file)

    if show:
        plt.show()


# ---------------------------------------------------------------------------
# CircuitsVis visualization helper (for offline / notebook use)
# ---------------------------------------------------------------------------

def visualize_attention_maps_circuitsvis(
    npz_path: str,
    model: Optional[Any] = None,
    str_tokens: Optional[list[str]] = None,
    output_dir: Optional[str] = None,
) -> dict[str, Any]:
    """Visualise attention patterns from a saved .npz file using circuitsvis.

    Loads the ``pattern_layer_{idx}`` arrays (shape ``(n_heads, seq_len, seq_len)``)
    written by :meth:`AttentionMapCollector.save` and renders them with
    ``circuitsvis.attention.attention_heads``.

    Two modes for token labels (tried in order):
      1. ``str_tokens`` passed directly – a list of string tokens.
      2. ``model`` provided – calls ``model.to_str_tokens(input_ids)`` using the
         ``input_ids`` array stored in the .npz file.
      3. Fallback – positional labels ``["0", "1", ...]``.

    Args:
        npz_path:   Path to the .npz file produced by
                    :meth:`AttentionMapCollector.save`.
        model:      Optional TransformerLens model. Used only to convert the
                    saved ``input_ids`` to string tokens via
                    ``model.to_str_tokens``.  Pass ``None`` if you supply
                    ``str_tokens`` directly or are happy with positional labels.
        str_tokens: Optional pre-computed list of string tokens to use as
                    labels.  Takes priority over ``model``.
        output_dir: Directory to write per-layer ``.html`` files.  Defaults to
                    the same directory as ``npz_path``.

    Returns:
        A dict mapping ``"pattern_layer_{idx}"`` to the circuitsvis HTML object
        returned by ``cv.attention.attention_heads``.  You can call
        ``.show()`` on each value in a Jupyter notebook, or iterate over
        ``.html`` to get the raw HTML string.

    Example (notebook)::

        from verl.utils.attention_map_utils import visualize_attention_maps_circuitsvis

        vis = visualize_attention_maps_circuitsvis(
            npz_path="./attention_maps/step_000001_student.npz",
            model=model,          # optional: for token labels
        )
        # Display layer 28 in the notebook:
        vis["pattern_layer_28"].show()

    Raises:
        ImportError: if ``circuitsvis`` is not installed.
    """
    try:
        import circuitsvis as cv
    except ImportError:
        raise ImportError(
            "circuitsvis is required for this visualisation. "
            "Install with: pip install circuitsvis"
        )

    npz_path = Path(npz_path)
    if output_dir is None:
        output_dir = npz_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(str(npz_path))
    stem = npz_path.stem  # e.g. "step_000001_student"

    # ------------------------------------------------------------------
    # Resolve token labels
    # ------------------------------------------------------------------
    tokens: Optional[list[str]] = None

    if str_tokens is not None:
        # Caller passed labels directly – use as-is.
        tokens = list(str_tokens)
    elif model is not None and "input_ids" in data:
        # Convert stored input_ids to string tokens via the model.
        # data["input_ids"] shape: (batch, seq_len); use first batch item.
        ids_np = data["input_ids"]
        first_seq = ids_np[0] if ids_np.ndim == 2 else ids_np
        import torch as _torch
        ids_tensor = _torch.tensor(first_seq, dtype=_torch.long)
        try:
            model_bridge = _get_model_bridge(model)
            if hasattr(model_bridge, "to_str_tokens"):
                tokens = model_bridge.to_str_tokens(ids_tensor)
            elif hasattr(model, "to_str_tokens"):
                tokens = model.to_str_tokens(ids_tensor)
            else:
                raise AttributeError("Model does not expose to_str_tokens")
        except Exception as exc:
            logger.warning(
                "[visualize_attention_maps_circuitsvis] model.to_str_tokens failed (%s); "
                "falling back to positional labels.",
                exc,
            )
            tokens = None

    # ------------------------------------------------------------------
    # Build circuitsvis visualisations for each pattern_layer_* key
    # ------------------------------------------------------------------
    results: dict[str, Any] = {}

    pattern_keys = [k for k in data.files if k.startswith("pattern_layer_")]
    if not pattern_keys:
        logger.warning(
            "[visualize_attention_maps_circuitsvis] No 'pattern_layer_*' keys found in %s. "
            "Make sure the .npz was saved by AttentionMapCollector (which now also caches "
            "hook_pattern). Keys present: %s",
            npz_path,
            list(data.files),
        )
        return results

    for layer_key in sorted(pattern_keys):
        # pattern: (n_heads, seq_len, seq_len) – already averaged over batch
        pattern = data[layer_key]
        n_heads, seq_len, _ = pattern.shape

        # Build fallback positional labels if we still have none.
        effective_tokens: list[str] = tokens if tokens is not None else [str(i) for i in range(seq_len)]

        # circuitsvis expects:
        #   tokens:    list[str]                  length = seq_len
        #   attention: Tensor / ndarray of shape  (n_heads, seq_len, seq_len)
        import torch as _torch
        attention_tensor = _torch.tensor(pattern)  # (n_heads, seq_len, seq_len)

        vis = cv.attention.attention_heads(
            tokens=effective_tokens,
            attention=attention_tensor,
        )

        results[layer_key] = vis

        # Write HTML file for offline viewing
        html_out = output_dir / f"{stem}_{layer_key}.html"
        try:
            html_str = vis.html() if callable(getattr(vis, "html", None)) else str(vis)
            html_out.write_text(html_str, encoding="utf-8")
            logger.info(
                "[visualize_attention_maps_circuitsvis] Saved %s  (%d heads, %d tokens)",
                html_out,
                n_heads,
                len(effective_tokens),
            )
        except Exception as exc:
            logger.warning(
                "[visualize_attention_maps_circuitsvis] Could not write HTML for %s: %s",
                layer_key,
                exc,
            )

    return results

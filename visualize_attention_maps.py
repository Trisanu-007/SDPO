#!/usr/bin/env python3
"""
visualize_attention_maps.py
============================
Offline visualisation of attention maps saved by SDPO's AttentionMapCollector.

Saved file layout (one per forward call):
    <ATTN_SAVE_DIR>/step_{step:06d}_{role}.npz

NPZ contents:
    "layer_<name>"  -> np.ndarray  (n_heads, seq_len, seq_len)   — averaged over batch
    "input_ids"     -> np.ndarray  (batch, seq_len)               — optional

Usage
-----
    # Run from the SDPO project root on Sharanga (or any machine with the maps):
    python visualize_attention_maps.py

    # Override the attention-map directory:
    python visualize_attention_maps.py --attn-dir /scratch/hrishikesh/users/tri/sdpo_results/attention_maps_verl

    # Control output directory, number of token bins, figure dpi:
    python visualize_attention_maps.py \\
        --attn-dir /scratch/hrishikesh/users/tri/sdpo_results/attention_maps_verl \\
        --out-dir  ./attn_viz \\
        --n-bins   64 \\
        --dpi      150 \\
        --total-epochs 3

Output (written to <out-dir>/):
    1. grouped_density/     — token-bin density heatmaps per step per role
    2. output_token_attn/   — zoom into the response token region per step per role
    3. trends/              — per-step trend lines (entropy, sparsity, mean response attn)
    4. epoch_evolution/     — side-by-side panels comparing epochs for the same sample
    5. student_vs_teacher/  — side-by-side comparison at same step
    6. per_head/            — per-head attention grids (first/last step)
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Optional GPU acceleration via CuPy (drop-in numpy replacement on CUDA)
# ---------------------------------------------------------------------------
# Set xp = cupy when --gpu is passed and cupy is available; else xp = numpy.
# All compute-heavy functions (binning, entropy, sparsity) use xp so they
# run on GPU transparently. matplotlib always receives plain numpy arrays.

_GPU_REQUESTED = False   # updated in main() after arg parsing
xp = np                  # default: use numpy (CPU)


def _init_gpu(requested: bool) -> bool:
    """Try to enable CuPy. Returns True if GPU is active."""
    global xp, _GPU_REQUESTED
    _GPU_REQUESTED = requested
    if not requested:
        return False
    try:
        import cupy as cp
        # Quick smoke-test: allocate a tiny array
        _ = cp.array([1.0])
        xp = cp
        print("  [GPU] CuPy detected — compute ops will run on GPU.", flush=True)
        return True
    except Exception as e:
        print(f"  [GPU] CuPy unavailable ({e}), falling back to CPU numpy.", flush=True)
        xp = np
        return False


def _to_numpy(arr) -> np.ndarray:
    """Convert cupy array to numpy (no-op if already numpy)."""
    if xp is not np:
        return xp.asnumpy(arr)
    return arr

# ---------------------------------------------------------------------------
# Matplotlib setup – use non-interactive Agg backend so this runs headless
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
STUDENT_COLOR = "#4C9BE8"   # blue
TEACHER_COLOR = "#E8824C"   # orange
CMAPS = {"student": "Blues", "teacher": "Oranges"}
ROLES = ("student", "teacher")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_ATTN_DIR = "/scratch/hrishikesh/users/tri/sdpo_results/attention_maps_verl"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualise SDPO attention maps")
    p.add_argument(
        "--attn-dir",
        default=DEFAULT_ATTN_DIR,
        help="Directory containing the .npz attention-map files",
    )
    p.add_argument(
        "--out-dir",
        default="./attn_viz",
        help="Output directory for all generated figures",
    )
    p.add_argument(
        "--n-bins",
        type=int,
        default=64,
        help="Number of token bins used for grouped density heatmaps (default: 64)",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=130,
        help="DPI for saved figures (default: 130)",
    )
    p.add_argument(
        "--total-epochs",
        type=int,
        default=3,
        help="Total number of training epochs (used to infer epoch boundaries)",
    )
    p.add_argument(
        "--prompt-len",
        type=int,
        default=None,
        help=(
            "Known prompt token length. When provided, the output-token region "
            "is defined as [prompt_len:]. Otherwise it is guessed from the "
            "causal attention pattern."
        ),
    )
    p.add_argument(
        "--max-steps-plot",
        type=int,
        default=200,
        help="Maximum number of steps to include in trend plots (default: 200)",
    )
    p.add_argument(
        "--gpu",
        action="store_true",
        default=False,
        help=(
            "Use GPU (CuPy) for compute-heavy ops: binning, entropy, sparsity. "
            "Requires cupy to be installed in the active conda env. "
            "Falls back to CPU numpy automatically if cupy is unavailable."
        ),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

_NPZ_RE = re.compile(r"step_(\d+)_(student|teacher)\.npz$")


def discover_files(attn_dir: str) -> dict[int, dict[str, Path]]:
    """
    Scan *attn_dir* for step_XXXXXX_{role}.npz files.

    Returns:
        { step_int: { "student": Path, "teacher": Path } }
    """
    files: dict[int, dict[str, Path]] = {}
    pattern = os.path.join(attn_dir, "step_*_*.npz")
    for fpath in sorted(glob.glob(pattern)):
        m = _NPZ_RE.search(os.path.basename(fpath))
        if m:
            step = int(m.group(1))
            role = m.group(2)
            files.setdefault(step, {})[role] = Path(fpath)
    return files


def load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(str(path))
    return {k: data[k] for k in data.files}


def get_attn_mean(arrays: dict) -> np.ndarray:
    """Mean-over-heads attention matrix from a single npz dict.

    Returns shape (seq_len, seq_len) as a plain numpy array.
    """
    layer_keys = [k for k in arrays if k.startswith("layer_")]
    if not layer_keys:
        raise KeyError(f"No layer_* keys found. Available: {list(arrays.keys())}")
    mats = []
    for lk in layer_keys:
        a = xp.asarray(arrays[lk])  # move to GPU if xp=cupy
        mats.append(a.mean(axis=0))  # (seq, seq)
    result = xp.stack(mats, axis=0).mean(axis=0)  # (seq, seq)
    return _to_numpy(result)  # always return numpy for downstream use


def guess_prompt_len(attn_mean: np.ndarray) -> int:
    """
    Heuristic: in a causal LM, output (response) tokens attend to ALL previous
    tokens. The prompt tokens only attend to themselves. Look for the first
    position whose outgoing attention spread exceeds a threshold.

    Falls back to seq_len // 4 if heuristic fails.
    """
    seq = attn_mean.shape[0]
    threshold = 1.0 / seq
    spread = (attn_mean > threshold).sum(axis=-1)  # (seq,)
    median_spread = np.median(spread[:max(1, seq // 4)])
    candidates = np.where(spread > 2 * median_spread)[0]
    if len(candidates) > 0:
        return int(candidates[0])
    return seq // 4


# ---------------------------------------------------------------------------
# Token binning
# ---------------------------------------------------------------------------

def bin_attention(attn: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Average-pool a (seq, seq) attention matrix into (n_bins, n_bins).
    Uses xp (numpy or cupy) reshape+mean — no Python loops.
    Handles sequences shorter than n_bins gracefully.
    """
    arr = xp.asarray(attn)
    seq = arr.shape[0]
    actual_bins = min(n_bins, seq)
    # Pad seq to be evenly divisible by actual_bins
    rem = seq % actual_bins
    if rem != 0:
        pad = actual_bins - rem
        arr = xp.pad(arr, ((0, pad), (0, pad)), mode="constant", constant_values=0.0)
    padded_seq = arr.shape[0]
    bin_size = padded_seq // actual_bins
    result = (
        arr
        .reshape(actual_bins, bin_size, actual_bins, bin_size)
        .mean(axis=(1, 3))
    )
    return _to_numpy(result).astype(np.float32)


def make_bin_labels(seq_len: int, n_bins: int, prompt_len: int) -> list:
    """Tick labels: 'P0', 'P1', ..., 'R0', 'R1', ... where P=prompt, R=response."""
    actual_bins = min(n_bins, seq_len)
    # Bin boundaries after padding (same logic as bin_attention)
    rem = seq_len % actual_bins
    padded_seq = seq_len + ((actual_bins - rem) if rem != 0 else 0)
    bin_size = padded_seq // actual_bins
    labels = []
    r_counter = 0
    p_counter = 0
    for i in range(actual_bins):
        mid = int((i + 0.5) * bin_size)
        if mid < prompt_len:
            labels.append(f"P{p_counter}")
            p_counter += 1
        else:
            labels.append(f"R{r_counter}")
            r_counter += 1
    return labels


# ---------------------------------------------------------------------------
# Metrics per step
# ---------------------------------------------------------------------------

def attention_entropy(attn_mean: np.ndarray) -> float:
    """Mean per-row entropy of the (seq, seq) attention matrix (nats). GPU-aware."""
    arr = xp.asarray(attn_mean)
    eps = 1e-12
    p = arr + eps
    p = p / p.sum(axis=-1, keepdims=True)
    ent = -(p * xp.log(p)).sum(axis=-1)
    return float(_to_numpy(ent).mean())


def attention_sparsity(attn_mean: np.ndarray, threshold: float = 0.01) -> float:
    """Fraction of attention weights below *threshold* (proxy for sparsity). GPU-aware."""
    arr = xp.asarray(attn_mean)
    return float(_to_numpy((arr < threshold)).mean())


def mean_response_attention(attn_mean: np.ndarray, prompt_len: int) -> float:
    """
    Mean attention that response tokens (rows prompt_len:) place on other
    response tokens (columns prompt_len:). GPU-aware.
    """
    if prompt_len >= attn_mean.shape[0]:
        return 0.0
    arr = xp.asarray(attn_mean)
    sub = arr[prompt_len:, prompt_len:]
    return float(_to_numpy(sub).mean())


# ---------------------------------------------------------------------------
# Plot 1 — grouped density heatmap
# ---------------------------------------------------------------------------

def plot_grouped_density(
    attn_mean: np.ndarray,
    step: int,
    role: str,
    prompt_len: int,
    n_bins: int,
    out_dir: Path,
    dpi: int,
) -> None:
    binned = bin_attention(attn_mean, n_bins)
    seq_len = attn_mean.shape[0]
    labels = make_bin_labels(seq_len, n_bins, prompt_len)

    actual_bins = binned.shape[0]
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(binned, cmap=CMAPS[role], aspect="auto", interpolation="nearest")

    # Overlay a rectangle around the response<->response block
    r_start_bin = next((i for i, lb in enumerate(labels) if lb.startswith("R")), actual_bins)
    if r_start_bin < actual_bins:
        from matplotlib.patches import Rectangle
        width = actual_bins - r_start_bin
        rect = Rectangle(
            (r_start_bin - 0.5, r_start_bin - 0.5),
            width, width,
            linewidth=2, edgecolor="gold", facecolor="none", linestyle="--",
            label="Response<->Response"
        )
        ax.add_patch(rect)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.7)

    # Separator lines between prompt and response blocks
    if r_start_bin < actual_bins:
        ax.axvline(r_start_bin - 0.5, color="white", linewidth=1.2, linestyle=":")
        ax.axhline(r_start_bin - 0.5, color="white", linewidth=1.2, linestyle=":")

    tick_step = max(1, actual_bins // 16)
    ticks = list(range(0, actual_bins, tick_step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([labels[t] for t in ticks], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(ticks)
    ax.set_yticklabels([labels[t] for t in ticks], fontsize=7)

    ax.set_xlabel("Key token bin", fontsize=11)
    ax.set_ylabel("Query token bin", fontsize=11)
    ax.set_title(
        f"Grouped Attention Density  |  {role.capitalize()}  |  Step {step}\n"
        f"seq_len={seq_len}, prompt~{prompt_len} tok, bins={actual_bins}",
        fontsize=12,
    )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.1)
    plt.colorbar(im, cax=cax, label="Mean attention weight")

    plt.tight_layout()
    out_path = out_dir / f"step_{step:06d}_{role}_grouped_density.png"
    plt.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [grouped_density] {out_path.name}")


# ---------------------------------------------------------------------------
# Plot 2 — output-token focused attention
# ---------------------------------------------------------------------------

def plot_output_token_attn(
    attn_mean: np.ndarray,
    step: int,
    role: str,
    prompt_len: int,
    n_bins: int,
    out_dir: Path,
    dpi: int,
) -> None:
    """
    Show the response-token submatrix (rows=response queries, cols=all tokens),
    binned for readability.
    """
    seq_len = attn_mean.shape[0]
    if prompt_len >= seq_len:
        return

    resp_attn = attn_mean[prompt_len:, :]  # (resp_len, seq_len)
    resp_len = resp_attn.shape[0]

    actual_col_bins = min(n_bins, seq_len)
    actual_row_bins = min(n_bins, resp_len)
    col_bin_size = seq_len / actual_col_bins
    row_bin_size = resp_len / actual_row_bins

    binned = np.zeros((actual_row_bins, actual_col_bins), dtype=np.float32)
    for i in range(actual_row_bins):
        r0 = int(i * row_bin_size)
        r1 = max(int((i + 1) * row_bin_size), r0 + 1)
        for j in range(actual_col_bins):
            c0 = int(j * col_bin_size)
            c1 = max(int((j + 1) * col_bin_size), c0 + 1)
            binned[i, j] = resp_attn[r0:r1, c0:c1].mean()

    prompt_col_bins = int(prompt_len / col_bin_size)
    col_labels = [
        f"P{j}" if j < prompt_col_bins else f"R{j - prompt_col_bins}"
        for j in range(actual_col_bins)
    ]

    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(binned, cmap=CMAPS[role], aspect="auto", interpolation="nearest")

    if prompt_col_bins < actual_col_bins:
        ax.axvline(prompt_col_bins - 0.5, color="gold", linewidth=1.5, linestyle="--",
                   label="Prompt/Response boundary")
        ax.legend(loc="upper left", fontsize=8, framealpha=0.7)

    tick_step = max(1, actual_col_bins // 16)
    col_ticks = list(range(0, actual_col_bins, tick_step))
    ax.set_xticks(col_ticks)
    ax.set_xticklabels([col_labels[t] for t in col_ticks], rotation=45, ha="right", fontsize=7)

    row_tick_step = max(1, actual_row_bins // 8)
    row_ticks = list(range(0, actual_row_bins, row_tick_step))
    ax.set_yticks(row_ticks)
    ax.set_yticklabels([f"R{t}" for t in row_ticks], fontsize=7)

    ax.set_xlabel("Key token bin (P=prompt, R=response)", fontsize=11)
    ax.set_ylabel("Response token query bin", fontsize=11)
    ax.set_title(
        f"Output-Token Attention  |  {role.capitalize()}  |  Step {step}\n"
        f"resp_len={resp_len} tok, prompt_len={prompt_len} tok",
        fontsize=12,
    )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    plt.colorbar(im, cax=cax, label="Mean attention weight")

    plt.tight_layout()
    out_path = out_dir / f"step_{step:06d}_{role}_output_token_attn.png"
    plt.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [output_token_attn] {out_path.name}")


# ---------------------------------------------------------------------------
# Plot 3 — trend lines over steps
# ---------------------------------------------------------------------------

def plot_trends(
    metrics: dict,
    out_dir: Path,
    dpi: int,
    epoch_boundaries: Optional[list] = None,
) -> None:
    """
    metrics: { role: { step: { metric_name: value } } }
    """
    metric_names = ["entropy", "sparsity", "mean_resp_attn"]
    metric_labels = {
        "entropy":        "Mean Attention Entropy (nats)",
        "sparsity":       "Attention Sparsity (frac < 0.01)",
        "mean_resp_attn": "Mean Response Self-Attention Weight",
    }

    fig, axes = plt.subplots(len(metric_names), 1, figsize=(12, 4 * len(metric_names)), sharex=True)
    if len(metric_names) == 1:
        axes = [axes]

    for ax, mname in zip(axes, metric_names):
        for role, color, ls in [("student", STUDENT_COLOR, "-"), ("teacher", TEACHER_COLOR, "--")]:
            if role not in metrics or not metrics[role]:
                continue
            steps = sorted(metrics[role].keys())
            vals = [metrics[role][s].get(mname, float("nan")) for s in steps]
            ax.plot(steps, vals, color=color, linestyle=ls, linewidth=2,
                    marker="o", markersize=3, label=role.capitalize(), alpha=0.85)

        # Draw epoch boundary lines
        if epoch_boundaries:
            ylims = ax.get_ylim()
            for eb in epoch_boundaries[1:]:  # skip first (step 0)
                ax.axvline(eb, color="gray", linewidth=1, linestyle=":", alpha=0.7)

        ax.set_ylabel(metric_labels[mname], fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("#f9f9f9")

    # Add epoch labels on top of the first subplot
    if epoch_boundaries and len(axes) > 0:
        ax0 = axes[0]
        for ei, eb in enumerate(epoch_boundaries):
            next_eb = epoch_boundaries[ei + 1] if ei + 1 < len(epoch_boundaries) else None
            mid = (eb + next_eb) / 2 if next_eb else eb
            ax0.text(mid, ax0.get_ylim()[1], f"Ep {ei+1}", fontsize=8,
                     color="gray", ha="center", va="bottom")

    axes[-1].set_xlabel("Training Step", fontsize=11)
    fig.suptitle("Attention Map Metrics Over Training Steps", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_path = out_dir / "attention_trends.png"
    plt.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [trends] {out_path.name}")


# ---------------------------------------------------------------------------
# Plot 4 — epoch-over-epoch evolution (grouped density, same step index)
# ---------------------------------------------------------------------------

def plot_epoch_evolution(
    step_files: dict,
    epoch_boundaries: list,
    total_epochs: int,
    role: str,
    n_bins: int,
    prompt_len_override: Optional[int],
    out_dir: Path,
    dpi: int,
) -> None:
    """
    For each epoch, pick the first available step inside that epoch's range
    and show grouped density heatmaps side-by-side.
    """
    if not epoch_boundaries:
        return

    all_steps = sorted(step_files.keys())
    if not all_steps:
        return

    epoch_ranges = []
    for i, eb in enumerate(epoch_boundaries):
        next_eb = epoch_boundaries[i + 1] if i + 1 < len(epoch_boundaries) else all_steps[-1] + 1
        epoch_ranges.append((eb, next_eb))

    epoch_steps = []  # (epoch_num, step)
    for epoch_idx, (start, end) in enumerate(epoch_ranges):
        candidates = [s for s in all_steps if start <= s < end and role in step_files[s]]
        if candidates:
            epoch_steps.append((epoch_idx + 1, candidates[0]))

    if len(epoch_steps) < 2:
        print(f"  [epoch_evolution] Not enough epochs found for role={role}, skipping.")
        return

    n = len(epoch_steps)
    fig, axes = plt.subplots(1, n, figsize=(9 * n, 8))
    if n == 1:
        axes = [axes]

    for ax, (epoch_num, step) in zip(axes, epoch_steps):
        arrays = load_npz(step_files[step][role])
        attn_mean = get_attn_mean(arrays)
        seq_len = attn_mean.shape[0]

        prompt_len = prompt_len_override or guess_prompt_len(attn_mean)
        binned = bin_attention(attn_mean, n_bins)
        actual_bins = binned.shape[0]
        labels = make_bin_labels(seq_len, actual_bins, prompt_len)

        im = ax.imshow(binned, cmap=CMAPS[role], aspect="auto", interpolation="nearest")

        r_start_bin = next((i for i, lb in enumerate(labels) if lb.startswith("R")), actual_bins)
        if r_start_bin < actual_bins:
            from matplotlib.patches import Rectangle
            rect = Rectangle(
                (r_start_bin - 0.5, r_start_bin - 0.5),
                actual_bins - r_start_bin, actual_bins - r_start_bin,
                linewidth=2, edgecolor="gold", facecolor="none", linestyle="--",
            )
            ax.add_patch(rect)
            ax.axvline(r_start_bin - 0.5, color="white", linewidth=1.2, linestyle=":")
            ax.axhline(r_start_bin - 0.5, color="white", linewidth=1.2, linestyle=":")

        tick_step = max(1, actual_bins // 12)
        ticks = list(range(0, actual_bins, tick_step))
        ax.set_xticks(ticks)
        ax.set_xticklabels([labels[t] for t in ticks], rotation=45, ha="right", fontsize=7)
        ax.set_yticks(ticks)
        ax.set_yticklabels([labels[t] for t in ticks], fontsize=7)
        ax.set_title(f"Epoch {epoch_num}  |  Step {step}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Key token bin", fontsize=10)
        ax.set_ylabel("Query token bin", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Attn weight")

    fig.suptitle(
        f"Epoch-over-Epoch Attention Evolution  |  {role.capitalize()}  |  Same-index step",
        fontsize=15, fontweight="bold",
    )
    plt.tight_layout()

    out_path = out_dir / f"epoch_evolution_{role}.png"
    plt.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [epoch_evolution] {out_path.name}")


# ---------------------------------------------------------------------------
# Plot 5 — Student vs Teacher comparison side by side for same step
# ---------------------------------------------------------------------------

def plot_student_vs_teacher(
    student_attn: np.ndarray,
    teacher_attn: np.ndarray,
    step: int,
    prompt_len: int,
    n_bins: int,
    out_dir: Path,
    dpi: int,
) -> None:
    """Side-by-side grouped density: student (left) vs teacher (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    for ax, attn_mean, role in [(axes[0], student_attn, "student"), (axes[1], teacher_attn, "teacher")]:
        seq_len = attn_mean.shape[0]
        binned = bin_attention(attn_mean, n_bins)
        actual_bins = binned.shape[0]
        labels = make_bin_labels(seq_len, actual_bins, prompt_len)

        im = ax.imshow(binned, cmap=CMAPS[role], aspect="auto", interpolation="nearest")

        r_start_bin = next((i for i, lb in enumerate(labels) if lb.startswith("R")), actual_bins)
        if r_start_bin < actual_bins:
            from matplotlib.patches import Rectangle
            rect = Rectangle(
                (r_start_bin - 0.5, r_start_bin - 0.5),
                actual_bins - r_start_bin, actual_bins - r_start_bin,
                linewidth=2.5, edgecolor="gold", facecolor="none", linestyle="--",
                label="Response<->Response"
            )
            ax.add_patch(rect)
            ax.axvline(r_start_bin - 0.5, color="white", linewidth=1.2, linestyle=":")
            ax.axhline(r_start_bin - 0.5, color="white", linewidth=1.2, linestyle=":")
            ax.legend(loc="upper left", fontsize=8, framealpha=0.7)

        tick_step = max(1, actual_bins // 16)
        ticks = list(range(0, actual_bins, tick_step))
        ax.set_xticks(ticks)
        ax.set_xticklabels([labels[t] for t in ticks], rotation=45, ha="right", fontsize=7)
        ax.set_yticks(ticks)
        ax.set_yticklabels([labels[t] for t in ticks], fontsize=7)
        ax.set_xlabel("Key token bin", fontsize=11)
        ax.set_ylabel("Query token bin", fontsize=11)
        ax.set_title(f"{role.capitalize()}  |  Step {step}", fontsize=13, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Attn weight")

    fig.suptitle(f"Student vs Teacher Grouped Attention Density  |  Step {step}",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()

    out_path = out_dir / f"step_{step:06d}_student_vs_teacher.png"
    plt.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [student_vs_teacher] {out_path.name}")


# ---------------------------------------------------------------------------
# Plot 6 — per-head grid for a given step/role
# ---------------------------------------------------------------------------

def plot_per_head_grid(
    arrays: dict,
    step: int,
    role: str,
    prompt_len: int,
    out_dir: Path,
    dpi: int,
    max_heads: int = 16,
) -> None:
    layer_keys = [k for k in arrays if k.startswith("layer_")]
    for lk in layer_keys:
        a = arrays[lk]  # (n_heads, seq, seq)
        n_heads = a.shape[0]
        n_show = min(n_heads, max_heads)
        ncols = 4
        nrows = (n_show + ncols - 1) // ncols

        seq_len = a.shape[-1]
        if prompt_len < seq_len:
            a_show = a[:, prompt_len:, :]  # response queries only
        else:
            a_show = a

        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
        axes_flat = np.array(axes).flatten()
        for h in range(n_show):
            ax = axes_flat[h]
            im = ax.imshow(a_show[h], cmap=CMAPS[role], aspect="auto", interpolation="nearest")
            ax.set_title(f"Head {h}", fontsize=9)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for h in range(n_show, len(axes_flat)):
            axes_flat[h].axis("off")

        fig.suptitle(
            f"Per-Head Attention (response rows)  |  {role.capitalize()}  |  Step {step}  |  {lk}",
            fontsize=11, fontweight="bold",
        )
        plt.tight_layout()
        safe_lk = lk.replace(".", "_")
        out_path = out_dir / f"step_{step:06d}_{role}_{safe_lk}_per_head.png"
        plt.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  [per_head] {out_path.name}")


# ---------------------------------------------------------------------------
# Infer epoch boundaries from step numbers
# ---------------------------------------------------------------------------

def infer_epoch_boundaries(all_steps: list, total_epochs: int) -> list:
    """
    Naively assume steps are evenly distributed across epochs.
    Returns a list of starting step numbers for each epoch.
    """
    if not all_steps or total_epochs <= 0:
        return []
    n = len(all_steps)
    per_epoch = n / total_epochs
    boundaries = []
    for e in range(total_epochs):
        idx = min(int(e * per_epoch), n - 1)
        boundaries.append(all_steps[idx])
    return boundaries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Initialise GPU/CPU compute backend
    gpu_active = _init_gpu(args.gpu)

    attn_dir = Path(args.attn_dir)
    out_dir = Path(args.out_dir)

    if not attn_dir.exists():
        print(f"ERROR: Attention map directory does not exist: {attn_dir}", file=sys.stderr)
        print("  Make sure the training run has completed and maps are at the correct path.", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  SDPO Attention Map Visualiser")
    print(f"{'='*60}")
    print(f"  attn_dir   : {attn_dir}")
    print(f"  out_dir    : {out_dir}")
    print(f"  n_bins     : {args.n_bins}")
    print(f"  dpi        : {args.dpi}")
    print(f"  epochs     : {args.total_epochs}")
    print(f"  compute    : {'GPU (CuPy)' if gpu_active else 'CPU (numpy)'}")
    print(f"{'='*60}\n")

    # ── Discover files ──────────────────────────────────────────────────────
    step_files = discover_files(str(attn_dir))
    if not step_files:
        print(f"ERROR: No step_*.npz files found in {attn_dir}", file=sys.stderr)
        sys.exit(1)

    all_steps = sorted(step_files.keys())
    print(f"Found {len(all_steps)} steps with attention maps.")
    print(f"  Steps: {all_steps[0]} ... {all_steps[-1]}")
    for role in ROLES:
        count = sum(1 for s in all_steps if role in step_files[s])
        print(f"  {role.capitalize()}: {count} files")

    # Limit steps for trend plots
    plot_steps = all_steps[:args.max_steps_plot]

    # ── Create output subdirectories ─────────────────────────────────────────
    grouped_dir  = out_dir / "grouped_density"
    out_tok_dir  = out_dir / "output_token_attn"
    trend_dir    = out_dir / "trends"
    epoch_ev_dir = out_dir / "epoch_evolution"
    svt_dir      = out_dir / "student_vs_teacher"
    per_head_dir = out_dir / "per_head"

    for d in [grouped_dir, out_tok_dir, trend_dir, epoch_ev_dir, svt_dir, per_head_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Compute epoch boundaries ─────────────────────────────────────────────
    epoch_boundaries = infer_epoch_boundaries(all_steps, args.total_epochs)
    print(f"\nInferred epoch boundaries at steps: {epoch_boundaries}")

    # ── Accumulate metrics across steps ─────────────────────────────────────
    metrics = {r: {} for r in ROLES}

    print(f"\nProcessing {len(plot_steps)} steps for metrics...")
    for si, step in enumerate(plot_steps):
        if si % 20 == 0 or si == len(plot_steps) - 1:
            print(f"  metrics: step {si+1}/{len(plot_steps)}  (step id={step})", flush=True)
        for role in ROLES:
            if role not in step_files[step]:
                continue
            arrays = load_npz(step_files[step][role])
            attn_mean = get_attn_mean(arrays)
            prompt_len = args.prompt_len or guess_prompt_len(attn_mean)

            metrics[role][step] = {
                "entropy":        attention_entropy(attn_mean),
                "sparsity":       attention_sparsity(attn_mean),
                "mean_resp_attn": mean_response_attention(attn_mean, prompt_len),
            }
    print("  metrics: done.", flush=True)

    # ── Plot 1 & 2: Per-step density plots (cap at 50 steps for performance) ──
    # Sample evenly if more than 50 steps
    n_dense = min(50, len(plot_steps))
    indices = [int(i * (len(plot_steps) - 1) / max(1, n_dense - 1)) for i in range(n_dense)]
    dense_plot_steps = [plot_steps[i] for i in sorted(set(indices))]

    print(f"\n[1/6] Grouped density heatmaps ({len(dense_plot_steps)} steps)...")
    for step in dense_plot_steps:
        for role in ROLES:
            if role not in step_files[step]:
                continue
            arrays = load_npz(step_files[step][role])
            attn_mean = get_attn_mean(arrays)
            prompt_len = args.prompt_len or guess_prompt_len(attn_mean)
            plot_grouped_density(attn_mean, step, role, prompt_len, args.n_bins, grouped_dir, args.dpi)

    print(f"\n[2/6] Output-token attention plots ({len(dense_plot_steps)} steps)...")
    for step in dense_plot_steps:
        for role in ROLES:
            if role not in step_files[step]:
                continue
            arrays = load_npz(step_files[step][role])
            attn_mean = get_attn_mean(arrays)
            prompt_len = args.prompt_len or guess_prompt_len(attn_mean)
            plot_output_token_attn(attn_mean, step, role, prompt_len, args.n_bins, out_tok_dir, args.dpi)

    # ── Plot 3: Trend lines ─────────────────────────────────────────────────
    print(f"\n[3/6] Trend line plots...")
    plot_trends(metrics, trend_dir, args.dpi, epoch_boundaries)

    # ── Plot 4: Epoch evolution ─────────────────────────────────────────────
    print(f"\n[4/6] Epoch evolution panels...")
    for role in ROLES:
        plot_epoch_evolution(
            step_files, epoch_boundaries, args.total_epochs, role,
            args.n_bins, args.prompt_len, epoch_ev_dir, args.dpi,
        )

    # ── Plot 5: Student vs Teacher comparison ────────────────────────────────
    print(f"\n[5/6] Student vs Teacher comparison ({len(dense_plot_steps)} steps)...")
    for step in dense_plot_steps:
        if "student" not in step_files[step] or "teacher" not in step_files[step]:
            continue
        s_arrays = load_npz(step_files[step]["student"])
        t_arrays = load_npz(step_files[step]["teacher"])
        s_attn = get_attn_mean(s_arrays)
        t_attn = get_attn_mean(t_arrays)
        prompt_len = args.prompt_len or guess_prompt_len(s_attn)
        plot_student_vs_teacher(s_attn, t_attn, step, prompt_len, args.n_bins, svt_dir, args.dpi)

    # ── Plot 6: Per-head grids (first and last step only) ────────────────────
    print(f"\n[6/6] Per-head grids (first and last step)...")
    for step in ([all_steps[0]] + ([all_steps[-1]] if len(all_steps) > 1 else [])):
        for role in ROLES:
            if role not in step_files[step]:
                continue
            arrays = load_npz(step_files[step][role])
            attn_mean = get_attn_mean(arrays)
            prompt_len = args.prompt_len or guess_prompt_len(attn_mean)
            plot_per_head_grid(arrays, step, role, prompt_len, per_head_dir, args.dpi)

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  All plots saved to: {out_dir.resolve()}")
    print(f"  Subdirectories:")
    for d in [grouped_dir, out_tok_dir, trend_dir, epoch_ev_dir, svt_dir, per_head_dir]:
        count = len(list(d.glob("*.png")))
        print(f"    {d.name:25s}  {count:4d} PNGs")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

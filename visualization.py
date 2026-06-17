"""
utils/visualization.py
───────────────────────
All figure-generation code for the paper.

Figures produced
────────────────
Figure 1 : ROC curves — LoRA vs Textual Inversion (single call or overlay)
Figure 2 : Score distributions — members vs non-members
Figure 3 : Reconstruction grid — original | reconstructed | diff
Figure 4 : Rank ablation — privacy metrics vs LoRA rank
Figure 5 : Adapter comparison bar chart
Figure 6 : Privacy leakage heatmap (adapter × subject)
Figure 7 : Cumulative leakage curve (paper Figure 3)

All functions return matplotlib Figure objects and optionally save to disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe on servers and Colab

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score


# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "font.family":      "sans-serif",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "legend.fontsize":  10,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

PALETTE = {
    "lora_r4":   "#2196F3",
    "lora_r8":   "#42A5F5",
    "lora_r16":  "#1565C0",
    "lora_r32":  "#0D47A1",
    "lora_r64":  "#01579B",
    "ti":        "#E53935",
    "baseline":  "#9E9E9E",
}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, path: Optional[str], formats: Tuple[str, ...] = ("png", "pdf")) -> None:
    if path is None:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fig.savefig(p.with_suffix(f".{ext}"), bbox_inches="tight")
    print(f"  [viz] saved → {p.with_suffix('')}.{{png,pdf}}")


def _to01(t: torch.Tensor) -> np.ndarray:
    """Convert tensor in [-1,1] or [0,1] to [0,1] numpy HWC."""
    t = t.detach().float().cpu()
    if t.min() < -0.1:
        t = (t + 1) / 2
    return t.clamp(0, 1).permute(1, 2, 0).numpy()


# ──────────────────────────────────────────────────────────────────────────────
# Figure 1  –  ROC curve(s)
# ──────────────────────────────────────────────────────────────────────────────

def plot_roc(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    # {label: (member_scores, nonmember_scores)}
    higher_is_member: bool = True,
    title: str = "Membership Inference — ROC",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (6, 5),
) -> plt.Figure:
    """
    Plot one or more ROC curves on the same axes.

    Parameters
    ----------
    results : dict mapping series label → (member_scores, nonmember_scores)
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = list(PALETTE.values())

    for i, (label, (mscore, nmscore)) in enumerate(results.items()):
        scores = np.concatenate([mscore, nmscore])
        labels = np.concatenate([np.ones(len(mscore)), np.zeros(len(nmscore))])
        if not higher_is_member:
            scores = -scores

        fpr, tpr, _ = roc_curve(labels, scores)
        auc = roc_auc_score(labels, scores)
        auc_pr = average_precision_score(labels, scores)

        c = colors[i % len(colors)]
        ax.plot(fpr, tpr, lw=2, color=c,
                label=f"{label}  AUC={auc:.3f}  AP={auc_pr:.3f}")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.500)")
    ax.fill_between([0, 1], [0, 1], alpha=0.04, color="grey")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Figure 2  –  Score distributions
# ──────────────────────────────────────────────────────────────────────────────

def plot_score_distributions(
    member_scores:    np.ndarray,
    nonmember_scores: np.ndarray,
    xlabel: str = "Likelihood-ratio score",
    title:  str = "MIA Score Distributions",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (7, 4),
) -> plt.Figure:
    """Histogram of member vs non-member scores."""
    fig, ax = plt.subplots(figsize=figsize)

    all_scores = np.concatenate([member_scores, nonmember_scores])
    bins = np.linspace(all_scores.min(), all_scores.max(), 50)

    ax.hist(member_scores,    bins=bins, alpha=0.65, color="#1565C0",
            label=f"Members     (n={len(member_scores)})", density=True)
    ax.hist(nonmember_scores, bins=bins, alpha=0.65, color="#E53935",
            label=f"Non-members (n={len(nonmember_scores)})", density=True)

    ax.axvline(member_scores.mean(),    color="#1565C0", ls="--", lw=1.5,
               label=f"Member μ = {member_scores.mean():.3f}")
    ax.axvline(nonmember_scores.mean(), color="#E53935", ls="--", lw=1.5,
               label=f"Non-member μ = {nonmember_scores.mean():.3f}")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Figure 3  –  Reconstruction grid
# ──────────────────────────────────────────────────────────────────────────────

def plot_reconstruction_grid(
    originals:       List[torch.Tensor],  # [3, H, W] each
    reconstructions: List[torch.Tensor],
    metrics:         List[Dict[str, float]],
    n_show:          int = 6,
    title:           str = "Reconstruction Attack",
    save_path:       Optional[str] = None,
) -> plt.Figure:
    """
    Grid layout:  row 0 = originals,  row 1 = reconstructions,  row 2 = |diff|.
    """
    n = min(n_show, len(originals))
    fig, axes = plt.subplots(3, n, figsize=(n * 2.2, 7))

    if n == 1:
        axes = axes.reshape(3, 1)

    for j in range(n):
        orig  = _to01(originals[j])
        recon = _to01(reconstructions[j])
        diff  = np.abs(orig - recon)

        # Row 0: original
        axes[0, j].imshow(orig)
        axes[0, j].axis("off")
        if j == 0:
            axes[0, j].set_ylabel("Original", rotation=90, labelpad=4)

        # Row 1: reconstruction
        axes[1, j].imshow(recon)
        axes[1, j].axis("off")
        m = metrics[j]
        axes[1, j].set_title(
            f"LPIPS={m.get('lpips', 0):.3f}\nPSNR={m.get('psnr', 0):.1f}",
            fontsize=8,
        )
        if j == 0:
            axes[1, j].set_ylabel("Reconstructed", rotation=90, labelpad=4)

        # Row 2: pixel difference
        axes[2, j].imshow(diff, cmap="hot", vmin=0, vmax=0.5)
        axes[2, j].axis("off")
        if j == 0:
            axes[2, j].set_ylabel("|Difference|", rotation=90, labelpad=4)

    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Figure 4  –  Rank ablation
# ──────────────────────────────────────────────────────────────────────────────

def plot_rank_ablation(
    ranks:          List[int],
    metrics_series: Dict[str, List[float]],
    # {"AUC-ROC": [...], "LPIPS ↓": [...], "1-SSIM": [...]}
    title:    str = "Privacy Leakage vs LoRA Rank",
    save_path: Optional[str] = None,
    figsize:  Tuple[int, int] = (7, 5),
) -> plt.Figure:
    """
    Multi-metric line chart vs. LoRA rank (log-x axis).
    Annotates the Spearman ρ for the primary metric.
    """
    from scipy.stats import spearmanr

    fig, ax = plt.subplots(figsize=figsize)
    log_ranks = np.log2(ranks)
    colors = ["#1565C0", "#E53935", "#2E7D32", "#6A1B9A", "#F57C00"]

    for i, (label, vals) in enumerate(metrics_series.items()):
        rho, _ = spearmanr(ranks, vals)
        ax.plot(log_ranks, vals, marker="o", lw=2, color=colors[i % len(colors)],
                label=f"{label}  (ρ={rho:+.2f})")

    ax.set_xticks(log_ranks)
    ax.set_xticklabels([str(r) for r in ranks])
    ax.set_xlabel("LoRA Rank (r)")
    ax.set_ylabel("Metric Value")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Figure 5  –  Adapter comparison bar chart
# ──────────────────────────────────────────────────────────────────────────────

def plot_adapter_comparison(
    adapter_labels: List[str],
    metrics_series: Dict[str, List[float]],
    title:    str = "LoRA vs Textual Inversion — Privacy Metrics",
    save_path: Optional[str] = None,
    figsize:  Tuple[int, int] = (9, 5),
    errors:   Optional[Dict[str, List[float]]] = None,
) -> plt.Figure:
    """
    Grouped bar chart comparing adapter types across multiple metrics.

    Parameters
    ----------
    adapter_labels : e.g. ["LoRA r=4", "LoRA r=16", "LoRA r=64", "TI"]
    metrics_series : {"AUC-ROC": [0.67, 0.72, 0.78, 0.81], ...}
    errors         : std dev per series (same structure), optional
    """
    n_adapters = len(adapter_labels)
    n_metrics  = len(metrics_series)
    x = np.arange(n_adapters)
    width = 0.75 / n_metrics

    fig, ax = plt.subplots(figsize=figsize)
    colors = ["#1565C0", "#E53935", "#2E7D32", "#6A1B9A", "#F57C00"]

    for i, (metric, vals) in enumerate(metrics_series.items()):
        offset = width * (i - (n_metrics - 1) / 2)
        yerr   = errors[metric] if (errors and metric in errors) else None
        ax.bar(
            x + offset, vals, width,
            color=colors[i % len(colors)],
            label=metric,
            yerr=yerr,
            capsize=3,
            error_kw={"elinewidth": 1.2},
        )

    ax.set_xticks(x)
    ax.set_xticklabels(adapter_labels, rotation=20, ha="right")
    ax.set_ylabel("Metric Value")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.25, axis="y")

    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Figure 6  –  Privacy heatmap
# ──────────────────────────────────────────────────────────────────────────────

def plot_privacy_heatmap(
    matrix:        np.ndarray,   # [n_adapters, n_subjects]
    adapter_labels: List[str],
    subject_labels: List[str],
    metric_name:   str = "AUC-ROC",
    title:         str = "Privacy Leakage Heatmap",
    save_path:     Optional[str] = None,
    figsize:       Tuple[int, int] = (12, 5),
    vmin: float = 0.5,
    vmax: float = 1.0,
) -> plt.Figure:
    """
    Heatmap of privacy leakage (rows = adapters, cols = subjects).
    """
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        matrix, cmap="YlOrRd", aspect="auto", vmin=vmin, vmax=vmax
    )

    ax.set_xticks(np.arange(len(subject_labels)))
    ax.set_yticks(np.arange(len(adapter_labels)))
    ax.set_xticklabels(subject_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(adapter_labels)

    # Annotate cells
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            color = "white" if v > (vmin + vmax) / 2 + 0.1 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color=color, fontsize=7)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(metric_name, rotation=270, labelpad=14)

    ax.set_xlabel("Subject ID")
    ax.set_ylabel("Adapter")
    ax.set_title(title)

    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Figure 7  –  Cumulative leakage curve
# ──────────────────────────────────────────────────────────────────────────────

def plot_cumulative_leakage(
    series: Dict[str, np.ndarray],
    # {adapter_label: sorted_leakage_scores_across_subjects}
    title:    str = "Cumulative Privacy Leakage",
    xlabel:   str = "Fraction of Subjects",
    ylabel:   str = "AUC-ROC",
    save_path: Optional[str] = None,
    figsize:  Tuple[int, int] = (6, 5),
) -> plt.Figure:
    """
    CDF plot: for what fraction of subjects does the attack exceed AUC = x?
    """
    fig, ax = plt.subplots(figsize=figsize)
    colors = list(PALETTE.values())

    for i, (label, scores) in enumerate(series.items()):
        sorted_s = np.sort(scores)
        frac = np.linspace(0, 1, len(sorted_s))
        ax.plot(frac, sorted_s, lw=2, color=colors[i % len(colors)], label=label)

    ax.axhline(0.5, color="grey", ls="--", lw=1, label="Random baseline")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Batch save
# ──────────────────────────────────────────────────────────────────────────────

def save_all(
    figures: Dict[str, plt.Figure],
    output_dir: str,
    formats: Tuple[str, ...] = ("png", "pdf"),
) -> None:
    """Save all figures to output_dir/<name>.<ext>."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, fig in figures.items():
        for ext in formats:
            p = out / f"{name}.{ext}"
            fig.savefig(p, bbox_inches="tight", dpi=300)
            print(f"  saved {p}")
    plt.close("all")

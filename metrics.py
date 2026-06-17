"""
utils/metrics.py
────────────────
Shared evaluation metrics for all experiment pipelines.

Functions
─────────
compute_mia_metrics          : AUC-ROC, AUC-PR, TPR@FPR, advantage
compute_reconstruction_metrics: LPIPS, MSE, PSNR, SSIM
compute_privacy_score        : composite privacy leakage score ∈ [0,1]
bootstrap_ci                 : 95% confidence interval via bootstrap
wilcoxon_test                : non-parametric significance test
format_table                 : pretty-print results table
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import (
    spearmanr,
    wilcoxon as _wilcoxon,
    bootstrap as _bootstrap,
)
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    average_precision_score,
    precision_recall_curve,
)


# ──────────────────────────────────────────────────────────────────────────────
# MIA metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_mia_metrics(
    member_scores:     np.ndarray,
    nonmember_scores:  np.ndarray,
    higher_is_member:  bool = True,
) -> Dict[str, float]:
    """
    Compute standard MIA evaluation metrics.

    Parameters
    ----------
    member_scores, nonmember_scores : 1-D float arrays
    higher_is_member : True  when score is a likelihood-ratio (higher = member)
                       False when score is a loss value (lower = member → flip)

    Returns
    -------
    Dictionary with:
      auc_roc         : Area under ROC curve
      auc_pr          : Area under Precision-Recall curve
      tpr_at_fpr_001  : TPR when FPR ≤ 1%  (privacy auditing standard)
      tpr_at_fpr_005  : TPR when FPR ≤ 5%
      tpr_at_fpr_010  : TPR when FPR ≤ 10%
      max_advantage   : max(TPR - FPR)  =  Bernstein advantage
      balanced_acc    : balanced accuracy at optimal threshold
    """
    scores = np.concatenate([member_scores, nonmember_scores])
    labels = np.concatenate([np.ones(len(member_scores)),
                              np.zeros(len(nonmember_scores))])

    if not higher_is_member:
        scores = -scores   # flip so that higher always means "more likely member"

    # Guard against degenerate inputs
    if len(np.unique(labels)) < 2:
        return {k: float("nan") for k in [
            "auc_roc", "auc_pr", "tpr_at_fpr_001", "tpr_at_fpr_005",
            "tpr_at_fpr_010", "max_advantage", "balanced_acc"
        ]}

    auc_roc = float(roc_auc_score(labels, scores))
    auc_pr  = float(average_precision_score(labels, scores))

    fpr, tpr, thresholds = roc_curve(labels, scores)

    def _tpr_at(target_fpr: float) -> float:
        idx = np.searchsorted(fpr, target_fpr, side="right") - 1
        idx = int(np.clip(idx, 0, len(tpr) - 1))
        return float(tpr[idx])

    advantage = float(np.max(tpr - fpr))

    # Balanced accuracy: TPR+TNR / 2 at best threshold
    balanced = float(np.max((tpr + (1 - fpr)) / 2))

    return {
        "auc_roc":        auc_roc,
        "auc_pr":         auc_pr,
        "tpr_at_fpr_001": _tpr_at(0.01),
        "tpr_at_fpr_005": _tpr_at(0.05),
        "tpr_at_fpr_010": _tpr_at(0.10),
        "max_advantage":  advantage,
        "balanced_acc":   balanced,
        "n_members":      int(len(member_scores)),
        "n_nonmembers":   int(len(nonmember_scores)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Reconstruction metrics
# ──────────────────────────────────────────────────────────────────────────────

def _ssim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Per-image SSIM (both tensors in [0,1], shape [3,H,W])."""
    a, b = a.float(), b.float()
    c1, c2 = 0.01**2, 0.03**2
    mu1, mu2 = a.mean(), b.mean()
    s1  = ((a - mu1)**2).mean()
    s2  = ((b - mu2)**2).mean()
    s12 = ((a - mu1)*(b - mu2)).mean()
    return float(((2*mu1*mu2+c1)*(2*s12+c2)) /
                 ((mu1**2+mu2**2+c1)*(s1+s2+c2)))


def compute_reconstruction_metrics(
    reconstructed: torch.Tensor,  # [B, 3, H, W] or [3, H, W]
    ground_truth:  torch.Tensor,
    lpips_net: Optional[object] = None,   # pass pre-loaded lpips.LPIPS instance
) -> Dict[str, float]:
    """
    LPIPS ↓, MSE ↓, PSNR ↑, SSIM ↑ for a batch of image pairs.

    Images should be in [-1, 1] (VAE-normalised) or [0, 1].
    """
    if reconstructed.dim() == 3:
        reconstructed = reconstructed.unsqueeze(0)
        ground_truth  = ground_truth.unsqueeze(0)

    # Normalise to [0, 1]
    def _to01(x: torch.Tensor) -> torch.Tensor:
        return ((x + 1) / 2).clamp(0, 1) if x.min() < -0.1 else x.clamp(0, 1)

    r = _to01(reconstructed.float().cpu())
    g = _to01(ground_truth.float().cpu())

    B = r.shape[0]

    # MSE per image
    mse = F.mse_loss(r, g, reduction="none").mean(dim=[1, 2, 3]).numpy()
    psnr = 10 * np.log10(1.0 / (mse + 1e-8))

    # SSIM per image
    ssim = np.array([_ssim(r[i], g[i]) for i in range(B)])

    # LPIPS
    if lpips_net is not None:
        r_lp = (r * 2 - 1)
        g_lp = (g * 2 - 1)
        with torch.no_grad():
            lp_vals = np.array([
                lpips_net(r_lp[i:i+1], g_lp[i:i+1]).item()
                for i in range(B)
            ])
    else:
        lp_vals = np.zeros(B)

    return {
        "lpips_mean": float(lp_vals.mean()),
        "lpips_std":  float(lp_vals.std()),
        "mse_mean":   float(mse.mean()),
        "mse_std":    float(mse.std()),
        "psnr_mean":  float(psnr.mean()),
        "psnr_std":   float(psnr.std()),
        "ssim_mean":  float(ssim.mean()),
        "ssim_std":   float(ssim.std()),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Composite privacy score
# ──────────────────────────────────────────────────────────────────────────────

def compute_privacy_score(
    mia_metrics:   Dict[str, float],
    recon_metrics: Optional[Dict[str, float]] = None,
    w_auc:         float = 0.35,
    w_adv:         float = 0.25,
    w_tpr:         float = 0.20,
    w_recon:       float = 0.20,
) -> float:
    """
    Composite privacy leakage score ∈ [0, 1].

    Combines MIA AUC, advantage, TPR@1%FPR, and reconstruction quality.
    Higher score = more leakage.

    Paper uses this to produce a single ranking of adapter configurations.
    """
    # MIA contribution
    auc_contrib = (mia_metrics.get("auc_roc", 0.5) - 0.5) * 2  # map [0.5,1]→[0,1]
    auc_contrib = float(np.clip(auc_contrib, 0.0, 1.0))

    adv_contrib = float(np.clip(mia_metrics.get("max_advantage", 0.0), 0.0, 1.0))
    tpr_contrib = float(np.clip(mia_metrics.get("tpr_at_fpr_001", 0.0), 0.0, 1.0))

    score = w_auc * auc_contrib + w_adv * adv_contrib + w_tpr * tpr_contrib

    # Reconstruction contribution
    if recon_metrics is not None:
        # Lower LPIPS = better reconstruction = more leakage
        # Typical LPIPS ∈ [0, 0.7]; normalise inversely
        lpips = recon_metrics.get("lpips_mean", 0.5)
        recon_contrib = float(np.clip(1.0 - lpips / 0.7, 0.0, 1.0))
        score += w_recon * recon_contrib

    return float(np.clip(score, 0.0, 1.0))


# ──────────────────────────────────────────────────────────────────────────────
# Statistical tests
# ──────────────────────────────────────────────────────────────────────────────

def bootstrap_ci(
    data:       np.ndarray,
    statistic:  Callable[[np.ndarray], float],
    n_boot:     int = 2000,
    confidence: float = 0.95,
    seed:       int = 42,
) -> Tuple[float, float, float]:
    """
    Non-parametric bootstrap confidence interval.

    Returns (point_estimate, lower_bound, upper_bound).
    """
    rng = np.random.default_rng(seed)
    estimates = np.array([
        statistic(rng.choice(data, size=len(data), replace=True))
        for _ in range(n_boot)
    ])
    alpha = 1 - confidence
    lo = float(np.percentile(estimates, alpha / 2 * 100))
    hi = float(np.percentile(estimates, (1 - alpha / 2) * 100))
    return float(statistic(data)), lo, hi


def wilcoxon_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> Tuple[float, float]:
    """
    Wilcoxon signed-rank test for paired score arrays.
    Returns (statistic, p_value).
    """
    n = min(len(scores_a), len(scores_b))
    stat, p = _wilcoxon(scores_a[:n], scores_b[:n], alternative="two-sided")
    return float(stat), float(p)


def spearman_rank(
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[float, float]:
    """
    Spearman rank correlation.  Returns (rho, p_value).
    Used for: rank vs leakage correlation (ρ=0.83 in paper).
    """
    rho, pval = spearmanr(x, y)
    return float(rho), float(pval)


# ──────────────────────────────────────────────────────────────────────────────
# Pretty printing
# ──────────────────────────────────────────────────────────────────────────────

def format_table(
    rows: Dict[str, Dict[str, float]],
    title: str = "Results",
    float_fmt: str = ".4f",
) -> str:
    """
    Format a dict-of-dicts as a plain-text table.

    Parameters
    ----------
    rows : {row_label: {metric: value}}

    Returns
    -------
    str  (print it or write to a file)
    """
    if not rows:
        return "(empty table)"

    all_cols = []
    for d in rows.values():
        for k in d:
            if k not in all_cols:
                all_cols.append(k)

    col_w = max(len(c) for c in all_cols) + 2
    row_w = max(len(r) for r in rows) + 2

    header = f"{'Method':<{row_w}}" + "".join(f"{c:>{col_w}}" for c in all_cols)
    sep    = "-" * len(header)
    lines  = [f"\n{'='*len(header)}", f"{title:^{len(header)}}", f"{'='*len(header)}",
               header, sep]

    for row_label, metrics in rows.items():
        row = f"{row_label:<{row_w}}"
        for col in all_cols:
            v = metrics.get(col)
            if v is None:
                row += f"{'—':>{col_w}}"
            elif isinstance(v, float):
                row += f"{v:>{col_w}{float_fmt}}"
            else:
                row += f"{str(v):>{col_w}}"
        lines.append(row)

    lines += [f"{'='*len(header)}", ""]
    return "\n".join(lines)

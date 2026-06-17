"""
paper_results_analysis.py
──────────────────────────
Post-hoc analysis script.

Given saved JSON results from the experiment pipelines, this script
  • Regenerates every paper table (Table 1, 2, 3)
  • Regenerates every paper figure (Figures 1–7)
  • Computes Spearman ρ between rank and leakage
  • Prints a concise final summary

Use this when you have already run the attacks and just need to
re-generate figures (e.g. after changing colour scheme or layout).

Usage
─────
# After running run_all_experiments.sh:
python paper_results_analysis.py --results_dir ./results

# Point at specific JSON files:
python paper_results_analysis.py \\
    --mia_json     results/mia_lora_rank16/mia_multi_lora.json \\
    --recon_json   results/recon_lora_rank16/recon_multi_lora.json \\
    --ablation_json results/rank_ablation/rank_ablation_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from utils.metrics import compute_mia_metrics, compute_privacy_score, format_table
from utils.visualization import (
    plot_roc, plot_score_distributions,
    plot_rank_ablation, plot_adapter_comparison,
    plot_privacy_heatmap, plot_cumulative_leakage,
    save_all,
)


# ──────────────────────────────────────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────────────────────────────────────

def _load(path: Optional[str]) -> Optional[Dict]:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        print(f"  ⚠  not found: {path}")
        return None
    return json.loads(p.read_text())


def _glob_newest(directory: Path, pattern: str) -> Optional[Path]:
    hits = sorted(directory.glob(pattern))
    return hits[0] if hits else None


# ──────────────────────────────────────────────────────────────────────────────
# Auto-discovery from a results_dir
# ──────────────────────────────────────────────────────────────────────────────

def _discover(results_dir: Path) -> Dict[str, Optional[Path]]:
    """Discover JSON result files under results_dir."""
    disc: Dict[str, Optional[Path]] = {
        "mia_lora":       None,
        "mia_ti":         None,
        "recon_lora":     None,
        "recon_ti":       None,
        "ablation":       None,
        "stat":           None,
        "comparison":     None,
    }

    # MIA
    for rank in [16, 64, 32, 8, 4]:
        p = results_dir / f"mia_lora_rank{rank}" / f"mia_multi_lora.json"
        if p.exists():
            disc["mia_lora"] = p
            break
    disc["mia_ti"] = _glob_newest(results_dir, "mia_ti/mia_multi_textual_inversion.json")

    # Reconstruction
    for rank in [16, 64, 32, 8, 4]:
        p = results_dir / f"recon_lora_rank{rank}" / f"recon_multi_lora.json"
        if p.exists():
            disc["recon_lora"] = p
            break
    disc["recon_ti"] = _glob_newest(results_dir, "recon_ti/recon_multi_textual_inversion.json")

    # Ablation
    disc["ablation"] = _glob_newest(
        results_dir / "rank_ablation", "rank_ablation_summary.json"
    )

    # Statistical
    disc["stat"] = _glob_newest(results_dir / "statistical", "stat_summary.json")

    # Comparison
    disc["comparison"] = _glob_newest(results_dir / "comparison", "comparison_summary.json")

    print("\n[Discovery]")
    for k, v in disc.items():
        status = f"✓ {v.relative_to(results_dir)}" if v else "⚠  not found"
        print(f"  {k:<18}  {status}")

    return disc


# ──────────────────────────────────────────────────────────────────────────────
# Table 1 — MIA + Reconstruction per adapter
# ──────────────────────────────────────────────────────────────────────────────

def build_table1(
    mia_files:   Dict[str, Optional[Path]],
    recon_files: Dict[str, Optional[Path]],
) -> str:
    """
    mia_files  : {label: path_to_mia_multi_*.json}
    recon_files: {label: path_to_recon_multi_*.json}
    """
    rows: Dict[str, Dict] = {}

    for label in sorted(set(list(mia_files.keys()) + list(recon_files.keys()))):
        mia_data   = _load(str(mia_files.get(label)))
        recon_data = _load(str(recon_files.get(label)))

        mia_agg   = (mia_data   or {}).get("aggregate", {})
        recon_agg = (recon_data or {}).get("aggregate", {})

        rows[label] = {
            "AUC-ROC":     mia_agg.get("auc_roc",         float("nan")),
            "TPR@1%FPR":   mia_agg.get("tpr_at_fpr_001",  float("nan")),
            "Advantage":   mia_agg.get("max_advantage",   float("nan")),
            "LPIPS ↓":     recon_agg.get("lpips_mean",    float("nan")),
            "PSNR ↑ (dB)": recon_agg.get("psnr_mean",    float("nan")),
            "SSIM ↑":      recon_agg.get("ssim_mean",     float("nan")),
        }
        if mia_agg and recon_agg:
            rows[label]["Privacy Score"] = compute_privacy_score(
                mia_agg, {"lpips_mean": recon_agg.get("lpips_mean", 0.5)}
            )

    return format_table(rows, title="Table 1: Privacy Leakage — All Adapters")


# ──────────────────────────────────────────────────────────────────────────────
# Table 2 — Rank ablation
# ──────────────────────────────────────────────────────────────────────────────

def build_table2(ablation_data: Dict) -> str:
    rank_results = ablation_data.get("rank_results", {})
    rows: Dict[str, Dict] = {}

    for rank_str, res in sorted(rank_results.items(), key=lambda x: int(x[0])):
        r = int(rank_str)
        rows[f"LoRA r={r}"] = {
            "AUC-ROC":    res.get("auc_roc",    float("nan")),
            "TPR@1%FPR":  res.get("tpr_001",    float("nan")),
            "Advantage":  res.get("advantage",  float("nan")),
            "LPIPS ↓":    res.get("lpips",      float("nan")),
            "PSNR ↑":     res.get("psnr",       float("nan")),
            "SSIM ↑":     res.get("ssim",       float("nan")),
        }

    sp_lpips = ablation_data.get("spearman_lpips", {})
    sp_auc   = ablation_data.get("spearman_auc",   {})

    header = format_table(rows, title="Table 2: Rank Ablation")

    footer = (
        f"\n  Spearman ρ (rank vs LPIPS):    {sp_lpips.get('spearman_rho', float('nan')):+.3f}  "
        f"p={sp_lpips.get('p_value', float('nan')):.4f}\n"
        f"  Spearman ρ (rank vs AUC-ROC):  {sp_auc.get('spearman_rho', float('nan')):+.3f}  "
        f"p={sp_auc.get('p_value', float('nan')):.4f}\n"
    )
    return header + footer


# ──────────────────────────────────────────────────────────────────────────────
# Table 3 — Statistical leakage
# ──────────────────────────────────────────────────────────────────────────────

def build_table3(stat_data: Dict) -> str:
    rows: Dict[str, Dict] = {}
    for label, vals in stat_data.items():
        if label.startswith("_"):
            continue
        if not isinstance(vals, dict):
            continue
        rows[label] = {
            "Weight Norm":   vals.get("weight_norm_mean",  float("nan")),
            "SV Entropy":    vals.get("sv_entropy_mean",   float("nan")),
            "Stable Rank":   vals.get("stable_rank_mean",  float("nan")),
            "Fisher (mean)": vals.get("fisher_mean",       float("nan")),
        }

    sp = stat_data.get("_spearman_rank_vs_norm", {})
    header = format_table(rows, title="Table 3: Statistical Leakage Analysis")
    if sp:
        header += (
            f"\n  Spearman ρ (rank vs weight norm):  "
            f"{sp.get('spearman_rho', float('nan')):+.3f}  "
            f"p={sp.get('p_value', float('nan')):.4f}\n"
        )
    return header


# ──────────────────────────────────────────────────────────────────────────────
# Figure generation
# ──────────────────────────────────────────────────────────────────────────────

def build_figures(
    mia_multi_by_label: Dict[str, Optional[Dict]],
    recon_multi_by_label: Dict[str, Optional[Dict]],
    ablation_data: Optional[Dict],
    stat_data: Optional[Dict],
    output_dir: Path,
) -> None:
    figs = {}

    # ── Figure 1: ROC overlay ─────────────────────────────────────────────────
    roc_series = {}
    for label, data in mia_multi_by_label.items():
        if data is None:
            continue
        all_m  = np.concatenate([p["member_scores"]    for p in data.get("per_subject", [])])
        all_nm = np.concatenate([p["nonmember_scores"] for p in data.get("per_subject", [])])
        if len(all_m) and len(all_nm):
            roc_series[label] = (all_m, all_nm)

    if roc_series:
        figs["figure1_mia_roc"] = plot_roc(
            roc_series,
            higher_is_member=True,
            title="Figure 1: MIA ROC Curves — LoRA vs Textual Inversion",
        )

    # ── Figure 2: Score distributions (best MIA run) ─────────────────────────
    for label, data in mia_multi_by_label.items():
        if data is None:
            continue
        all_m  = np.concatenate([p["member_scores"]    for p in data.get("per_subject", [])])
        all_nm = np.concatenate([p["nonmember_scores"] for p in data.get("per_subject", [])])
        if len(all_m) and len(all_nm):
            figs["figure2_score_dist"] = plot_score_distributions(
                all_m, all_nm,
                title=f"Figure 2: Score Distributions ({label})",
                xlabel="Likelihood-ratio score",
            )
            break  # one example is enough

    # ── Figure 4: Rank ablation ───────────────────────────────────────────────
    if ablation_data:
        rank_results = ablation_data.get("rank_results", {})
        valid = sorted(
            [(int(k), v) for k, v in rank_results.items()
             if isinstance(v, dict)],
            key=lambda x: x[0]
        )
        if len(valid) >= 2:
            ranks = [r for r, _ in valid]
            figs["figure4_rank_ablation"] = plot_rank_ablation(
                ranks=ranks,
                metrics_series={
                    "AUC-ROC":   [v.get("auc_roc", np.nan) for _, v in valid],
                    "LPIPS ↓":   [v.get("lpips",   np.nan) for _, v in valid],
                    "TPR@1%FPR": [v.get("tpr_001", np.nan) for _, v in valid],
                },
                title="Figure 4: Privacy Leakage vs LoRA Rank",
            )

    # ── Figure 5: Adapter comparison bar chart ────────────────────────────────
    bar_labels: List[str] = []
    bar_auc:    List[float] = []
    bar_lpips:  List[float] = []
    bar_tpr:    List[float] = []

    for label in sorted(mia_multi_by_label.keys()):
        mia_data   = mia_multi_by_label.get(label)
        recon_data = recon_multi_by_label.get(label)
        if mia_data is None or recon_data is None:
            continue
        m_agg = mia_data.get("aggregate", {})
        r_agg = recon_data.get("aggregate", {})
        bar_labels.append(label)
        bar_auc.append(m_agg.get("auc_roc", np.nan))
        bar_lpips.append(r_agg.get("lpips_mean", np.nan))
        bar_tpr.append(m_agg.get("tpr_at_fpr_001", np.nan))

    if bar_labels:
        figs["figure5_bar_comparison"] = plot_adapter_comparison(
            adapter_labels=bar_labels,
            metrics_series={
                "AUC-ROC":   bar_auc,
                "LPIPS ↓":   bar_lpips,
                "TPR@1%FPR": bar_tpr,
            },
            title="Figure 5: Privacy Leakage — All Adapters",
        )

    # ── Figure 7: Cumulative leakage CDF ─────────────────────────────────────
    cdf_series = {}
    for label, data in mia_multi_by_label.items():
        if data is None:
            continue
        per_auc = [p["metrics"]["auc_roc"] for p in data.get("per_subject", [])
                   if "metrics" in p]
        if per_auc:
            cdf_series[label] = np.array(per_auc)

    if cdf_series:
        figs["figure7_cumulative_leakage"] = plot_cumulative_leakage(
            cdf_series,
            title="Figure 7: Cumulative MIA Leakage (CDF over subjects)",
            ylabel="AUC-ROC",
        )

    # ── Save ──────────────────────────────────────────────────────────────────
    save_all(figs, output_dir)
    print(f"\n  Saved {len(figs)} figures to {output_dir}")
    for name in figs:
        print(f"    {name}.pdf / .png")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Discover result files ─────────────────────────────────────────────────
    results_dir = Path(args.results_dir)
    disc = _discover(results_dir)

    # ── Build per-label MIA and recon dicts ───────────────────────────────────
    # Collect all mia_lora_rank* and mia_ti directories
    mia_by_label:   Dict[str, Optional[Dict]] = {}
    recon_by_label: Dict[str, Optional[Dict]] = {}

    for rank in [4, 8, 16, 32, 64]:
        mia_p   = results_dir / f"mia_lora_rank{rank}"   / "mia_multi_lora.json"
        recon_p = results_dir / f"recon_lora_rank{rank}" / "recon_multi_lora.json"
        label   = f"LoRA r={rank}"
        mia_by_label[label]   = _load(str(mia_p))   if mia_p.exists()   else None
        recon_by_label[label] = _load(str(recon_p)) if recon_p.exists() else None

    ti_mia_p   = results_dir / "mia_ti"   / "mia_multi_textual_inversion.json"
    ti_recon_p = results_dir / "recon_ti" / "recon_multi_textual_inversion.json"
    mia_by_label["Textual Inversion"]   = _load(str(ti_mia_p))   if ti_mia_p.exists()   else None
    recon_by_label["Textual Inversion"] = _load(str(ti_recon_p)) if ti_recon_p.exists() else None

    # Allow manual override via CLI
    if args.mia_json:
        mia_by_label["LoRA r=16 (manual)"] = _load(args.mia_json)
    if args.recon_json:
        recon_by_label["LoRA r=16 (manual)"] = _load(args.recon_json)

    ablation_data = _load(args.ablation_json or str(disc.get("ablation") or ""))
    stat_data     = _load(args.stat_json     or str(disc.get("stat")     or ""))

    # ── Tables ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    table1 = build_table1(
        {l: Path(str(v)) if v else None for l, v in mia_by_label.items()},
        {l: Path(str(v)) if v else None for l, v in recon_by_label.items()},
    )
    # Rebuild with actual loaded dicts
    rows_t1: Dict[str, Dict] = {}
    for label in sorted(set(list(mia_by_label.keys()) + list(recon_by_label.keys()))):
        mia_agg   = (mia_by_label.get(label)   or {}).get("aggregate", {})
        recon_agg = (recon_by_label.get(label) or {}).get("aggregate", {})
        if not mia_agg and not recon_agg:
            continue
        rows_t1[label] = {
            "AUC-ROC":     mia_agg.get("auc_roc",        float("nan")),
            "TPR@1%FPR":   mia_agg.get("tpr_at_fpr_001", float("nan")),
            "LPIPS ↓":     recon_agg.get("lpips_mean",   float("nan")),
            "PSNR ↑":      recon_agg.get("psnr_mean",    float("nan")),
        }
        if mia_agg and recon_agg:
            rows_t1[label]["Privacy Score"] = compute_privacy_score(
                mia_agg, {"lpips_mean": recon_agg.get("lpips_mean", 0.5)}
            )
    table1_str = format_table(rows_t1, title="Table 1: Privacy Leakage — All Adapters")
    print(table1_str)
    (out / "table1.txt").write_text(table1_str)

    if ablation_data:
        t2 = build_table2(ablation_data)
        print(t2)
        (out / "table2.txt").write_text(t2)

    if stat_data:
        t3 = build_table3(stat_data)
        print(t3)
        (out / "table3.txt").write_text(t3)

    # ── Figures ───────────────────────────────────────────────────────────────
    build_figures(
        mia_multi_by_label   = mia_by_label,
        recon_multi_by_label = recon_by_label,
        ablation_data        = ablation_data,
        stat_data            = stat_data,
        output_dir           = out / "figures",
    )

    print(f"\n✓ Analysis complete.  All outputs in {out}")


# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Regenerate paper tables and figures from cached results")
    p.add_argument("--results_dir",   type=str, default="./results",
                   help="Root results directory (auto-discovers JSON files)")
    p.add_argument("--output_dir",    type=str, default="./results/paper_outputs",
                   help="Where to write tables (.txt) and figures (.pdf/.png)")
    p.add_argument("--mia_json",      type=str, default=None,
                   help="Manual override: path to a mia_multi_*.json")
    p.add_argument("--recon_json",    type=str, default=None,
                   help="Manual override: path to a recon_multi_*.json")
    p.add_argument("--ablation_json", type=str, default=None,
                   help="Path to rank_ablation_summary.json")
    p.add_argument("--stat_json",     type=str, default=None,
                   help="Path to stat_summary.json")
    return p.parse_args()


if __name__ == "__main__":
    main()

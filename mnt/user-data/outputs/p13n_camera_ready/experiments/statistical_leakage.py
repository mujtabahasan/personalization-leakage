"""
experiments/statistical_leakage.py
────────────────────────────────────
White-box statistical leakage analysis pipeline.

Reproduces:
  • Paper Table 3  — Fisher information, weight norms, spectral stats
  • Paper Figure 6 — Privacy heatmap across adapter × subject
  • Paper finding  — Spearman ρ = 0.83 between LoRA rank and LPIPS leakage

Requires access to the raw adapter weight files (white-box setting).

Usage examples
──────────────
# Analyse one LoRA adapter
python experiments/statistical_leakage.py \\
    --mode single \\
    --data_dir ./data \\
    --adapter_path ./adapters/lora_rank16_subject0.safetensors \\
    --subject_id 0

# Full analysis across all adapters (Table 3)
python experiments/statistical_leakage.py \\
    --mode full \\
    --data_dir ./data \\
    --lora_dir ./adapters \\
    --ti_dir   ./adapters_ti \\
    --ranks 4 8 16 32 64 \\
    --num_subjects 30 \\
    --output_dir ./results/statistical
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.diffusion_wrapper import DiffusionWrapper
from attacks.statistical_leakage import (
    fisher_information_proxy,
    weight_norm_stats,
    spectral_analysis,
    gradient_alignment,
    rank_leakage_spearman,
    full_statistical_analysis,
)
from utils.data_utils import SubjectLoader, find_adapter_path, load_adapter_state
from utils.metrics import format_table
from utils.visualization import plot_privacy_heatmap, save_all


# ──────────────────────────────────────────────────────────────────────────────
# Mode: single adapter
# ──────────────────────────────────────────────────────────────────────────────

def run_single(args: argparse.Namespace) -> None:
    loader = SubjectLoader(args.data_dir, args.subject_id, max_images=10)
    m_imgs, m_pr, nm_imgs, nm_pr = loader.all_splits()

    wrapper = DiffusionWrapper(
        model_id=args.model_id, device=args.device,
        dtype=torch.float16 if args.device == "cuda" else torch.float32,
    )

    from safetensors.torch import load_file
    lora_state = load_file(args.adapter_path)

    print(f"\n[StatLeakage] Analysing {args.adapter_path}\n")

    results = full_statistical_analysis(
        wrapper         = wrapper,
        lora_state      = lora_state,
        member_images   = m_imgs,
        nonmember_images= nm_imgs,
        prompts         = m_pr,
        compute_fisher  = args.compute_fisher,
        compute_gradient= args.compute_gradient,
    )

    print(f"\n{'='*60}")
    print("Statistical Leakage Results")
    print(f"{'='*60}")
    for k, v in results.items():
        print(f"  {k:<30}  {v:.5f}" if isinstance(v, float) else f"  {k:<30}  {v}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"stat_single_{args.subject_id}.json").write_text(
        json.dumps(results, indent=2)
    )
    print(f"\n✓ Results in {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Mode: full analysis — all adapters, all subjects (Table 3)
# ──────────────────────────────────────────────────────────────────────────────

def run_full(args: argparse.Namespace) -> None:
    splits     = json.loads((Path(args.data_dir) / "splits.json").read_text())
    sids       = list(splits.keys())[: args.num_subjects]
    out        = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # configs: list of (label, adapter_dir, adapter_type, rank)
    configs = []
    if args.lora_dir:
        for r in args.ranks:
            configs.append((f"LoRA r={r}", Path(args.lora_dir), "lora", r))
    if args.ti_dir:
        configs.append(("TI", Path(args.ti_dir), "textual_inversion", None))

    summary_rows: Dict[str, Dict] = {}

    # For heatmap: matrix[config_idx, subject_idx] = fisher_mean (or weight_norm)
    heatmap_values = np.zeros((len(configs), len(sids)))
    heatmap_label  = "Fisher Information (mean)"

    for cfg_idx, (label, adapter_dir, adapter_type, rank) in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"  Config: {label}")
        print(f"{'='*60}\n")

        per_subject_stats: List[Dict] = []
        fisher_vals, norm_vals, sv_entropy_vals = [], [], []

        wrapper = DiffusionWrapper(
            model_id=args.model_id, device=args.device,
            dtype=torch.float16 if args.device == "cuda" else torch.float32,
        )

        for sid_idx, sid in enumerate(sids):
            # Find adapter
            adapter_path = find_adapter_path(adapter_dir, adapter_type, sid, rank)
            if adapter_path is None:
                print(f"  ⚠ no adapter for subject {sid}, skipping")
                continue

            # Load adapter state
            lora_state = load_adapter_state(adapter_path, adapter_type, wrapper)

            # For TI we cannot compute Fisher/spectral (no lora_A/B)
            if adapter_type == "textual_inversion":
                per_subject_stats.append({"subject_id": sid, "adapter_type": "ti"})
                continue

            # Load images
            try:
                loader  = SubjectLoader(args.data_dir, sid, max_images=5)
                m_imgs, m_pr, nm_imgs, nm_pr = loader.all_splits()
            except Exception as e:
                print(f"  ⚠ failed to load images for {sid}: {e}")
                continue

            # Weight norm (fast, no GPU)
            wn = weight_norm_stats(lora_state)
            norm_vals.append(wn["weight_norm_mean"])

            # Spectral (fast)
            sp = spectral_analysis(lora_state)
            sv_entropy_vals.append(sp["sv_entropy_mean"])

            # Fisher (slow — skip if not requested)
            fisher_val = float("nan")
            if args.compute_fisher:
                fi = fisher_information_proxy(
                    wrapper, lora_state, m_imgs[:3], m_pr[:3], num_timesteps=3
                )
                fisher_val = fi["fisher_mean"]
                fisher_vals.append(fisher_val)

            heatmap_values[cfg_idx, sid_idx] = wn["weight_norm_mean"]

            per_subject_stats.append({
                "subject_id":     sid,
                "adapter_type":   adapter_type,
                "rank":           rank,
                "weight_norm_mean":   wn["weight_norm_mean"],
                "sv_entropy_mean":    sp["sv_entropy_mean"],
                "stable_rank_mean":   sp["stable_rank_mean"],
                "fisher_mean":        fisher_val,
            })

            print(f"  subject {sid}:  ‖W‖={wn['weight_norm_mean']:.4f}  "
                  f"sv_ent={sp['sv_entropy_mean']:.3f}  "
                  f"fisher={fisher_val:.4e}" if not np.isnan(fisher_val) else
                  f"  subject {sid}:  ‖W‖={wn['weight_norm_mean']:.4f}  "
                  f"sv_ent={sp['sv_entropy_mean']:.3f}")

        # Aggregate
        valid = [s for s in per_subject_stats if "weight_norm_mean" in s]
        if valid:
            summary_rows[label] = {
                "weight_norm_mean":  float(np.mean([s["weight_norm_mean"]  for s in valid])),
                "sv_entropy_mean":   float(np.mean([s["sv_entropy_mean"]   for s in valid])),
                "stable_rank_mean":  float(np.mean([s["stable_rank_mean"]  for s in valid])),
                "n_subjects":        len(valid),
            }
            if fisher_vals:
                summary_rows[label]["fisher_mean"] = float(np.mean(fisher_vals))

        # Save per-config JSON
        (out / f"stat_{label.replace(' ', '_').replace('=', '')}.json").write_text(
            json.dumps({"config": label, "per_subject": per_subject_stats}, indent=2,
                       default=lambda x: float(x) if isinstance(x, (float, np.floating)) else x)
        )

        del wrapper
        torch.cuda.empty_cache()

    # ── Spearman: rank vs weight norm ─────────────────────────────────────────
    if args.lora_dir and len(args.ranks) >= 3:
        lora_labels = [f"LoRA r={r}" for r in args.ranks if f"LoRA r={r}" in summary_rows]
        if len(lora_labels) >= 3:
            wn_per_rank = [summary_rows[l]["weight_norm_mean"] for l in lora_labels]
            ranks_used  = [int(l.split("=")[1]) for l in lora_labels]
            rho = rank_leakage_spearman(ranks_used, wn_per_rank)
            print(f"\n[Spearman] rank vs weight_norm: "
                  f"ρ={rho['spearman_rho']:+.3f}  p={rho['p_value']:.4f}")
            summary_rows["_spearman_rank_vs_norm"] = rho  # type: ignore

    # ── Summary table (Table 3) ───────────────────────────────────────────────
    display_rows = {k: v for k, v in summary_rows.items() if not k.startswith("_")}
    print(format_table(display_rows, title="Table 3: Statistical Leakage Analysis"))

    # ── Save summary ──────────────────────────────────────────────────────────
    (out / "stat_summary.json").write_text(
        json.dumps(summary_rows, indent=2,
                   default=lambda x: float(x) if isinstance(x, (float, np.floating)) else x)
    )

    # ── Heatmap figure (Figure 6) ─────────────────────────────────────────────
    adapter_labels  = [c[0] for c in configs]
    subject_labels  = [f"s{s}" for s in sids]
    heatmap_label   = "Weight Norm ‖A@B‖"

    figs = {
        "figure6_privacy_heatmap": plot_privacy_heatmap(
            matrix         = heatmap_values,
            adapter_labels = adapter_labels,
            subject_labels = subject_labels,
            metric_name    = heatmap_label,
            title          = f"Statistical Leakage Heatmap — {heatmap_label}",
            vmin           = 0.0,
            vmax           = float(np.nanmax(heatmap_values)) or 1.0,
        )
    }
    save_all(figs, out)
    print(f"\n✓ Statistical analysis complete. Results in {out}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Statistical Leakage Analysis")
    p.add_argument("--mode",            choices=["single", "full"], default="full")
    p.add_argument("--data_dir",        type=str, default="./data")
    p.add_argument("--adapter_path",    type=str, default=None,
                   help="Path to single adapter (single mode)")
    p.add_argument("--subject_id",      type=str, default="0")
    p.add_argument("--lora_dir",        type=str, default=None)
    p.add_argument("--ti_dir",          type=str, default=None)
    p.add_argument("--ranks",           nargs="+", type=int, default=[4, 8, 16, 32, 64])
    p.add_argument("--num_subjects",    type=int, default=30)
    p.add_argument("--output_dir",      type=str, default="./results/statistical")
    p.add_argument("--model_id",        type=str, default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--device",          type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--compute_fisher",  action="store_true",
                   help="Compute Fisher information (slow; requires backward pass)")
    p.add_argument("--compute_gradient",action="store_true",
                   help="Compute gradient alignment (slow)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[StatLeakage Pipeline]  mode={args.mode}  device={args.device}")

    if args.mode == "single":
        if not args.adapter_path:
            raise ValueError("--adapter_path required for single mode")
        run_single(args)
    else:
        run_full(args)


if __name__ == "__main__":
    main()

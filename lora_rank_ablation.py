"""
experiments/lora_rank_ablation.py
───────────────────────────────────
LoRA rank ablation study.

Reproduces:
  • Paper Table 2  — MIA and reconstruction metrics across ranks
  • Paper Figure 4 — privacy metrics vs. LoRA rank
  • Paper finding  — Spearman ρ = 0.83 between rank and LPIPS leakage

The script expects one adapter directory per rank, e.g.:
  adapters/rank4/   lora_rank4_subject0.safetensors  …
  adapters/rank8/   lora_rank8_subject0.safetensors  …
  adapters/rank16/  …
  adapters/rank32/  …
  adapters/rank64/  …

Alternatively, pass --adapters_dir with adapters named
  lora_rank<R>_subject<ID>.safetensors  inside a single flat directory.

Usage examples
──────────────
# Full paper ablation (requires pre-trained adapters)
python experiments/lora_rank_ablation.py \\
    --data_dir ./data \\
    --adapters_root ./adapters \\
    --ranks 4 8 16 32 64 \\
    --num_subjects 30 \\
    --output_dir ./results/rank_ablation

# Quick test: single subject, two ranks
python experiments/lora_rank_ablation.py \\
    --data_dir ./data \\
    --adapters_root ./adapters \\
    --ranks 4 16 \\
    --num_subjects 1 \\
    --output_dir ./results/rank_ablation_test
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
from attacks.membership_inference import run_mia
from attacks.data_reconstruction import run_reconstruction
from attacks.statistical_leakage import rank_leakage_spearman
from utils.metrics import format_table
from utils.visualization import plot_rank_ablation, save_all


# ──────────────────────────────────────────────────────────────────────────────
# Core
# ──────────────────────────────────────────────────────────────────────────────

def run_rank_ablation(
    data_dir:    str,
    adapters_root: str,
    ranks:       List[int],
    num_subjects: int,
    output_dir:  str,
    model_id:    str,
    device:      str,
    attack_mode: str = "prompt_generation",
    n_candidates: int = 10,
    num_steps:   int = 50,
    num_timesteps: int = 20,
    scorer:      str = "likelihood_ratio",
    subject_ids: Optional[List[str]] = None,
) -> Dict:
    """
    Run MIA + reconstruction for every rank in `ranks`.

    Returns
    -------
    dict with per-rank results and aggregate
    """
    splits = json.loads((Path(data_dir) / "splits.json").read_text())

    if subject_ids is None:
        subject_ids = list(splits.keys())[:num_subjects]

    adapters_root = Path(adapters_root)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Storage: per-rank results
    rank_results: Dict[int, Dict] = {}

    for rank in ranks:
        print(f"\n{'='*65}")
        print(f"  LoRA rank = {rank}")
        print(f"{'='*65}\n")

        # Adapter directory / naming convention
        # Try flat dir first: adapters_root/lora_rank{R}_subject*.safetensors
        flat_hits = list(adapters_root.glob(f"lora_rank{rank}_subject*.safetensors"))
        if flat_hits:
            adapter_dir = adapters_root
        else:
            # Try sub-directory: adapters_root/rank{R}/
            adapter_dir = adapters_root / f"rank{rank}"
            if not adapter_dir.exists():
                print(f"  ⚠ No adapters found for rank {rank}, skipping.")
                continue

        # ── load wrapper once per rank ────────────────────────────────────────
        wrapper = DiffusionWrapper(
            model_id=model_id, device=device,
            dtype=torch.float16 if device == "cuda" else torch.float32,
        )

        # ── MIA ───────────────────────────────────────────────────────────────
        mia_results = run_mia(
            wrapper       = wrapper,
            data_dir      = data_dir,
            adapter_dir   = str(adapter_dir),
            adapter_type  = "lora",
            subject_ids   = subject_ids,
            scorer        = scorer,
            num_timesteps = num_timesteps,
            rank          = rank,
        )
        mia_agg = mia_results["aggregate"]

        # ── Reconstruction ────────────────────────────────────────────────────
        recon_results = run_reconstruction(
            wrapper       = wrapper,
            data_dir      = data_dir,
            adapter_dir   = str(adapter_dir),
            adapter_type  = "lora",
            subject_ids   = subject_ids,
            attack_mode   = attack_mode,
            n_candidates  = n_candidates,
            num_steps     = num_steps,
            rank          = rank,
            output_dir    = str(out / f"rank{rank}_images"),
        )
        recon_agg = recon_results["aggregate"]

        # ── Store ─────────────────────────────────────────────────────────────
        rank_results[rank] = {
            "mia":          mia_agg,
            "reconstruction": recon_agg,
            # Flattened conveniences
            "auc_roc":      mia_agg["auc_roc"],
            "tpr_001":      mia_agg["tpr_at_fpr_001"],
            "advantage":    mia_agg["max_advantage"],
            "lpips":        recon_agg["lpips_mean"],
            "mse":          recon_agg["mse_mean"],
            "psnr":         recon_agg["psnr_mean"],
            "ssim":         recon_agg["ssim_mean"],
            "n_subjects":   mia_agg["num_subjects"],
        }

        # Save per-rank JSON
        (out / f"rank{rank}_results.json").write_text(
            json.dumps({
                "rank": rank,
                "mia": mia_results,
                "reconstruction": recon_results,
            }, indent=2,
               default=lambda x: float(x) if isinstance(x, np.floating) else x)
        )

        print(f"\n  rank={rank}  AUC={mia_agg['auc_roc']:.3f}  "
              f"LPIPS={recon_agg['lpips_mean']:.3f}  "
              f"PSNR={recon_agg['psnr_mean']:.1f}dB")

        del wrapper
        torch.cuda.empty_cache()

    # ── Spearman correlation (paper: ρ = 0.83 for LPIPS) ─────────────────────
    valid_ranks = [r for r in ranks if r in rank_results]

    if len(valid_ranks) >= 3:
        lpips_vals  = [rank_results[r]["lpips"]   for r in valid_ranks]
        auc_vals    = [rank_results[r]["auc_roc"]  for r in valid_ranks]

        rho_lpips   = rank_leakage_spearman(valid_ranks, lpips_vals)
        rho_auc     = rank_leakage_spearman(valid_ranks, auc_vals)

        print(f"\n[Spearman]  rank vs LPIPS:   ρ = {rho_lpips['spearman_rho']:+.3f}  "
              f"p = {rho_lpips['p_value']:.4f}")
        print(f"[Spearman]  rank vs AUC-ROC: ρ = {rho_auc['spearman_rho']:+.3f}  "
              f"p = {rho_auc['p_value']:.4f}")
    else:
        rho_lpips = rho_auc = {}

    # ── Summary table ─────────────────────────────────────────────────────────
    table = {
        f"LoRA r={r}": {
            "AUC-ROC":  rank_results[r]["auc_roc"],
            "TPR@1%FPR":rank_results[r]["tpr_001"],
            "Advantage":rank_results[r]["advantage"],
            "LPIPS ↓":  rank_results[r]["lpips"],
            "PSNR ↑":   rank_results[r]["psnr"],
            "SSIM ↑":   rank_results[r]["ssim"],
        }
        for r in valid_ranks
    }
    print(format_table(table, title="Rank Ablation Summary (Table 2)"))

    # ── Save summary ──────────────────────────────────────────────────────────
    summary = {
        "ranks":         valid_ranks,
        "rank_results":  {str(r): rank_results[r] for r in valid_ranks},
        "spearman_lpips":rho_lpips,
        "spearman_auc":  rho_auc,
    }
    (out / "rank_ablation_summary.json").write_text(
        json.dumps(summary, indent=2,
                   default=lambda x: float(x) if isinstance(x, np.floating) else x)
    )

    # ── Figures ───────────────────────────────────────────────────────────────
    figs = {}

    # Primary ablation plot (Figure 4)
    figs["figure4_rank_ablation"] = plot_rank_ablation(
        ranks  = valid_ranks,
        metrics_series = {
            "AUC-ROC":    [rank_results[r]["auc_roc"]  for r in valid_ranks],
            "LPIPS ↓":    [rank_results[r]["lpips"]    for r in valid_ranks],
            "TPR@1%FPR":  [rank_results[r]["tpr_001"]  for r in valid_ranks],
        },
        title = "Privacy Leakage vs LoRA Rank",
    )

    # PSNR / SSIM subplot
    figs["rank_recon_quality"] = plot_rank_ablation(
        ranks  = valid_ranks,
        metrics_series = {
            "PSNR (dB) ↑": [rank_results[r]["psnr"] for r in valid_ranks],
            "SSIM ↑":       [rank_results[r]["ssim"] for r in valid_ranks],
        },
        title = "Reconstruction Quality vs LoRA Rank",
    )

    save_all(figs, out)
    print(f"\n✓ Rank ablation complete. Results in {out}")

    return summary


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA Rank Ablation Study")
    p.add_argument("--data_dir",      type=str, default="./data")
    p.add_argument("--adapters_root", type=str, default="./adapters",
                   help="Root dir containing rank-specific adapter files")
    p.add_argument("--ranks",         nargs="+", type=int, default=[4, 8, 16, 32, 64])
    p.add_argument("--num_subjects",  type=int, default=30)
    p.add_argument("--output_dir",    type=str, default="./results/rank_ablation")
    p.add_argument("--model_id",      type=str, default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--device",        type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--attack_mode",   type=str, default="prompt_generation",
                   choices=["prompt_generation", "gradient_extraction"])
    p.add_argument("--n_candidates",  type=int, default=10)
    p.add_argument("--num_steps",     type=int, default=50)
    p.add_argument("--num_timesteps", type=int, default=20)
    p.add_argument("--scorer",        type=str, default="likelihood_ratio")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[Rank Ablation]  ranks={args.ranks}  subjects={args.num_subjects}  "
          f"device={args.device}")
    run_rank_ablation(
        data_dir      = args.data_dir,
        adapters_root = args.adapters_root,
        ranks         = args.ranks,
        num_subjects  = args.num_subjects,
        output_dir    = args.output_dir,
        model_id      = args.model_id,
        device        = args.device,
        attack_mode   = args.attack_mode,
        n_candidates  = args.n_candidates,
        num_steps     = args.num_steps,
        num_timesteps = args.num_timesteps,
        scorer        = args.scorer,
    )


if __name__ == "__main__":
    main()

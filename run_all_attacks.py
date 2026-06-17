"""
experiments/run_all_attacks.py
────────────────────────────────
Orchestrates ALL experiments in the paper.

Running this script end-to-end reproduces every table and figure.
It reads from a config JSON so that individual experiments can be
re-run without re-running the whole pipeline.

Expected directory layout after training
─────────────────────────────────────────
adapters/
  lora_rank4_subject0.safetensors
  lora_rank8_subject0.safetensors
  …
  lora_rank4_subject29.safetensors
  …
adapters_ti/
  ti_subject0.pt
  …

Usage
─────
# Full paper reproduction
python experiments/run_all_attacks.py \\
    --data_dir ./data \\
    --lora_dir ./adapters \\
    --ti_dir   ./adapters_ti \\
    --output_dir ./results \\
    --num_subjects 30

# Skip training; only regenerate figures from cached JSON
python experiments/run_all_attacks.py \\
    --only_figures \\
    --results_cache ./results/run_cache.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ──────────────────────────────────────────────────────────────────────────────
# Sub-experiment runners (import lazily to keep startup fast)
# ──────────────────────────────────────────────────────────────────────────────

def _run_mia_all(args: argparse.Namespace, output_dir: Path, sids: List[str]) -> Dict:
    """Run MIA for all adapter configs and return aggregated dict."""
    from models.diffusion_wrapper import DiffusionWrapper
    from attacks.membership_inference import run_mia
    from utils.visualization import plot_roc, save_all

    results: Dict[str, Dict] = {}
    roc_series = {}

    configs = []
    if args.lora_dir:
        for r in args.ranks:
            configs.append(("lora", Path(args.lora_dir), f"LoRA r={r}", r))
    if args.ti_dir:
        configs.append(("textual_inversion", Path(args.ti_dir), "TI", None))

    for adapter_type, adapter_dir, label, rank in configs:
        print(f"\n[MIA] {label} …")
        t0 = time.time()

        wrapper = DiffusionWrapper(
            model_id=args.model_id, device=args.device,
            dtype=torch.float16 if args.device == "cuda" else torch.float32,
        )
        res = run_mia(
            wrapper, args.data_dir, str(adapter_dir),
            adapter_type, sids,
            scorer=args.scorer, num_timesteps=args.num_timesteps, rank=rank,
        )
        del wrapper
        torch.cuda.empty_cache()

        results[label] = res
        all_m  = np.concatenate([p["member_scores"]    for p in res["per_subject"]])
        all_nm = np.concatenate([p["nonmember_scores"] for p in res["per_subject"]])
        roc_series[label] = (all_m, all_nm)

        print(f"  {label}: AUC={res['aggregate']['auc_roc']:.4f}  [{time.time()-t0:.0f}s]")

    # Save JSON
    (output_dir / "mia_all.json").write_text(
        json.dumps(results, indent=2,
                   default=lambda x: float(x) if isinstance(x, np.floating) else x)
    )

    # Figure 1: ROC
    fig = plot_roc(
        roc_series,
        higher_is_member=(args.scorer == "likelihood_ratio"),
        title="MIA ROC Curves — All Adapters (Figure 1)",
    )
    save_all({"figure1_mia_roc": fig}, output_dir)

    return results


def _run_recon_all(args: argparse.Namespace, output_dir: Path, sids: List[str]) -> Dict:
    """Run reconstruction attack for all adapter configs."""
    from models.diffusion_wrapper import DiffusionWrapper
    from attacks.data_reconstruction import run_reconstruction
    from utils.visualization import save_all

    results: Dict[str, Dict] = {}

    configs = []
    if args.lora_dir:
        for r in args.ranks:
            configs.append(("lora", Path(args.lora_dir), f"LoRA r={r}", r))
    if args.ti_dir:
        configs.append(("textual_inversion", Path(args.ti_dir), "TI", None))

    for adapter_type, adapter_dir, label, rank in configs:
        print(f"\n[Reconstruction] {label} …")
        t0 = time.time()

        wrapper = DiffusionWrapper(
            model_id=args.model_id, device=args.device,
            dtype=torch.float16 if args.device == "cuda" else torch.float32,
        )
        res = run_reconstruction(
            wrapper, args.data_dir, str(adapter_dir),
            adapter_type, sids,
            attack_mode=args.attack_mode,
            n_candidates=args.n_candidates,
            num_steps=args.num_steps,
            rank=rank,
            output_dir=str(output_dir / f"images_{label.replace(' ', '_').replace('=', '')}"),
        )
        del wrapper
        torch.cuda.empty_cache()

        results[label] = res
        agg = res["aggregate"]
        print(f"  {label}: LPIPS={agg['lpips_mean']:.3f}  PSNR={agg['psnr_mean']:.1f}  "
              f"[{time.time()-t0:.0f}s]")

    (output_dir / "recon_all.json").write_text(
        json.dumps(results, indent=2,
                   default=lambda x: float(x) if isinstance(x, np.floating) else x)
    )
    return results


def _run_rank_ablation(args: argparse.Namespace, output_dir: Path, sids: List[str]) -> Dict:
    """Run rank ablation and produce Figure 4."""
    from experiments.lora_rank_ablation import run_rank_ablation
    return run_rank_ablation(
        data_dir      = args.data_dir,
        adapters_root = args.lora_dir,
        ranks         = args.ranks,
        num_subjects  = len(sids),
        output_dir    = str(output_dir / "rank_ablation"),
        model_id      = args.model_id,
        device        = args.device,
        attack_mode   = args.attack_mode,
        n_candidates  = args.n_candidates,
        num_steps     = args.num_steps,
        num_timesteps = args.num_timesteps,
        scorer        = args.scorer,
        subject_ids   = sids,
    )


def _produce_table1(mia_results: Dict, recon_results: Dict) -> str:
    """Format Table 1 from paper."""
    from utils.metrics import format_table, compute_privacy_score

    rows: Dict[str, Dict] = {}
    all_labels = set(mia_results.keys()) | set(recon_results.keys())

    for label in sorted(all_labels):
        mia_agg   = mia_results.get(label, {}).get("aggregate", {})
        recon_agg = recon_results.get(label, {}).get("aggregate", {})

        rows[label] = {
            "AUC-ROC":    mia_agg.get("auc_roc", float("nan")),
            "TPR@1%FPR":  mia_agg.get("tpr_at_fpr_001", float("nan")),
            "Advantage":  mia_agg.get("max_advantage", float("nan")),
            "LPIPS ↓":    recon_agg.get("lpips_mean", float("nan")),
            "PSNR ↑":     recon_agg.get("psnr_mean", float("nan")),
            "SSIM ↑":     recon_agg.get("ssim_mean", float("nan")),
        }
        if mia_agg and recon_agg:
            rows[label]["Privacy Score"] = compute_privacy_score(mia_agg, recon_agg)

    return format_table(rows, title="Table 1: Privacy Leakage — All Adapters")


# ──────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cache_file = out / "run_cache.json"

    # ── Load subject IDs ──────────────────────────────────────────────────────
    splits = json.loads((Path(args.data_dir) / "splits.json").read_text())
    sids   = list(splits.keys())[:args.num_subjects]
    print(f"\n[RunAll]  subjects={len(sids)}  device={args.device}")
    print(f"  ranks={args.ranks}  scorer={args.scorer}")
    print(f"  output → {out}\n")

    # ── Figure-only mode ──────────────────────────────────────────────────────
    if args.only_figures:
        print("[only_figures] Loading from cache …")
        cache = json.loads(Path(args.results_cache or str(cache_file)).read_text())
        mia_results   = cache.get("mia", {})
        recon_results = cache.get("recon", {})
    else:
        # ── Step 1: MIA ───────────────────────────────────────────────────────
        print("=" * 65)
        print("Step 1/3 — Membership Inference Attacks")
        print("=" * 65)
        mia_results = _run_mia_all(args, out, sids)

        # ── Step 2: Reconstruction ────────────────────────────────────────────
        print("\n" + "=" * 65)
        print("Step 2/3 — Data Reconstruction Attacks")
        print("=" * 65)
        recon_results = _run_recon_all(args, out, sids)

        # Cache
        cache_file.write_text(
            json.dumps({"mia": mia_results, "recon": recon_results}, indent=2,
                       default=lambda x: float(x) if isinstance(x, np.floating) else x)
        )

        # ── Step 3: Rank ablation ─────────────────────────────────────────────
        if args.lora_dir and len(args.ranks) > 1:
            print("\n" + "=" * 65)
            print("Step 3/3 — LoRA Rank Ablation")
            print("=" * 65)
            _run_rank_ablation(args, out, sids)

    # ── Table 1 ───────────────────────────────────────────────────────────────
    table1 = _produce_table1(mia_results, recon_results)
    print(table1)
    (out / "table1.txt").write_text(table1)

    print(f"\n✓ All experiments complete.  Results → {out}")
    print(f"  table1.txt            — Table 1")
    print(f"  figure1_mia_roc.*     — Figure 1")
    print(f"  rank_ablation/        — Figure 4 + Table 2")
    print(f"  mia_all.json          — Full MIA results")
    print(f"  recon_all.json        — Full reconstruction results")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run all privacy-leakage experiments")
    p.add_argument("--data_dir",     type=str, default="./data")
    p.add_argument("--lora_dir",     type=str, default=None)
    p.add_argument("--ti_dir",       type=str, default=None)
    p.add_argument("--ranks",        nargs="+", type=int, default=[4, 8, 16, 32, 64])
    p.add_argument("--num_subjects", type=int, default=30)
    p.add_argument("--output_dir",   type=str, default="./results")
    p.add_argument("--model_id",     type=str, default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--device",       type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--attack_mode",  type=str, default="prompt_generation")
    p.add_argument("--n_candidates", type=int, default=10)
    p.add_argument("--num_steps",    type=int, default=50)
    p.add_argument("--num_timesteps",type=int, default=20)
    p.add_argument("--scorer",       type=str, default="likelihood_ratio")
    p.add_argument("--only_figures", action="store_true")
    p.add_argument("--results_cache",type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    main()

"""
experiments/compare_adapters.py
─────────────────────────────────
Full LoRA vs Textual Inversion comparison.

Reproduces:
  • Paper Table 1 (all rows simultaneously)
  • Paper Figure 1 (overlaid ROC curves)
  • Paper Figure 5 (bar chart comparison)
  • Paper key finding: TI leaks more than LoRA despite 530× fewer parameters

Key paper finding
─────────────────
  TI (768 params) leaks MORE than LoRA r=64 (>> 768 params).
  Reconstruction LPIPS:  TI=0.19  vs  LoRA r=16=0.31

Usage
─────
python experiments/compare_adapters.py \\
    --data_dir ./data \\
    --lora_dir ./adapters \\
    --ti_dir   ./adapters_ti \\
    --ranks 4 16 64 \\
    --num_subjects 30 \\
    --output_dir ./results/comparison
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.diffusion_wrapper import DiffusionWrapper
from attacks.membership_inference import run_mia
from attacks.data_reconstruction import run_reconstruction
from utils.metrics import format_table, compute_privacy_score
from utils.visualization import (
    plot_roc,
    plot_adapter_comparison,
    plot_privacy_heatmap,
    save_all,
)


# ──────────────────────────────────────────────────────────────────────────────

def _param_count_lora(rank: int, n_layers: int = 32, dim: int = 768) -> int:
    """Approximate parameter count for LoRA at given rank."""
    return n_layers * 2 * rank * dim


def _param_count_ti(dim: int = 768) -> int:
    """TI uses one embedding vector."""
    return dim


def _format_ratio(rank: int) -> str:
    return f"{_param_count_lora(rank) / _param_count_ti():.0f}×"


# ──────────────────────────────────────────────────────────────────────────────

def run_full_comparison(
    data_dir:     str,
    lora_dir:     Optional[str],
    ti_dir:       Optional[str],
    ranks:        List[int],
    num_subjects: int,
    output_dir:   str,
    model_id:     str,
    device:       str,
    n_candidates: int = 10,
    num_steps:    int = 50,
    num_timesteps: int = 20,
    scorer:       str = "likelihood_ratio",
    subject_ids:  Optional[List[str]] = None,
) -> None:

    splits = json.loads((Path(data_dir) / "splits.json").read_text())
    if subject_ids is None:
        subject_ids = list(splits.keys())[:num_subjects]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Collect per-config results ────────────────────────────────────────────
    # config_key → (mia_agg, recon_agg, all_m_scores, all_nm_scores)
    configs: Dict[str, Dict] = {}

    # LoRA configs
    if lora_dir:
        for rank in ranks:
            label     = f"LoRA r={rank}"
            param_cnt = _param_count_lora(rank)
            ratio_str = _format_ratio(rank)

            print(f"\n[Compare] {label}  ({param_cnt:,} params,  {ratio_str} vs TI)")

            adapter_dir = Path(lora_dir)
            wrapper = DiffusionWrapper(
                model_id=model_id, device=device,
                dtype=torch.float16 if device == "cuda" else torch.float32,
            )

            mia_res   = run_mia(wrapper, data_dir, str(adapter_dir),
                                "lora", subject_ids, scorer, num_timesteps, rank=rank)
            recon_res = run_reconstruction(wrapper, data_dir, str(adapter_dir),
                                           "lora", subject_ids,
                                           attack_mode="prompt_generation",
                                           n_candidates=n_candidates,
                                           num_steps=num_steps, rank=rank)

            all_m  = np.concatenate([p["member_scores"]    for p in mia_res["per_subject"]])
            all_nm = np.concatenate([p["nonmember_scores"] for p in mia_res["per_subject"]])

            configs[label] = {
                "mia":      mia_res["aggregate"],
                "recon":    recon_res["aggregate"],
                "m_scores": all_m,
                "nm_scores":all_nm,
                "params":   param_cnt,
                "rank":     rank,
            }

            del wrapper
            torch.cuda.empty_cache()

    # Textual Inversion config
    if ti_dir:
        label     = "Textual Inversion"
        param_cnt = _param_count_ti()

        print(f"\n[Compare] {label}  ({param_cnt} params)")

        adapter_dir = Path(ti_dir)
        wrapper = DiffusionWrapper(
            model_id=model_id, device=device,
            dtype=torch.float16 if device == "cuda" else torch.float32,
        )

        mia_res   = run_mia(wrapper, data_dir, str(adapter_dir),
                            "textual_inversion", subject_ids, scorer, num_timesteps)
        recon_res = run_reconstruction(wrapper, data_dir, str(adapter_dir),
                                       "textual_inversion", subject_ids,
                                       attack_mode="prompt_generation",
                                       n_candidates=n_candidates,
                                       num_steps=num_steps)

        all_m  = np.concatenate([p["member_scores"]    for p in mia_res["per_subject"]])
        all_nm = np.concatenate([p["nonmember_scores"] for p in mia_res["per_subject"]])

        configs[label] = {
            "mia":      mia_res["aggregate"],
            "recon":    recon_res["aggregate"],
            "m_scores": all_m,
            "nm_scores":all_nm,
            "params":   param_cnt,
            "rank":     None,
        }

        del wrapper
        torch.cuda.empty_cache()

    # ── Build summary table ───────────────────────────────────────────────────
    table_rows: Dict[str, Dict] = {}
    for label, cfg in configs.items():
        ps = compute_privacy_score(cfg["mia"], cfg["recon"])
        table_rows[label] = {
            "Params":      cfg["params"],
            "AUC-ROC":     cfg["mia"]["auc_roc"],
            "TPR@1%FPR":   cfg["mia"]["tpr_at_fpr_001"],
            "Advantage":   cfg["mia"]["max_advantage"],
            "LPIPS ↓":     cfg["recon"]["lpips_mean"],
            "PSNR ↑":      cfg["recon"]["psnr_mean"],
            "SSIM ↑":      cfg["recon"]["ssim_mean"],
            "Privacy Score": ps,
        }

    print(format_table(table_rows, title="Table 1: Privacy Leakage Comparison"))

    # Print parameter efficiency insight
    if "Textual Inversion" in configs and lora_dir:
        ti_lpips   = configs["Textual Inversion"]["recon"]["lpips_mean"]
        ti_params  = configs["Textual Inversion"]["params"]
        for label, cfg in configs.items():
            if "LoRA" in label and cfg["recon"]["lpips_mean"] > ti_lpips:
                ratio = cfg["params"] / ti_params
                print(f"\n  → {label} ({cfg['params']:,} params, {ratio:.0f}× more than TI)")
                print(f"    leaks LESS than TI despite more parameters:")
                print(f"    LPIPS {cfg['recon']['lpips_mean']:.3f} vs TI {ti_lpips:.3f}")

    # ── Figures ───────────────────────────────────────────────────────────────
    figs = {}

    # Figure 1: ROC overlay
    roc_series = {
        label: (cfg["m_scores"], cfg["nm_scores"])
        for label, cfg in configs.items()
    }
    higher = (scorer == "likelihood_ratio")
    figs["figure1_roc_comparison"] = plot_roc(
        roc_series, higher_is_member=higher,
        title="MIA ROC: LoRA vs Textual Inversion",
    )

    # Figure 5: Bar chart
    adapter_labels = list(configs.keys())
    figs["figure5_bar_comparison"] = plot_adapter_comparison(
        adapter_labels = adapter_labels,
        metrics_series = {
            "AUC-ROC":   [configs[l]["mia"]["auc_roc"]         for l in adapter_labels],
            "LPIPS ↓":   [configs[l]["recon"]["lpips_mean"]     for l in adapter_labels],
            "TPR@1%FPR": [configs[l]["mia"]["tpr_at_fpr_001"]   for l in adapter_labels],
        },
        errors = {
            "AUC-ROC":  [0.0] * len(adapter_labels),   # fill with std if available
        },
        title = "Privacy Leakage: LoRA vs Textual Inversion",
    )

    # Privacy heatmap (adapter × subject) — requires per-subject AUC
    try:
        subject_set = list(set.intersection(*[
            set(p["subject_id"] for p in configs[l]["mia"].get("per_subject", []))
            for l in configs if "per_subject" in configs[l].get("mia", {})
        ]))
    except (TypeError, KeyError):
        subject_set = []

    save_all(figs, out)

    # Save summary JSON
    (out / "comparison_summary.json").write_text(
        json.dumps(
            {label: {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                     for k, v in cfg.items() if k not in ("m_scores", "nm_scores")}
             for label, cfg in configs.items()},
            indent=2,
            default=lambda x: float(x) if isinstance(x, np.floating) else x,
        )
    )

    print(f"\n✓ Comparison complete. Results in {out}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Adapter Privacy Comparison")
    p.add_argument("--data_dir",     type=str, default="./data")
    p.add_argument("--lora_dir",     type=str, default=None)
    p.add_argument("--ti_dir",       type=str, default=None)
    p.add_argument("--ranks",        nargs="+", type=int, default=[4, 16, 64])
    p.add_argument("--num_subjects", type=int, default=30)
    p.add_argument("--output_dir",   type=str, default="./results/comparison")
    p.add_argument("--model_id",     type=str, default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--device",       type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n_candidates", type=int, default=10)
    p.add_argument("--num_steps",    type=int, default=50)
    p.add_argument("--num_timesteps",type=int, default=20)
    p.add_argument("--scorer",       type=str, default="likelihood_ratio")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.lora_dir and not args.ti_dir:
        print("Provide --lora_dir and/or --ti_dir")
        return
    run_full_comparison(
        data_dir     = args.data_dir,
        lora_dir     = args.lora_dir,
        ti_dir       = args.ti_dir,
        ranks        = args.ranks,
        num_subjects = args.num_subjects,
        output_dir   = args.output_dir,
        model_id     = args.model_id,
        device       = args.device,
        n_candidates = args.n_candidates,
        num_steps    = args.num_steps,
        num_timesteps= args.num_timesteps,
        scorer       = args.scorer,
    )


if __name__ == "__main__":
    main()

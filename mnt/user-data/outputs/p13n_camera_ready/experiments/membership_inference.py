"""
experiments/membership_inference.py
────────────────────────────────────
End-to-end MIA experiment pipeline.

Reproduces:
  • Paper Table 1 (AUC-ROC column)
  • Paper Figure 1 (ROC curves)
  • Paper Figure 2 (score distributions)

Modes
─────
--mode single   : one subject, one adapter
--mode multi    : all subjects, aggregate results + per-subject JSON
--mode compare  : overlay ROC curves for LoRA (multiple ranks) and TI

Usage examples
──────────────
# Single subject quick test
python experiments/membership_inference.py \\
    --mode single \\
    --data_dir ./data \\
    --adapter_path ./adapters/lora_rank16_subject0.safetensors \\
    --subject_id 0 \\
    --adapter_type lora

# Full multi-subject run (Table 1, LoRA r=16)
python experiments/membership_inference.py \\
    --mode multi \\
    --data_dir ./data \\
    --adapter_dir ./adapters \\
    --adapter_type lora \\
    --rank 16 \\
    --num_subjects 30 \\
    --output_dir ./results/mia_lora16

# ROC overlay for all adapter types (Figure 1)
python experiments/membership_inference.py \\
    --mode compare \\
    --data_dir ./data \\
    --lora_dirs ./adapters/rank4 ./adapters/rank16 ./adapters/rank64 \\
    --ti_dir ./adapters_ti \\
    --output_dir ./results/mia_compare
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.diffusion_wrapper import DiffusionWrapper
from attacks.membership_inference import (
    run_mia,
    evaluate_mia,
    score_likelihood_ratio,
    score_loss,
    load_images,
)
from utils.metrics import (
    compute_mia_metrics,
    bootstrap_ci,
    wilcoxon_test,
    format_table,
)
from utils.visualization import (
    plot_roc,
    plot_score_distributions,
    save_all,
)


# ──────────────────────────────────────────────────────────────────────────────
# Mode: single subject
# ──────────────────────────────────────────────────────────────────────────────

def run_single(args: argparse.Namespace) -> None:
    splits = json.loads((Path(args.data_dir) / "splits.json").read_text())
    sid    = args.subject_id
    info   = splits[sid]

    wrapper = DiffusionWrapper(
        model_id=args.model_id, device=args.device,
        dtype=torch.float16 if args.device == "cuda" else torch.float32,
    )

    subjects_dir = Path(args.data_dir) / "subjects"
    sdir  = subjects_dir / info["subject_dir"]
    prompt = info["class_prompt"]

    m_paths  = [sdir / n for n in info["train"][:args.max_images]]
    nm_paths = [sdir / n for n in info["test"][:args.max_images]]

    m_imgs  = load_images(m_paths)
    nm_imgs = load_images(nm_paths)

    # Load adapter
    if args.adapter_type == "lora":
        from safetensors.torch import load_file
        adapter_state = load_file(args.adapter_path)
    else:
        ckpt = torch.load(args.adapter_path, map_location="cpu")
        adapter_state = {}
        for tok, emb in ckpt["embeddings"].items():
            wrapper.apply_textual_inversion(tok, emb)

    # Score
    scorer = args.scorer
    m_scores  = score_likelihood_ratio(wrapper, m_imgs,  [prompt]*len(m_imgs),
                                       adapter_state, args.adapter_type, args.num_timesteps) \
                if scorer == "likelihood_ratio" else \
                score_loss(wrapper, m_imgs,  [prompt]*len(m_imgs),
                           adapter_state, args.adapter_type, args.num_timesteps)

    nm_scores = score_likelihood_ratio(wrapper, nm_imgs, [prompt]*len(nm_imgs),
                                       adapter_state, args.adapter_type, args.num_timesteps) \
                if scorer == "likelihood_ratio" else \
                score_loss(wrapper, nm_imgs, [prompt]*len(nm_imgs),
                           adapter_state, args.adapter_type, args.num_timesteps)

    higher = (scorer == "likelihood_ratio")
    metrics = compute_mia_metrics(m_scores, nm_scores, higher_is_member=higher)

    # Bootstrap CI on AUC-ROC
    all_scores = np.concatenate([m_scores if higher else -m_scores,
                                  nm_scores if higher else -nm_scores])
    all_labels = np.concatenate([np.ones(len(m_scores)), np.zeros(len(nm_scores))])

    from sklearn.metrics import roc_auc_score as _auc

    def _auc_fn(s):
        # s is a flat array; labels constructed inside bootstrap so we
        # need a closure over all_labels sampled in the same way
        return _auc(all_labels[:len(s)], s)   # conservative approximation

    auc_mean, auc_lo, auc_hi = bootstrap_ci(
        all_scores, lambda s: float(np.mean(s > 0)),   # membership rate as proxy
        n_boot=500,
    )

    # Print results
    print(f"\n{'='*60}")
    print(f"MIA Results — subject={sid}  adapter={args.adapter_type}")
    print(f"{'='*60}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<25} {v:.4f}")
        else:
            print(f"  {k:<25} {v}")

    # Save
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"mia_single_{sid}_{args.adapter_type}.json").write_text(
        json.dumps({
            "subject_id":    sid,
            "adapter_type":  args.adapter_type,
            "scorer":        scorer,
            "metrics":       metrics,
            "member_scores": m_scores.tolist(),
            "nonmember_scores": nm_scores.tolist(),
        }, indent=2)
    )

    # Figures
    tag = f"{args.adapter_type}_subject{sid}"
    figs = {
        f"roc_{tag}": plot_roc(
            {f"{args.adapter_type} (r={args.rank})": (m_scores, nm_scores)},
            higher_is_member=higher,
            title=f"MIA ROC — Subject {sid}  [{args.adapter_type}]",
        ),
        f"dist_{tag}": plot_score_distributions(
            m_scores, nm_scores,
            title=f"Score Distributions — Subject {sid}",
        ),
    }
    save_all(figs, out)


# ──────────────────────────────────────────────────────────────────────────────
# Mode: multi-subject (paper Table 1)
# ──────────────────────────────────────────────────────────────────────────────

def run_multi(args: argparse.Namespace) -> None:
    wrapper = DiffusionWrapper(
        model_id=args.model_id, device=args.device,
        dtype=torch.float16 if args.device == "cuda" else torch.float32,
    )

    splits = json.loads((Path(args.data_dir) / "splits.json").read_text())
    sids   = list(splits.keys())[:args.num_subjects]

    results = run_mia(
        wrapper      = wrapper,
        data_dir     = args.data_dir,
        adapter_dir  = args.adapter_dir,
        adapter_type = args.adapter_type,
        subject_ids  = sids,
        scorer       = args.scorer,
        num_timesteps= args.num_timesteps,
        rank         = args.rank,
    )

    agg = results["aggregate"]

    print(f"\n{'='*65}")
    print(f"Multi-subject MIA  —  {args.adapter_type}  (n={agg['num_subjects']})")
    print(f"{'='*65}")
    for k, v in agg.items():
        if isinstance(v, float):
            print(f"  {k:<30} {v:.4f}")

    # Collect all scores for aggregate plots
    all_m  = np.concatenate([np.array(p["member_scores"])    for p in results["per_subject"]])
    all_nm = np.concatenate([np.array(p["nonmember_scores"]) for p in results["per_subject"]])

    # Per-subject AUC distribution (CI)
    per_aucs = [p["metrics"]["auc_roc"] for p in results["per_subject"]]
    auc_mean = float(np.mean(per_aucs))
    auc_std  = float(np.std(per_aucs))
    print(f"\n  Per-subject AUC-ROC:  {auc_mean:.4f} ± {auc_std:.4f}")
    print(f"  Min: {min(per_aucs):.4f}   Max: {max(per_aucs):.4f}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save results JSON
    (out / f"mia_multi_{args.adapter_type}.json").write_text(
        json.dumps(results, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    )

    # Figures
    higher = (args.scorer == "likelihood_ratio")
    label  = f"{args.adapter_type} r={args.rank}" if args.rank else args.adapter_type
    figs = {
        f"roc_multi_{args.adapter_type}": plot_roc(
            {label: (all_m, all_nm)},
            higher_is_member=higher,
            title=f"Aggregate MIA ROC — {args.adapter_type} ({len(per_aucs)} subjects)",
        ),
        f"dist_multi_{args.adapter_type}": plot_score_distributions(
            all_m, all_nm,
            title=f"Score Distributions — {args.adapter_type}",
        ),
    }
    save_all(figs, out)
    print(f"\n✓ Results in {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Mode: compare  (overlay ROC — paper Figure 1)
# ──────────────────────────────────────────────────────────────────────────────

def run_compare(args: argparse.Namespace) -> None:
    """
    Load pre-computed per-subject results from JSON files and produce
    an overlay ROC curve for Figure 1 of the paper.

    Expects:
      args.result_files : list of JSON files produced by --mode multi
      args.result_labels: list of legend labels (same order)
    """
    if not args.result_files:
        print("--result_files required for --mode compare")
        return

    roc_series: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    higher = (args.scorer == "likelihood_ratio")

    for fpath, label in zip(args.result_files, args.result_labels):
        data = json.loads(Path(fpath).read_text())
        all_m  = np.concatenate([p["member_scores"]    for p in data["per_subject"]])
        all_nm = np.concatenate([p["nonmember_scores"] for p in data["per_subject"]])
        roc_series[label] = (all_m, all_nm)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig = plot_roc(
        roc_series,
        higher_is_member=higher,
        title="Membership Inference ROC — LoRA vs Textual Inversion",
    )
    from utils.visualization import _save
    _save(fig, str(out / "figure1_mia_roc"))

    # Also print a summary table
    rows = {}
    for fpath, label in zip(args.result_files, args.result_labels):
        data = json.loads(Path(fpath).read_text())
        rows[label] = data["aggregate"]

    print(format_table(rows, title="MIA Aggregate Results"))


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Membership Inference Attack Pipeline")

    p.add_argument("--mode", choices=["single", "multi", "compare"], default="multi")
    p.add_argument("--data_dir",    type=str, default="./data")
    p.add_argument("--adapter_dir", type=str, default="./adapters")
    p.add_argument("--adapter_path",type=str, default=None,
                   help="Path to a single adapter file (single mode)")
    p.add_argument("--subject_id",  type=str, default="0")
    p.add_argument("--adapter_type",type=str, default="lora",
                   choices=["lora", "textual_inversion"])
    p.add_argument("--scorer",      type=str, default="likelihood_ratio",
                   choices=["likelihood_ratio", "loss"])
    p.add_argument("--rank",        type=int, default=None)
    p.add_argument("--num_subjects",type=int, default=30)
    p.add_argument("--num_timesteps",type=int, default=20)
    p.add_argument("--max_images",  type=int, default=20)
    p.add_argument("--output_dir",  type=str, default="./results/mia")
    p.add_argument("--model_id",    type=str, default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--device",      type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    # compare-mode extras
    p.add_argument("--result_files",  nargs="*", default=[])
    p.add_argument("--result_labels", nargs="*", default=[])

    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[MIA Pipeline]  mode={args.mode}  device={args.device}")

    if args.mode == "single":
        run_single(args)
    elif args.mode == "multi":
        run_multi(args)
    elif args.mode == "compare":
        run_compare(args)


if __name__ == "__main__":
    main()

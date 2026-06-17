"""
experiments/data_extraction.py
────────────────────────────────
End-to-end data-reconstruction attack pipeline.

Reproduces:
  • Paper Table 1 (LPIPS, MSE, PSNR columns)
  • Paper Figure 3 (reconstruction grid)
  • Paper supplementary qualitative results

Modes
─────
--mode single      : one adapter, one subject, generate + evaluate
--mode multi       : all subjects, aggregate metrics
--mode compare     : LoRA vs TI side-by-side (paper Table 1, both adapters)
--mode budget_sweep: vary n_candidates 1..N, plot LPIPS vs budget

Usage examples
──────────────
# Quick single test
python experiments/data_extraction.py \\
    --mode single \\
    --data_dir ./data \\
    --adapter_path ./adapters/lora_rank16_subject0.safetensors \\
    --subject_id 0 \\
    --adapter_type lora \\
    --n_candidates 10

# Full multi-subject run  (Table 1 rows: LoRA r=16)
python experiments/data_extraction.py \\
    --mode multi \\
    --data_dir ./data \\
    --adapter_dir ./adapters \\
    --adapter_type lora \\
    --rank 16 \\
    --num_subjects 30 \\
    --output_dir ./results/recon_lora16

# LoRA vs TI comparison
python experiments/data_extraction.py \\
    --mode compare \\
    --data_dir ./data \\
    --lora_dir ./adapters \\
    --ti_dir ./adapters_ti \\
    --rank 16 \\
    --output_dir ./results/recon_compare
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
from attacks.data_reconstruction import (
    run_reconstruction,
    attack_prompt_generation,
    attack_gradient_extraction,
    ReconstructionEvaluator,
    load_images,
    tensor_to_pil,
)
from utils.metrics import compute_reconstruction_metrics, format_table
from utils.visualization import plot_reconstruction_grid, save_all


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_adapter(path: Path, adapter_type: str, wrapper: DiffusionWrapper) -> Dict:
    if adapter_type == "lora":
        from safetensors.torch import load_file
        return load_file(path)
    elif adapter_type == "textual_inversion":
        ckpt = torch.load(path, map_location="cpu")
        for tok, emb in ckpt["embeddings"].items():
            wrapper.apply_textual_inversion(tok, emb)
        return {}
    else:
        raise ValueError(adapter_type)


def _find_adapter(adapter_dir: Path, adapter_type: str,
                  sid: str, rank: Optional[int]) -> Optional[Path]:
    if adapter_type == "lora":
        r = f"rank{rank}_" if rank else "*"
        hits = list(adapter_dir.glob(f"lora_{r}subject{sid}.safetensors"))
        if not hits:
            hits = list(adapter_dir.glob(f"*subject{sid}.safetensors"))
    else:
        hits = list(adapter_dir.glob(f"ti_subject{sid}.pt"))
    return hits[0] if hits else None


# ──────────────────────────────────────────────────────────────────────────────
# Mode: single
# ──────────────────────────────────────────────────────────────────────────────

def run_single(args: argparse.Namespace) -> None:
    splits  = json.loads((Path(args.data_dir) / "splits.json").read_text())
    sid     = args.subject_id
    info    = splits[sid]
    sdir    = Path(args.data_dir) / "subjects" / info["subject_dir"]
    prompt  = info["class_prompt"]

    wrapper = DiffusionWrapper(
        model_id=args.model_id, device=args.device,
        dtype=torch.float16 if args.device == "cuda" else torch.float32,
    )

    adapter_state = _load_adapter(Path(args.adapter_path), args.adapter_type, wrapper)

    # Ground-truth member images
    gt_paths = [sdir / n for n in info["train"][:args.num_images]]
    gt_imgs  = load_images(gt_paths)   # [N, 3, 512, 512] in [-1,1]

    evaluator = ReconstructionEvaluator(device="cpu")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n[Reconstruction] subject={sid}  adapter={args.adapter_type}  "
          f"mode={args.attack_mode}  n_candidates={args.n_candidates}")

    if args.attack_mode == "prompt_generation":
        candidates = attack_prompt_generation(
            wrapper, prompt, adapter_state, args.adapter_type,
            n_candidates=args.n_candidates, num_steps=args.num_steps,
        )
        # Best-of-N per ground-truth image
        results: List[Dict] = []
        best_recons: List[torch.Tensor] = []
        for gt in gt_imgs:
            gt01 = ((gt + 1) / 2).clamp(0, 1).cpu()
            mse_vals = [torch.nn.functional.mse_loss(c.cpu(), gt01).item()
                        for c in candidates]
            best = candidates[int(np.argmin(mse_vals))]
            results.append(evaluator.evaluate(best, gt))
            best_recons.append(best)

    elif args.attack_mode == "gradient_extraction":
        results = []
        best_recons = []
        for gt in gt_imgs:
            recon = attack_gradient_extraction(
                wrapper, adapter_state, args.adapter_type,
                prompt=prompt, num_steps=args.num_steps,
            )
            results.append(evaluator.evaluate(recon, gt))
            best_recons.append(recon)
    else:
        raise ValueError(f"Unknown attack mode: {args.attack_mode}")

    avg = {k: float(np.mean([r[k] for r in results])) for k in results[0]}

    print(f"\n  LPIPS  = {avg['lpips']:.4f}")
    print(f"  MSE    = {avg['mse']:.4f}")
    print(f"  PSNR   = {avg['psnr']:.2f} dB")
    print(f"  SSIM   = {avg['ssim']:.4f}")

    # Save JSON
    payload = {
        "subject_id":   sid,
        "adapter_type": args.adapter_type,
        "attack_mode":  args.attack_mode,
        "avg_metrics":  avg,
        "per_image":    results,
    }
    (out / f"recon_single_{sid}_{args.adapter_type}.json").write_text(
        json.dumps(payload, indent=2)
    )

    # Save images
    img_dir = out / "images"
    img_dir.mkdir(exist_ok=True)
    for i, (gt, rc) in enumerate(zip(gt_imgs, best_recons)):
        tensor_to_pil(gt).save(img_dir / f"gt_{i:02d}.png")
        tensor_to_pil(rc).save(img_dir / f"recon_{i:02d}.png")

    # Figure
    figs = {
        f"recon_grid_{sid}_{args.adapter_type}": plot_reconstruction_grid(
            [gt.cpu() for gt in gt_imgs],
            best_recons,
            results,
            n_show=min(6, len(gt_imgs)),
            title=f"Reconstruction — subject {sid} [{args.adapter_type}]",
        )
    }
    save_all(figs, out)
    print(f"\n✓ Results in {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Mode: multi-subject
# ──────────────────────────────────────────────────────────────────────────────

def run_multi(args: argparse.Namespace) -> None:
    wrapper = DiffusionWrapper(
        model_id=args.model_id, device=args.device,
        dtype=torch.float16 if args.device == "cuda" else torch.float32,
    )

    splits   = json.loads((Path(args.data_dir) / "splits.json").read_text())
    sids     = list(splits.keys())[:args.num_subjects]
    out      = Path(args.output_dir)
    adapter_dir = Path(args.adapter_dir)

    results = run_reconstruction(
        wrapper       = wrapper,
        data_dir      = args.data_dir,
        adapter_dir   = str(adapter_dir),
        adapter_type  = args.adapter_type,
        subject_ids   = sids,
        attack_mode   = args.attack_mode,
        n_candidates  = args.n_candidates,
        num_steps     = args.num_steps,
        rank          = args.rank,
        output_dir    = str(out / "images"),
    )

    agg = results["aggregate"]
    print(f"\n{'='*60}")
    print(f"Aggregate  —  {args.adapter_type}  n={agg['n_subjects']}")
    print(f"  LPIPS  {agg['lpips_mean']:.4f} ± {agg['lpips_std']:.4f}")
    print(f"  MSE    {agg['mse_mean']:.4f}")
    print(f"  PSNR   {agg['psnr_mean']:.2f} dB")
    print(f"  SSIM   {agg['ssim_mean']:.4f}")
    print(f"{'='*60}")

    out.mkdir(parents=True, exist_ok=True)
    (out / f"recon_multi_{args.adapter_type}.json").write_text(
        json.dumps(results, indent=2,
                   default=lambda x: float(x) if isinstance(x, np.floating) else x)
    )
    print(f"\n✓ Results in {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Mode: compare  (paper Table 1)
# ──────────────────────────────────────────────────────────────────────────────

def run_compare(args: argparse.Namespace) -> None:
    """
    Compare LoRA (multiple ranks) vs Textual Inversion.
    Loads pre-computed JSONs or runs fresh experiments per adapter.
    """
    if args.result_files:
        # Load pre-computed
        table_rows: Dict[str, Dict] = {}
        for fpath, label in zip(args.result_files, args.result_labels):
            data = json.loads(Path(fpath).read_text())
            table_rows[label] = data["aggregate"]
        print(format_table(table_rows, title="Reconstruction Comparison"))
        return

    # Run fresh
    splits   = json.loads((Path(args.data_dir) / "splits.json").read_text())
    sids     = list(splits.keys())[:args.num_subjects]

    configs = []
    if args.lora_dir:
        configs.append(("lora", Path(args.lora_dir), f"LoRA r={args.rank}"))
    if args.ti_dir:
        configs.append(("textual_inversion", Path(args.ti_dir), "TI"))

    table_rows = {}

    for adapter_type, adapter_dir, label in configs:
        wrapper = DiffusionWrapper(
            model_id=args.model_id, device=args.device,
            dtype=torch.float16 if args.device == "cuda" else torch.float32,
        )
        results = run_reconstruction(
            wrapper, args.data_dir, str(adapter_dir),
            adapter_type, sids,
            attack_mode=args.attack_mode,
            n_candidates=args.n_candidates,
            rank=args.rank if adapter_type == "lora" else None,
        )
        table_rows[label] = results["aggregate"]

        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"recon_{adapter_type}.json").write_text(
            json.dumps(results, indent=2,
                       default=lambda x: float(x) if isinstance(x, np.floating) else x)
        )
        del wrapper
        torch.cuda.empty_cache()

    print(format_table(table_rows, title="Reconstruction: LoRA vs TI"))


# ──────────────────────────────────────────────────────────────────────────────
# Mode: budget sweep
# ──────────────────────────────────────────────────────────────────────────────

def run_budget_sweep(args: argparse.Namespace) -> None:
    """Vary n_candidates, record LPIPS — shows diminishing returns."""
    import matplotlib.pyplot as plt

    splits  = json.loads((Path(args.data_dir) / "splits.json").read_text())
    sid     = args.subject_id
    info    = splits[sid]
    sdir    = Path(args.data_dir) / "subjects" / info["subject_dir"]

    wrapper = DiffusionWrapper(
        model_id=args.model_id, device=args.device,
        dtype=torch.float16 if args.device == "cuda" else torch.float32,
    )
    adapter_state = _load_adapter(Path(args.adapter_path), args.adapter_type, wrapper)

    gt_paths = [sdir / n for n in info["train"][:3]]
    gt_imgs  = load_images(gt_paths)
    evaluator = ReconstructionEvaluator(device="cpu")
    prompt   = info["class_prompt"]

    budgets = list(range(1, args.max_budget + 1, args.budget_step))
    lpips_vals = []

    # Pre-generate max_budget candidates once
    all_candidates = attack_prompt_generation(
        wrapper, prompt, adapter_state, args.adapter_type,
        n_candidates=args.max_budget, num_steps=args.num_steps,
    )

    for budget in budgets:
        cands = all_candidates[:budget]
        per_gt = []
        for gt in gt_imgs:
            gt01 = ((gt + 1) / 2).clamp(0, 1).cpu()
            mse_vals = [torch.nn.functional.mse_loss(c.cpu(), gt01).item()
                        for c in cands]
            best = cands[int(np.argmin(mse_vals))]
            per_gt.append(evaluator.evaluate(best, gt)["lpips"])
        lpips_vals.append(float(np.mean(per_gt)))
        print(f"  budget={budget:3d}  LPIPS={lpips_vals[-1]:.4f}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(budgets, lpips_vals, marker="o", color="#1565C0", lw=2)
    ax.set_xlabel("Number of Candidates (budget)")
    ax.set_ylabel("LPIPS ↓")
    ax.set_title(f"Attack Budget vs Reconstruction Quality — {args.adapter_type}")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save_all({"budget_sweep": fig}, out)

    (out / "budget_sweep.json").write_text(
        json.dumps({"budgets": budgets, "lpips": lpips_vals}, indent=2)
    )
    print(f"\n✓ Budget sweep in {out}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Data Reconstruction Attack Pipeline")
    p.add_argument("--mode", choices=["single", "multi", "compare", "budget_sweep"],
                   default="multi")
    p.add_argument("--data_dir",      type=str, default="./data")
    p.add_argument("--adapter_dir",   type=str, default="./adapters")
    p.add_argument("--adapter_path",  type=str, default=None)
    p.add_argument("--adapter_type",  type=str, default="lora",
                   choices=["lora", "textual_inversion"])
    p.add_argument("--subject_id",    type=str, default="0")
    p.add_argument("--num_subjects",  type=int, default=30)
    p.add_argument("--num_images",    type=int, default=5,
                   help="Member images to reconstruct per subject")
    p.add_argument("--attack_mode",   type=str, default="prompt_generation",
                   choices=["prompt_generation", "gradient_extraction"])
    p.add_argument("--n_candidates",  type=int, default=10)
    p.add_argument("--num_steps",     type=int, default=50)
    p.add_argument("--rank",          type=int, default=None)
    p.add_argument("--output_dir",    type=str, default="./results/reconstruction")
    p.add_argument("--model_id",      type=str, default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--device",        type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    # compare mode
    p.add_argument("--lora_dir",      type=str, default=None)
    p.add_argument("--ti_dir",        type=str, default=None)
    p.add_argument("--result_files",  nargs="*", default=[])
    p.add_argument("--result_labels", nargs="*", default=[])
    # budget sweep
    p.add_argument("--max_budget",    type=int, default=20)
    p.add_argument("--budget_step",   type=int, default=2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[Reconstruction Pipeline]  mode={args.mode}  device={args.device}")

    if args.mode == "single":
        run_single(args)
    elif args.mode == "multi":
        run_multi(args)
    elif args.mode == "compare":
        run_compare(args)
    elif args.mode == "budget_sweep":
        run_budget_sweep(args)


if __name__ == "__main__":
    main()

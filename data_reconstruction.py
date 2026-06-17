"""
attacks/data_reconstruction.py
────────────────────────────────
Training-data reconstruction (extraction) attack on personalized adapters.

Attack modes
────────────
1. prompt_generation  : generate N images from the subject prompt; best-of-N
                        is returned as the reconstruction (simplest).
2. ddim_inversion     : DDIM-invert a reference image then denoise with the
                        adapter — tests latent-space alignment.
3. gradient_extraction: optimise a pixel image to minimise denoising loss
                        under the adapter (no text needed).

Evaluation metrics: LPIPS ↓, MSE ↓, PSNR ↑, SSIM ↑.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Optional LPIPS
try:
    import lpips as _lpips_lib  # type: ignore
    _LPIPS_AVAILABLE = True
except ImportError:
    _LPIPS_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

NORM   = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
DENORM = transforms.Normalize([-1.0, -1.0, -1.0], [2.0, 2.0, 2.0])  # [-1,1]→[0,1]

_to_tensor = transforms.Compose([
    transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.LANCZOS),
    transforms.ToTensor(),
    NORM,
])


def load_images(paths: List[Path]) -> torch.Tensor:
    """Returns [N, 3, 512, 512] in [-1, 1]."""
    return torch.stack([_to_tensor(Image.open(p).convert("RGB")) for p in paths])


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """[3, H, W] in [-1,1] or [0,1] → PIL."""
    t = t.detach().cpu().float()
    if t.min() < 0:
        t = (t + 1) / 2
    t = t.clamp(0, 1)
    arr = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def _ssim_single(a: torch.Tensor, b: torch.Tensor) -> float:
    """Simplified SSIM in [0,1] range on CPU."""
    a, b = a.float(), b.float()
    mu1, mu2 = a.mean(), b.mean()
    s1  = ((a - mu1) ** 2).mean()
    s2  = ((b - mu2) ** 2).mean()
    s12 = ((a - mu1) * (b - mu2)).mean()
    c1, c2 = 0.01**2, 0.03**2
    return float(
        ((2*mu1*mu2 + c1) * (2*s12 + c2)) /
        ((mu1**2 + mu2**2 + c1) * (s1 + s2 + c2))
    )


# ──────────────────────────────────────────────────────────────────────────────
# Metric computation
# ──────────────────────────────────────────────────────────────────────────────

class ReconstructionEvaluator:
    """Compute LPIPS, MSE, PSNR, SSIM between reconstructed and ground-truth."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        if _LPIPS_AVAILABLE:
            self._lpips = _lpips_lib.LPIPS(net="alex", verbose=False).to(device)
            self._lpips.eval()
        else:
            self._lpips = None
            print("[ReconstructionEvaluator] lpips not installed; LPIPS will be 0.")

    @torch.no_grad()
    def evaluate(
        self,
        reconstructed: torch.Tensor,   # [C, H, W] in [-1,1] or [0,1]
        ground_truth:  torch.Tensor,   # [C, H, W]
    ) -> Dict[str, float]:
        """Compute all metrics for one image pair."""

        def _to01(x: torch.Tensor) -> torch.Tensor:
            return ((x + 1) / 2).clamp(0, 1) if x.min() < -0.1 else x.clamp(0, 1)

        recon_01 = _to01(reconstructed.float().cpu())
        gt_01    = _to01(ground_truth.float().cpu())

        # MSE / PSNR
        mse  = F.mse_loss(recon_01, gt_01).item()
        psnr = float(10 * np.log10(1.0 / (mse + 1e-8)))

        # SSIM
        ssim = _ssim_single(recon_01, gt_01)

        # LPIPS (expects [-1, 1])
        if self._lpips is not None:
            recon_lp = (recon_01 * 2 - 1).unsqueeze(0).to(self.device)
            gt_lp    = (gt_01    * 2 - 1).unsqueeze(0).to(self.device)
            lp = self._lpips(recon_lp, gt_lp).item()
        else:
            lp = 0.0

        return {"lpips": lp, "mse": mse, "psnr": psnr, "ssim": ssim}

    def evaluate_batch(
        self,
        reconstructed_list: List[torch.Tensor],
        ground_truth_list:  List[torch.Tensor],
    ) -> Dict[str, float]:
        """Average metrics over a list of image pairs."""
        per = [
            self.evaluate(r, g)
            for r, g in zip(reconstructed_list, ground_truth_list)
        ]
        return {
            k: float(np.mean([m[k] for m in per]))
            for k in per[0]
        }


# ──────────────────────────────────────────────────────────────────────────────
# Attack modes
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def attack_prompt_generation(
    wrapper,
    prompt:        str,
    adapter_state: Dict[str, torch.Tensor],
    adapter_type:  str,
    n_candidates:  int = 10,
    num_steps:     int = 50,
    guidance:      float = 7.5,
) -> List[torch.Tensor]:
    """
    Generate n_candidates images under the adapter, return as list of [3,H,W]
    tensors in [0, 1].
    """
    if adapter_type == "lora":
        wrapper.apply_lora(adapter_state)

    imgs = []
    for seed in range(n_candidates):
        t = wrapper.generate(
            prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance,
            seed=seed,
            output_type="tensor",
        )
        imgs.append(t[0])   # [3, H, W] in [0,1]

    if adapter_type == "lora":
        wrapper.remove_lora(adapter_state)

    return imgs


def attack_gradient_extraction(
    wrapper,
    adapter_state: Dict[str, torch.Tensor],
    adapter_type:  str,
    prompt:        str = "a photo of a person",
    num_steps:     int = 400,
    lr:            float = 0.05,
    tv_weight:     float = 1e-3,
) -> torch.Tensor:
    """
    Optimise a pixel image to minimise denoising loss under the adapter.
    Returns a [3, 512, 512] float32 tensor in [0, 1].
    """
    device = wrapper.device
    dtype  = wrapper.dtype

    if adapter_type == "lora":
        wrapper.apply_lora(adapter_state)

    # Initialise in unconstrained space, project via sigmoid
    z = torch.randn(1, 3, 512, 512, device=device, requires_grad=True)
    opt = torch.optim.Adam([z], lr=lr)

    pbar = tqdm(range(num_steps), desc="Gradient extraction", leave=False)
    for step in pbar:
        opt.zero_grad()

        pixel = torch.sigmoid(z)                    # [0, 1]
        pixel_norm = pixel * 2 - 1                  # [-1, 1] for VAE

        with torch.no_grad():
            latents = wrapper.vae.encode(
                pixel_norm.to(dtype=dtype)
            ).latent_dist.sample() * wrapper._vae_scale

        noise = torch.randn_like(latents)
        t = torch.randint(
            0, wrapper.scheduler.config.num_train_timesteps, (1,), device=device
        ).long()
        noisy = wrapper.scheduler.add_noise(latents, noise, t)

        text_emb = wrapper.encode_text(prompt)  # [1, seq, dim]

        pred = wrapper.unet(
            noisy.to(dtype=dtype), t,
            encoder_hidden_states=text_emb.to(dtype=dtype)
        ).sample

        loss = F.mse_loss(pred.float(), noise.float())

        # Total-variation regulariser to suppress high-freq noise
        tv = (
            (pixel[:, :, 1:, :] - pixel[:, :, :-1, :]).abs().mean() +
            (pixel[:, :, :, 1:] - pixel[:, :, :, :-1]).abs().mean()
        )
        (loss + tv_weight * tv).backward()
        opt.step()

        if step % 50 == 0:
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "tv": f"{tv.item():.4f}"})

    if adapter_type == "lora":
        wrapper.remove_lora(adapter_state)

    with torch.no_grad():
        result = torch.sigmoid(z).squeeze(0).float().cpu()   # [3, 512, 512]
    return result


# ──────────────────────────────────────────────────────────────────────────────
# High-level pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_reconstruction(
    wrapper,
    data_dir:     str,
    adapter_dir:  str,
    adapter_type: str,
    subject_ids:  Optional[List[str]] = None,
    attack_mode:  str = "prompt_generation",
    n_candidates: int = 10,
    num_steps:    int = 50,
    rank:         Optional[int] = None,
    output_dir:   Optional[str] = None,
) -> Dict:
    """
    Run reconstruction attack across subjects.

    Returns dict with per-subject results and aggregate metrics.
    """
    import json
    splits = json.loads((Path(data_dir) / "splits.json").read_text())

    if subject_ids is None:
        subject_ids = list(splits.keys())

    subjects_dir = Path(data_dir) / "subjects"
    adapter_dir  = Path(adapter_dir)

    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
    else:
        out_path = None

    evaluator = ReconstructionEvaluator(device="cpu")

    all_lpips, all_mse, all_psnr, all_ssim = [], [], [], []
    per_subject: List[Dict] = []

    print(f"\n[Reconstruction] mode={attack_mode}  adapter_type={adapter_type}  "
          f"subjects={len(subject_ids)}\n")

    for sid in tqdm(subject_ids, desc="Subjects"):
        info = splits.get(sid)
        if info is None:
            continue

        # ── find adapter ────────────────────────────────────────────────────
        if adapter_type == "lora":
            rank_str   = f"rank{rank}_" if rank else "*"
            candidates = list(adapter_dir.glob(f"lora_{rank_str}subject{sid}.safetensors"))
            if not candidates:
                candidates = list(adapter_dir.glob(f"*subject{sid}.safetensors"))
            if not candidates:
                print(f"  ⚠ no LoRA for subject {sid}")
                continue
            from safetensors.torch import load_file
            adapter_state = load_file(candidates[0])
        elif adapter_type == "textual_inversion":
            candidates = list(adapter_dir.glob(f"ti_subject{sid}.pt"))
            if not candidates:
                continue
            ckpt = torch.load(candidates[0], map_location="cpu")
            for token, emb in ckpt["embeddings"].items():
                wrapper.apply_textual_inversion(token, emb)
            adapter_state = {}
        else:
            raise ValueError(adapter_type)

        # ── load ground-truth members (first 5) ─────────────────────────────
        sdir = subjects_dir / info["subject_dir"]
        gt_paths = [sdir / n for n in info["train"][:5]]
        gt_images = load_images(gt_paths)          # [N, 3, 512, 512] in [-1,1]

        prompt = info["class_prompt"]

        # ── reconstruct ──────────────────────────────────────────────────────
        if attack_mode == "prompt_generation":
            candidates_imgs = attack_prompt_generation(
                wrapper, prompt, adapter_state, adapter_type,
                n_candidates=n_candidates, num_steps=num_steps,
            )
            # Best-of-N: pick candidate closest to each GT image by MSE
            results: List[Dict] = []
            for gt_img in gt_images:
                gt_01 = ((gt_img + 1) / 2).clamp(0, 1).cpu()
                msev = [
                    F.mse_loss(c.cpu().float(), gt_01.float()).item()
                    for c in candidates_imgs
                ]
                best = candidates_imgs[int(np.argmin(msev))]
                results.append(evaluator.evaluate(best, gt_img))

        elif attack_mode == "gradient_extraction":
            results = []
            for gt_img in gt_images:
                recon = attack_gradient_extraction(
                    wrapper, adapter_state, adapter_type,
                    prompt=prompt, num_steps=num_steps,
                )
                results.append(evaluator.evaluate(recon, gt_img))

        else:
            raise ValueError(f"Unknown attack_mode: {attack_mode}")

        avg = {k: float(np.mean([r[k] for r in results])) for k in results[0]}

        all_lpips.append(avg["lpips"])
        all_mse.append(avg["mse"])
        all_psnr.append(avg["psnr"])
        all_ssim.append(avg["ssim"])

        per_subject.append({
            "subject_id":   sid,
            "avg_metrics":  avg,
            "per_image":    results,
        })

        print(f"  subject {sid}: LPIPS={avg['lpips']:.3f}  MSE={avg['mse']:.4f}  "
              f"PSNR={avg['psnr']:.1f}dB  SSIM={avg['ssim']:.3f}")

        # ── save reconstructions ─────────────────────────────────────────────
        if out_path and attack_mode == "prompt_generation":
            for i, cand in enumerate(candidates_imgs[:4]):
                tensor_to_pil(cand).save(
                    out_path / f"subject{sid}_candidate{i:02d}.png"
                )

    aggregate = {
        "lpips_mean": float(np.mean(all_lpips)),
        "lpips_std":  float(np.std(all_lpips)),
        "mse_mean":   float(np.mean(all_mse)),
        "psnr_mean":  float(np.mean(all_psnr)),
        "ssim_mean":  float(np.mean(all_ssim)),
        "n_subjects": len(per_subject),
    }

    print(f"\n[Reconstruction Aggregate]  "
          f"LPIPS={aggregate['lpips_mean']:.3f}±{aggregate['lpips_std']:.3f}  "
          f"PSNR={aggregate['psnr_mean']:.1f}dB")

    return {
        "adapter_type": adapter_type,
        "attack_mode":  attack_mode,
        "aggregate":    aggregate,
        "per_subject":  per_subject,
    }

"""
train_lora_adapters.py
──────────────────────
Train a LoRA adapter (victim model) for every subject in splits.json.

Paper setup
───────────
  • Base model : Stable Diffusion 1.5
  • Adapter    : LoRA injected into UNet attention {to_q, to_k, to_v, to_out.0}
  • Subjects   : 30 CelebA identities
  • Images/subj: 5–15 (train split from splits.json)
  • Ranks      : 4, 8, 16, 32, 64  (ablation; default = 16)
  • Epochs     : 200 (paper); 100 default here for speed
  • LR         : 1e-4

Usage
─────
# Single rank (paper default)
python train_lora_adapters.py --data_dir ./data --output_dir ./adapters --rank 16

# Rank ablation (all 5 ranks)
for r in 4 8 16 32 64; do
    python train_lora_adapters.py --rank $r --output_dir ./adapters/rank_$r
done
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

IMAGENET_NORM = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

class SubjectDataset(Dataset):
    """Images + repeated prompt for one subject (DreamBooth-style)."""

    def __init__(self, image_paths: List[Path], prompt: str, size: int = 512) -> None:
        self.image_paths = image_paths
        self.prompt      = prompt
        self.transform   = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor(),
            IMAGENET_NORM,
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img), self.prompt


# ──────────────────────────────────────────────────────────────────────────────
# Training function
# ──────────────────────────────────────────────────────────────────────────────

def train_one_subject(
    subject_id:   str,
    subject_dir:  str,
    image_names:  List[str],
    prompt:       str,
    output_path:  Path,
    *,
    model_id:     str,
    rank:         int,
    alpha:        float,
    num_epochs:   int,
    lr:           float,
    batch_size:   int,
    device:       str,
    dtype:        torch.dtype,
    seed:         int,
) -> None:
    """Train one LoRA adapter and save to output_path."""

    if output_path.exists():
        print(f"  [skip] {output_path.name} already exists.")
        return

    from models.diffusion_wrapper import DiffusionWrapper
    from models.adapter_utils import LoRAInjector

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # ── load model ──────────────────────────────────────────────────────────
    wrapper = DiffusionWrapper(model_id=model_id, device=device, dtype=dtype)

    # Freeze everything
    wrapper.unet.requires_grad_(False)
    wrapper.vae.requires_grad_(False)
    wrapper.text_encoder.requires_grad_(False)

    # ── inject LoRA ─────────────────────────────────────────────────────────
    lora_layers = LoRAInjector.inject(wrapper.unet, rank=rank, alpha=alpha)
    for layer in lora_layers.values():
        layer.to(device)

    params = LoRAInjector.get_parameters(lora_layers)
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)

    # ── dataset ──────────────────────────────────────────────────────────────
    subjects_dir = Path(subject_dir).parent          # data/subjects/
    image_paths  = [subjects_dir / subject_dir / n for n in image_names]
    dataset = SubjectDataset(image_paths, prompt)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # ── training loop ────────────────────────────────────────────────────────
    wrapper.unet.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for imgs, prompts in loader:
            imgs = imgs.to(device, dtype=dtype)

            # 1. Encode to latents
            with torch.no_grad():
                latents = wrapper.vae.encode(imgs).latent_dist.sample()
                latents = latents * wrapper._vae_scale

            # 2. Sample noise and timestep
            noise = torch.randn_like(latents)
            t = torch.randint(
                0, wrapper.scheduler.config.num_train_timesteps,
                (latents.shape[0],), device=device
            ).long()
            noisy = wrapper.scheduler.add_noise(latents, noise, t)

            # 3. Encode text
            with torch.no_grad():
                text_emb = wrapper.encode_text(list(prompts))

            # 4. Forward through UNet with LoRA fused temporarily
            optimizer.zero_grad()
            LoRAInjector.apply_to_unet(wrapper.unet, lora_layers, scale=1.0)

            noise_pred = wrapper.unet(
                noisy, t, encoder_hidden_states=text_emb
            ).sample

            LoRAInjector.remove_from_unet(wrapper.unet, lora_layers, scale=1.0)

            # 5. Diffusion loss
            loss = F.mse_loss(noise_pred.float(), noise.float())
            loss.backward()

            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        if (epoch + 1) % 20 == 0 or epoch == num_epochs - 1:
            avg = epoch_loss / len(loader)
            print(f"    epoch {epoch+1:>4d}/{num_epochs}  loss={avg:.5f}")

    # ── save ────────────────────────────────────────────────────────────────
    from safetensors.torch import save_file
    state = LoRAInjector.extract_state(lora_layers)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, output_path)
    print(f"  ✓ saved → {output_path}")

    # free GPU memory before next subject
    del wrapper, lora_layers, params, optimizer
    torch.cuda.empty_cache()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LoRA adapters for privacy experiments")
    p.add_argument("--data_dir",     type=str, default="./data")
    p.add_argument("--output_dir",   type=str, default="./adapters")
    p.add_argument("--model_id",     type=str, default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--rank",         type=int, default=16)
    p.add_argument("--alpha",        type=float, default=1.0)
    p.add_argument("--num_epochs",   type=int, default=100)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--batch_size",   type=int, default=1)
    p.add_argument("--num_subjects", type=int, default=None,
                   help="Limit to first N subjects (default: all)")
    p.add_argument("--subject_id",   type=str, default=None,
                   help="Train only this subject id")
    p.add_argument("--device",       type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",         type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    dtype = torch.float16 if args.device == "cuda" else torch.float32

    splits_file = Path(args.data_dir) / "splits.json"
    if not splits_file.exists():
        raise FileNotFoundError(f"splits.json not found at {splits_file}. Run data_preparation.py first.")
    splits = json.loads(splits_file.read_text())

    subject_ids = list(splits.keys())
    if args.subject_id is not None:
        subject_ids = [args.subject_id]
    elif args.num_subjects is not None:
        subject_ids = subject_ids[: args.num_subjects]

    print(f"Training LoRA rank={args.rank} for {len(subject_ids)} subjects …\n")

    output_dir = Path(args.output_dir)
    subjects_root = Path(args.data_dir) / "subjects"

    for sid in subject_ids:
        info = splits[sid]
        print(f"Subject {sid}  ({len(info['train'])} member images)  prompt='{info['class_prompt']}'")

        output_path = output_dir / f"lora_rank{args.rank}_subject{sid}.safetensors"

        train_one_subject(
            subject_id   = sid,
            subject_dir  = info["subject_dir"],
            image_names  = info["train"],
            prompt       = info["class_prompt"],
            output_path  = output_path,
            model_id     = args.model_id,
            rank         = args.rank,
            alpha        = args.alpha,
            num_epochs   = args.num_epochs,
            lr           = args.lr,
            batch_size   = args.batch_size,
            device       = args.device,
            dtype        = dtype,
            seed         = args.seed,
        )

    print("\n✓ All LoRA adapters trained.")


if __name__ == "__main__":
    main()

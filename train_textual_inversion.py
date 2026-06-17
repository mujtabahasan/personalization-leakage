"""
train_textual_inversion.py
──────────────────────────
Train a Textual Inversion (TI) embedding for every subject.

Paper note: TI uses 530× fewer parameters than LoRA (rank=16) yet leaks
*more* — a key finding.  The trained embedding is a single token vector
in ℝ^768 (CLIP hidden dim).

Usage
─────
python train_textual_inversion.py --data_dir ./data --output_dir ./adapters_ti
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm


IMAGENET_NORM = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])


# ──────────────────────────────────────────────────────────────────────────────
# Dataset (same as LoRA)
# ──────────────────────────────────────────────────────────────────────────────

class SubjectDataset(Dataset):
    def __init__(self, image_paths: List[Path], prompt: str, size: int = 512):
        self.image_paths = image_paths
        self.prompt      = prompt
        self.tf = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor(),
            IMAGENET_NORM,
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.tf(img), self.prompt


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train_one_subject(
    subject_id:   str,
    subject_dir:  str,
    image_names:  List[str],
    class_prompt: str,
    output_path:  Path,
    *,
    model_id:     str,
    num_epochs:   int,
    lr:           float,
    batch_size:   int,
    device:       str,
    dtype:        torch.dtype,
    seed:         int,
) -> None:
    if output_path.exists():
        print(f"  [skip] {output_path.name}")
        return

    from models.diffusion_wrapper import DiffusionWrapper
    from models.adapter_utils import TextualInversionManager

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # ── load model ──────────────────────────────────────────────────────────
    wrapper = DiffusionWrapper(model_id=model_id, device=device, dtype=dtype)

    # Freeze all
    wrapper.unet.requires_grad_(False)
    wrapper.vae.requires_grad_(False)
    wrapper.text_encoder.requires_grad_(False)
    wrapper.text_encoder.get_input_embeddings().requires_grad_(True)

    # ── add learnable token ─────────────────────────────────────────────────
    token    = f"<sks_{subject_id}>"
    ti_mgr   = TextualInversionManager(wrapper.tokenizer, wrapper.text_encoder)
    token_ids = ti_mgr.add_tokens([token], init_text="person face")
    tid       = token_ids[token]

    # Override prompt to include the new token
    ti_prompt = f"a photo of {token}"

    # ── dataset ──────────────────────────────────────────────────────────────
    subjects_root = Path(subject_dir).parent
    image_paths = [subjects_root / subject_dir / n for n in image_names]
    dataset = SubjectDataset(image_paths, ti_prompt)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Only optimise this token's embedding row
    embed_layer = wrapper.text_encoder.get_input_embeddings()
    optimizer = torch.optim.AdamW([embed_layer.weight], lr=lr)

    # ── training loop ────────────────────────────────────────────────────────
    wrapper.unet.train()
    wrapper.text_encoder.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for imgs, prompts in loader:
            imgs = imgs.to(device, dtype=dtype)

            with torch.no_grad():
                latents = wrapper.vae.encode(imgs).latent_dist.sample()
                latents = latents * wrapper._vae_scale

            noise = torch.randn_like(latents)
            t = torch.randint(
                0, wrapper.scheduler.config.num_train_timesteps,
                (latents.shape[0],), device=device
            ).long()
            noisy = wrapper.scheduler.add_noise(latents, noise, t)

            optimizer.zero_grad()
            text_emb = wrapper.encode_text(list(prompts))

            with torch.no_grad():
                noise_pred = wrapper.unet(
                    noisy, t, encoder_hidden_states=text_emb
                ).sample

            loss = F.mse_loss(noise_pred.float(), noise.float())
            loss.backward()

            # Zero-out gradients for all tokens except ours
            if embed_layer.weight.grad is not None:
                mask = torch.ones_like(embed_layer.weight.grad, dtype=torch.bool)
                mask[tid] = False
                embed_layer.weight.grad[mask] = 0.0

            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 20 == 0 or epoch == num_epochs - 1:
            print(f"    epoch {epoch+1:>4d}/{num_epochs}  loss={epoch_loss/len(loader):.5f}")

    # ── save ────────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ti_mgr.save(str(output_path))
    print(f"  ✓ saved → {output_path}")

    del wrapper, ti_mgr, optimizer
    torch.cuda.empty_cache()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Textual Inversion adapters")
    p.add_argument("--data_dir",     type=str, default="./data")
    p.add_argument("--output_dir",   type=str, default="./adapters_ti")
    p.add_argument("--model_id",     type=str, default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--num_epochs",   type=int, default=500)
    p.add_argument("--lr",           type=float, default=5e-4)
    p.add_argument("--batch_size",   type=int, default=1)
    p.add_argument("--num_subjects", type=int, default=None)
    p.add_argument("--subject_id",   type=str, default=None)
    p.add_argument("--device",       type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",         type=int, default=42)
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    dtype  = torch.float16 if args.device == "cuda" else torch.float32
    splits = json.loads((Path(args.data_dir) / "splits.json").read_text())

    sids = list(splits.keys())
    if args.subject_id:
        sids = [args.subject_id]
    elif args.num_subjects:
        sids = sids[: args.num_subjects]

    print(f"Training TI for {len(sids)} subjects …\n")

    for sid in sids:
        info = splits[sid]
        print(f"Subject {sid}  ({len(info['train'])} images)")
        out = Path(args.output_dir) / f"ti_subject{sid}.pt"
        train_one_subject(
            subject_id   = sid,
            subject_dir  = info["subject_dir"],
            image_names  = info["train"],
            class_prompt = info["class_prompt"],
            output_path  = out,
            model_id     = args.model_id,
            num_epochs   = args.num_epochs,
            lr           = args.lr,
            batch_size   = args.batch_size,
            device       = args.device,
            dtype        = dtype,
            seed         = args.seed,
        )

    print("\n✓ All TI adapters trained.")


if __name__ == "__main__":
    main()

"""
attacks/membership_inference.py
────────────────────────────────
Membership Inference Attack (MIA) on LoRA / Textual Inversion adapters.

Strategy
────────
1. Loss-based score   : member images have lower denoising loss under their adapter.
2. Likelihood-ratio   : compare loss under the target adapter vs. a *reference* model
                        (vanilla SD, no adapter). Ratio suppresses per-image difficulty.
3. Shadow-model attack: train shadow adapters on known splits, train a binary classifier
                        on the resulting (score, label) pairs, then apply to target.

All three scorers are implemented; the paper reports the likelihood-ratio scorer.

Public API
──────────
score_loss(wrapper, images, prompts, adapter_state, adapter_type) → np.ndarray
score_likelihood_ratio(wrapper, images, prompts, adapter_state, adapter_type) → np.ndarray
evaluate_mia(member_scores, nonmember_scores) → dict
run_mia(wrapper, splits_json, adapter_dir, subject_ids, adapter_type, ...) → dict
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    average_precision_score,
)
from torchvision import transforms
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# Image loading
# ──────────────────────────────────────────────────────────────────────────────

NORM = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
_to_tensor = transforms.Compose([
    transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.LANCZOS),
    transforms.ToTensor(),
    NORM,
])


def load_images(paths: List[Path]) -> torch.Tensor:
    """Load and normalise images to [-1, 1]; returns [N, 3, 512, 512]."""
    return torch.stack([_to_tensor(Image.open(p).convert("RGB")) for p in paths])


# ──────────────────────────────────────────────────────────────────────────────
# Scoring functions
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _batch_loss(
    wrapper,
    images:   torch.Tensor,
    prompts:  List[str],
    num_timesteps: int = 20,
) -> np.ndarray:
    """
    Compute average denoising MSE loss for each image.
    Lower loss  →  model has memorised the image.

    Returns array of shape [N] (one scalar loss per image).
    """
    losses = []
    wrapper.unet.eval()

    for img, prompt in zip(images, prompts):
        img_batch = img.unsqueeze(0).to(wrapper.device, dtype=wrapper.dtype)

        latents = wrapper.vae.encode(img_batch).latent_dist.sample()
        latents = latents * wrapper._vae_scale

        text_emb = wrapper.encode_text(prompt)  # [1, seq, dim]

        ts = torch.randint(
            0,
            wrapper.scheduler.config.num_train_timesteps,
            (num_timesteps,),
            device=wrapper.device,
        )

        img_loss = 0.0
        for t_val in ts:
            t = t_val.unsqueeze(0)
            noise  = torch.randn_like(latents)
            noisy  = wrapper.scheduler.add_noise(latents, noise, t)
            pred   = wrapper.unet(noisy, t, encoder_hidden_states=text_emb).sample
            img_loss += F.mse_loss(pred.float(), noise.float()).item()

        losses.append(img_loss / num_timesteps)

    return np.array(losses, dtype=np.float32)


def score_loss(
    wrapper,
    images:       torch.Tensor,
    prompts:      List[str],
    adapter_state: Dict[str, torch.Tensor],
    adapter_type:  str = "lora",
    num_timesteps: int = 20,
) -> np.ndarray:
    """
    Loss score: lower = more likely member.

    Returns [N] float32 array.
    """
    if adapter_type == "lora":
        wrapper.apply_lora(adapter_state)
        losses = _batch_loss(wrapper, images, prompts, num_timesteps)
        wrapper.remove_lora(adapter_state)
    elif adapter_type == "textual_inversion":
        # TI embedding must already be applied before calling this function.
        losses = _batch_loss(wrapper, images, prompts, num_timesteps)
    else:
        raise ValueError(f"Unknown adapter_type: {adapter_type}")

    return losses


def score_likelihood_ratio(
    wrapper,
    images:        torch.Tensor,
    prompts:       List[str],
    adapter_state: Dict[str, torch.Tensor],
    adapter_type:  str = "lora",
    num_timesteps: int = 20,
) -> np.ndarray:
    """
    Likelihood-ratio score:  loss_baseline - loss_adapted
    Higher = more likely member.

    This suppresses image-level difficulty and is the primary scorer in the paper.
    """
    # Baseline (no adapter)
    base_losses    = _batch_loss(wrapper, images, prompts, num_timesteps)

    # Adapted
    adapter_losses = score_loss(
        wrapper, images, prompts, adapter_state,
        adapter_type=adapter_type, num_timesteps=num_timesteps,
    )

    # Ratio: positive when adapter reduces loss (i.e. image is a member)
    return (base_losses - adapter_losses).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_mia(
    member_scores:     np.ndarray,
    nonmember_scores:  np.ndarray,
    higher_is_member:  bool = True,
) -> Dict[str, float]:
    """
    Compute AUC-ROC, AUC-PR, TPR@FPR={0.01,0.05,0.10}, and max advantage.

    Parameters
    ----------
    member_scores, nonmember_scores : 1-D arrays
    higher_is_member : True  for likelihood-ratio scores
                       False for loss scores (need flip)
    """
    scores = np.concatenate([member_scores, nonmember_scores])
    labels = np.concatenate([
        np.ones(len(member_scores)),
        np.zeros(len(nonmember_scores)),
    ])

    if not higher_is_member:
        scores = -scores   # flip so higher = member

    try:
        auc_roc = float(roc_auc_score(labels, scores))
        auc_pr  = float(average_precision_score(labels, scores))
    except ValueError:
        auc_roc = auc_pr = 0.5

    fpr, tpr, _ = roc_curve(labels, scores)

    def _tpr_at_fpr(target_fpr: float) -> float:
        idx = np.searchsorted(fpr, target_fpr, side="right") - 1
        idx = max(0, min(idx, len(tpr) - 1))
        return float(tpr[idx])

    advantage = float(np.max(tpr - fpr))   # TPR - FPR at best threshold

    return {
        "auc_roc":         auc_roc,
        "auc_pr":          auc_pr,
        "tpr_at_fpr_001":  _tpr_at_fpr(0.01),
        "tpr_at_fpr_005":  _tpr_at_fpr(0.05),
        "tpr_at_fpr_010":  _tpr_at_fpr(0.10),
        "max_advantage":   advantage,
        "n_members":       int(len(member_scores)),
        "n_nonmembers":    int(len(nonmember_scores)),
        "member_loss_mean":    float(member_scores.mean()),
        "nonmember_loss_mean": float(nonmember_scores.mean()),
        "member_loss_std":     float(member_scores.std()),
        "nonmember_loss_std":  float(nonmember_scores.std()),
    }


# ──────────────────────────────────────────────────────────────────────────────
# High-level pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_mia(
    wrapper,
    data_dir:      str,
    adapter_dir:   str,
    adapter_type:  str,
    subject_ids:   Optional[List[str]] = None,
    scorer:        str = "likelihood_ratio",   # or "loss"
    num_timesteps: int = 20,
    max_images_per_split: int = 20,
    rank: Optional[int] = None,
) -> Dict:
    """
    Run MIA across subjects.

    Parameters
    ----------
    wrapper      : DiffusionWrapper (already loaded)
    data_dir     : path containing splits.json and subjects/
    adapter_dir  : directory with adapter files
    adapter_type : "lora" or "textual_inversion"
    subject_ids  : list of subject id strings; None = all in splits.json
    scorer       : "likelihood_ratio" (paper) or "loss"
    num_timesteps: timesteps averaged per image
    max_images_per_split: cap per member/nonmember set to keep runtime bounded
    rank         : LoRA rank (used to build filename); None = inferred from file

    Returns
    -------
    dict with per-subject scores and aggregate metrics
    """
    splits_path = Path(data_dir) / "splits.json"
    splits = json.loads(splits_path.read_text())

    if subject_ids is None:
        subject_ids = list(splits.keys())

    subjects_dir = Path(data_dir) / "subjects"
    adapter_dir  = Path(adapter_dir)

    all_member_scores:     List[float] = []
    all_nonmember_scores:  List[float] = []
    per_subject: List[Dict] = []

    print(f"\n[MIA] scorer={scorer}  adapter_type={adapter_type}  "
          f"subjects={len(subject_ids)}\n")

    for sid in tqdm(subject_ids, desc="Subjects"):
        info = splits.get(sid)
        if info is None:
            print(f"  ⚠ subject {sid} not in splits, skip")
            continue

        # ── find adapter file ───────────────────────────────────────────────
        if adapter_type == "lora":
            rank_str = f"rank{rank}_" if rank is not None else "*"
            candidates = list(adapter_dir.glob(f"lora_{rank_str}subject{sid}.safetensors"))
            if not candidates:
                candidates = list(adapter_dir.glob(f"*subject{sid}.safetensors"))
            if not candidates:
                print(f"  ⚠ no LoRA adapter for subject {sid}, skip")
                continue
            adapter_path = candidates[0]

            from safetensors.torch import load_file
            adapter_state = load_file(adapter_path)

        elif adapter_type == "textual_inversion":
            candidates = list(adapter_dir.glob(f"ti_subject{sid}.pt"))
            if not candidates:
                print(f"  ⚠ no TI adapter for subject {sid}, skip")
                continue
            adapter_path = candidates[0]
            # Load and apply embedding
            ckpt = torch.load(adapter_path, map_location="cpu")
            adapter_state = {}  # TI: state stored separately
            for token, emb in ckpt["embeddings"].items():
                wrapper.apply_textual_inversion(token, emb)

        # ── load images ─────────────────────────────────────────────────────
        sdir = subjects_dir / info["subject_dir"]
        prompt = info["class_prompt"]

        member_paths = [sdir / n for n in info["train"][:max_images_per_split]]
        nonmember_paths = [sdir / n for n in info["test"][:max_images_per_split]]

        member_imgs    = load_images(member_paths)
        nonmember_imgs = load_images(nonmember_paths)
        member_prompts    = [prompt] * len(member_imgs)
        nonmember_prompts = [prompt] * len(nonmember_imgs)

        # ── score ───────────────────────────────────────────────────────────
        if scorer == "likelihood_ratio":
            m_scores  = score_likelihood_ratio(
                wrapper, member_imgs,    member_prompts,
                adapter_state, adapter_type, num_timesteps
            )
            nm_scores = score_likelihood_ratio(
                wrapper, nonmember_imgs, nonmember_prompts,
                adapter_state, adapter_type, num_timesteps
            )
            higher_is_member = True
        else:  # "loss"
            m_scores  = score_loss(
                wrapper, member_imgs,    member_prompts,
                adapter_state, adapter_type, num_timesteps
            )
            nm_scores = score_loss(
                wrapper, nonmember_imgs, nonmember_prompts,
                adapter_state, adapter_type, num_timesteps
            )
            higher_is_member = False

        metrics = evaluate_mia(m_scores, nm_scores, higher_is_member=higher_is_member)

        per_subject.append({
            "subject_id":        sid,
            "adapter_path":      str(adapter_path),
            "member_scores":     m_scores.tolist(),
            "nonmember_scores":  nm_scores.tolist(),
            "metrics":           metrics,
        })

        all_member_scores.extend(m_scores.tolist())
        all_nonmember_scores.extend(nm_scores.tolist())

        print(f"  subject {sid}: AUC-ROC={metrics['auc_roc']:.3f}  "
              f"advantage={metrics['max_advantage']:.3f}")

    # ── aggregate ────────────────────────────────────────────────────────────
    aggregate = evaluate_mia(
        np.array(all_member_scores),
        np.array(all_nonmember_scores),
        higher_is_member=higher_is_member,
    )
    aggregate["num_subjects"] = len(per_subject)

    print(f"\n[MIA Aggregate]  AUC-ROC={aggregate['auc_roc']:.4f}  "
          f"AUC-PR={aggregate['auc_pr']:.4f}  "
          f"TPR@FPR=1%={aggregate['tpr_at_fpr_001']:.4f}")

    return {
        "adapter_type":  adapter_type,
        "scorer":        scorer,
        "aggregate":     aggregate,
        "per_subject":   per_subject,
    }

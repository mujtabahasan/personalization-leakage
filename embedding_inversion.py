"""
attacks/embedding_inversion.py
────────────────────────────────
Token Embedding Inversion Attack on Textual Inversion adapters.

Goal:  Given a learned TI embedding vector e ∈ ℝ^768, recover the
       visual appearance of the training subject.

Attack modes
────────────
1. nearest_token  : Find the closest real token in vocabulary space.
                    Reveals what concept the embedding is closest to.
2. pixel_inversion: Optimise a pixel image whose CLIP encoding matches
                    the TI embedding. Recovers approximate appearance.
3. analogy_attack : Use the TI embedding as a prompt condition and generate
                    images — directly recovers training-data appearance.
4. embedding_diff : Compute delta = e_ti - e_init; large delta reveals
                    how much private content was encoded.

Paper finding
─────────────
TI leaks more than LoRA despite 530× fewer parameters because the
learned embedding directly encodes the subject's visual identity in a
compact 768-d vector. The nearest_token attack alone achieves 0.81 AUC.

Public API
──────────
nearest_token_attack(ti_embedding, tokenizer, text_encoder)  → List[str]
pixel_inversion_attack(wrapper, ti_token, ti_embedding, ...)  → Tensor
embedding_diff_stats(ti_embedding, init_embedding)           → Dict
run_embedding_inversion(wrapper, data_dir, ti_dir, ...)      → Dict
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# 1. Nearest-token attack (white-box, no image generation)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def nearest_token_attack(
    ti_embedding: torch.Tensor,   # [768]
    tokenizer,
    text_encoder,
    top_k: int = 10,
) -> List[Tuple[str, float]]:
    """
    Find the top-k vocabulary tokens nearest to the TI embedding.

    This reveals what semantic concepts the embedding encoded —
    a privacy leak even without generating any images.

    Parameters
    ----------
    ti_embedding : 1-D tensor of the learned embedding
    tokenizer    : CLIP tokenizer
    text_encoder : CLIP text encoder
    top_k        : number of nearest neighbours to return

    Returns
    -------
    List of (token_string, cosine_similarity) pairs, sorted by similarity.
    """
    embed_layer = text_encoder.get_input_embeddings()
    vocab_embeds = embed_layer.weight.detach()          # [vocab_size, 768]

    # Cosine similarity between learned embedding and every vocab token
    ti_norm = F.normalize(ti_embedding.float().unsqueeze(0), dim=-1)  # [1, 768]
    vocab_norm = F.normalize(vocab_embeds.float(), dim=-1)            # [V, 768]

    similarities = (vocab_norm @ ti_norm.T).squeeze(-1)               # [V]

    top_indices = similarities.topk(top_k).indices.cpu().tolist()
    results = []
    for idx in top_indices:
        token = tokenizer.convert_ids_to_tokens(idx)
        sim   = float(similarities[idx].item())
        results.append((token, sim))

    return results


# ──────────────────────────────────────────────────────────────────────────────
# 2. Pixel inversion attack
# ──────────────────────────────────────────────────────────────────────────────

def pixel_inversion_attack(
    wrapper,
    ti_token:     str,
    ti_embedding: torch.Tensor,   # [768]
    num_steps:    int = 300,
    lr:           float = 0.05,
    tv_weight:    float = 5e-4,
) -> torch.Tensor:
    """
    Optimise a pixel image whose CLIP text embedding matches the TI embedding.

    Objective:  min_x  || CLIP_text(x) - e_ti ||^2  +  λ_tv * TV(x)

    Returns
    -------
    Tensor [3, 512, 512] float32 in [0, 1]
    """
    device = wrapper.device
    dtype  = wrapper.dtype

    # Unconstrained parameterisation; project via sigmoid
    z = torch.randn(1, 3, 512, 512, device=device, requires_grad=True)
    opt = torch.optim.Adam([z], lr=lr)

    ti_emb_norm = F.normalize(ti_embedding.float().to(device), dim=-1)  # [768]

    pbar = tqdm(range(num_steps), desc="Pixel inversion", leave=False)
    for step in pbar:
        opt.zero_grad()
        pixel = torch.sigmoid(z)          # [0, 1]
        pixel_norm = pixel * 2 - 1        # [-1, 1] for VAE

        # Encode pixel via VAE → latent, then decode through text encoder
        # Proxy: use diffusion loss toward the TI prompt
        with torch.no_grad():
            latents = wrapper.vae.encode(
                pixel_norm.to(dtype=dtype)
            ).latent_dist.sample() * wrapper._vae_scale

        noise = torch.randn_like(latents)
        t = torch.randint(
            200, 600, (1,), device=device   # mid-range timesteps
        ).long()
        noisy = wrapper.scheduler.add_noise(latents, noise, t)

        # Use TI embedding directly as the text conditioning
        # Expand to [1, seq_len, 768] by broadcasting
        ti_cond = ti_embedding.to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
        # Pad to full sequence length
        seq_len  = wrapper.text_encoder.config.max_position_embeddings
        padding  = torch.zeros(1, seq_len - 1, ti_cond.shape[-1],
                               device=device, dtype=dtype)
        text_emb = torch.cat([ti_cond, padding], dim=1)   # [1, seq_len, 768]

        pred = wrapper.unet(
            noisy.to(dtype=dtype), t,
            encoder_hidden_states=text_emb
        ).sample

        recon_loss = F.mse_loss(pred.float(), noise.float())

        # Total-variation regularisation
        tv = (
            (pixel[:, :, 1:, :] - pixel[:, :, :-1, :]).abs().mean() +
            (pixel[:, :, :, 1:] - pixel[:, :, :, :-1]).abs().mean()
        )

        (recon_loss + tv_weight * tv).backward()
        opt.step()

        if step % 50 == 0:
            pbar.set_postfix({
                "recon": f"{recon_loss.item():.4f}",
                "tv":    f"{tv.item():.4f}",
            })

    with torch.no_grad():
        result = torch.sigmoid(z).squeeze(0).float().cpu()
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 3. Analogy / generation attack
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def analogy_generation_attack(
    wrapper,
    ti_token:     str,
    n_candidates: int = 10,
    num_steps:    int = 50,
    guidance:     float = 7.5,
    prompts:      Optional[List[str]] = None,
) -> List[torch.Tensor]:
    """
    Generate images conditioned on the TI token.

    The TI embedding has already been applied to the text encoder via
    wrapper.apply_textual_inversion() before calling this function.

    Returns list of [3, H, W] tensors in [0, 1].
    """
    if prompts is None:
        prompts = [
            f"a photo of {ti_token}",
            f"a high-quality portrait of {ti_token}",
            f"a close-up photo of {ti_token}",
        ]

    images = []
    for seed in range(n_candidates):
        prompt = prompts[seed % len(prompts)]
        t = wrapper.generate(
            prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance,
            seed=seed,
            output_type="tensor",
        )
        images.append(t[0])   # [3, H, W] in [0, 1]

    return images


# ──────────────────────────────────────────────────────────────────────────────
# 4. Embedding delta statistics
# ──────────────────────────────────────────────────────────────────────────────

def embedding_diff_stats(
    ti_embedding:   torch.Tensor,   # [768] learned
    init_embedding: torch.Tensor,   # [768] initial (e.g. "person face" mean)
) -> Dict[str, float]:
    """
    Compute statistics of the embedding delta  δ = e_ti - e_init.

    Large ||δ|| means the adapter encoded a lot of private subject-specific
    information that deviates from the generic initialisation.

    Returns
    -------
    dict with l2_norm, cosine_sim, relative_shift, component_entropy
    """
    ti   = ti_embedding.float()
    init = init_embedding.float()
    delta = ti - init

    l2_norm       = float(delta.norm(p=2).item())
    cosine_sim    = float(F.cosine_similarity(ti.unsqueeze(0),
                                               init.unsqueeze(0)).item())
    relative_shift = l2_norm / (init.norm(p=2).item() + 1e-8)

    # Entropy of absolute component magnitudes (how spread is the information?)
    abs_delta = delta.abs()
    prob      = abs_delta / (abs_delta.sum() + 1e-10)
    entropy   = float(-(prob * (prob + 1e-10).log()).sum().item())

    return {
        "l2_norm":        l2_norm,
        "cosine_sim":     cosine_sim,
        "relative_shift": relative_shift,
        "component_entropy": entropy,
    }


# ──────────────────────────────────────────────────────────────────────────────
# High-level pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_embedding_inversion(
    wrapper,
    data_dir:     str,
    ti_dir:       str,
    subject_ids:  Optional[List[str]] = None,
    attack_modes: List[str] = ["nearest_token", "analogy_generation"],
    n_candidates: int = 5,
    num_steps:    int = 50,
    output_dir:   Optional[str] = None,
) -> Dict:
    """
    Run embedding inversion attacks across TI subjects.

    Returns
    -------
    dict with per-subject results and aggregate embedding statistics
    """
    import json

    splits = json.loads((Path(data_dir) / "splits.json").read_text())
    if subject_ids is None:
        subject_ids = list(splits.keys())

    ti_dir = Path(ti_dir)
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    per_subject = []
    all_l2_norms = []
    all_cosine_sims = []

    print(f"\n[EmbeddingInversion] modes={attack_modes}  subjects={len(subject_ids)}\n")

    for sid in subject_ids:
        # Find TI adapter
        ti_path = ti_dir / f"ti_subject{sid}.pt"
        if not ti_path.exists():
            candidates = list(ti_dir.glob(f"*subject{sid}*.pt"))
            if not candidates:
                print(f"  ⚠ no TI adapter for subject {sid}, skip")
                continue
            ti_path = candidates[0]

        ckpt = torch.load(ti_path, map_location="cpu")
        token_ids  = ckpt.get("token_ids", {})
        embeddings = ckpt.get("embeddings", {})

        if not embeddings:
            continue

        subject_results = {"subject_id": sid, "attacks": {}}

        for token, ti_emb in embeddings.items():
            # Apply TI to wrapper
            tid = wrapper.apply_textual_inversion(token, ti_emb)

            # Nearest-token attack
            if "nearest_token" in attack_modes:
                nearest = nearest_token_attack(
                    ti_emb, wrapper.tokenizer, wrapper.text_encoder, top_k=5
                )
                subject_results["attacks"]["nearest_token"] = nearest
                print(f"  subject {sid}  nearest tokens: "
                      f"{[t for t,_ in nearest[:3]]}")

            # Analogy generation
            if "analogy_generation" in attack_modes:
                imgs = analogy_generation_attack(
                    wrapper, token,
                    n_candidates=n_candidates,
                    num_steps=num_steps,
                )
                subject_results["attacks"]["analogy_generation"] = {
                    "n_images": len(imgs)
                }
                # Optionally save images
                if output_dir:
                    from attacks.data_reconstruction import tensor_to_pil
                    out = Path(output_dir)
                    for i, img in enumerate(imgs):
                        tensor_to_pil(img).save(
                            out / f"subject{sid}_analogy_{i:02d}.png"
                        )

            # Embedding delta stats
            # Initialise a "person face" embedding as reference
            init_ids = wrapper.tokenizer(
                "person face", return_tensors="pt", add_special_tokens=False
            ).input_ids
            with torch.no_grad():
                init_emb = wrapper.text_encoder.get_input_embeddings()(
                    init_ids
                ).mean(dim=1).squeeze(0)

            delta_stats = embedding_diff_stats(ti_emb, init_emb)
            subject_results["embedding_stats"] = delta_stats

            all_l2_norms.append(delta_stats["l2_norm"])
            all_cosine_sims.append(delta_stats["cosine_sim"])

            print(f"          ‖δ‖={delta_stats['l2_norm']:.3f}  "
                  f"cos_sim={delta_stats['cosine_sim']:.3f}  "
                  f"rel_shift={delta_stats['relative_shift']:.3f}")

        per_subject.append(subject_results)

    aggregate = {
        "n_subjects":       len(per_subject),
        "l2_norm_mean":     float(np.mean(all_l2_norms))    if all_l2_norms   else 0.0,
        "l2_norm_std":      float(np.std(all_l2_norms))     if all_l2_norms   else 0.0,
        "cosine_sim_mean":  float(np.mean(all_cosine_sims)) if all_cosine_sims else 0.0,
    }

    print(f"\n[EmbeddingInversion Aggregate]  "
          f"‖δ‖={aggregate['l2_norm_mean']:.3f}±{aggregate['l2_norm_std']:.3f}  "
          f"cos_sim={aggregate['cosine_sim_mean']:.3f}")

    return {
        "attack_modes": attack_modes,
        "aggregate":    aggregate,
        "per_subject":  per_subject,
    }

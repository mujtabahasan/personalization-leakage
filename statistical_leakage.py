"""
attacks/statistical_leakage.py
────────────────────────────────
Statistical leakage metrics: weight-space analysis that quantifies how much
private information is encoded in the adapter weights themselves.

Metrics implemented
────────────────────
1. Fisher-information proxy  – measures sensitivity of the loss surface to each
                               weight; high Fisher → more memorisation.
2. Weight-norm               – L2 norm of the adapter delta (LoRA effective weight).
3. Gradient alignment        – cosine similarity between adapter gradients on member
                               vs. non-member images.
4. Spectral leakage          – singular values of LoRA A and B matrices reveal
                               effective rank and information capacity.
5. Spearman correlation      – correlates rank vs. leakage (ρ=0.83 in paper).

These metrics are "white-box" (adapter weights required) and complement the
black-box MIA / reconstruction attacks.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# 1. Fisher information proxy
# ──────────────────────────────────────────────────────────────────────────────

def fisher_information_proxy(
    wrapper,
    lora_state:    Dict[str, torch.Tensor],
    images:        torch.Tensor,   # [N, 3, H, W] in [-1, 1]
    prompts:       List[str],
    num_timesteps: int = 5,
) -> Dict[str, float]:
    """
    Estimate the diagonal Fisher information for the LoRA parameters.

    Fisher_i ≈ E[(∂L/∂θ_i)^2]

    Higher mean Fisher → the adapter has memorised specific image features
    and is sensitive to them.

    Returns
    -------
    dict with "fisher_mean", "fisher_max", "fisher_sum"
    """
    from models.adapter_utils import LoRAInjector

    # Re-inject LoRA as trainable modules so we can compute gradients
    lora_layers = LoRAInjector.inject(wrapper.unet, rank=_infer_rank(lora_state))
    LoRAInjector.load_state(lora_layers, lora_state)
    for layer in lora_layers.values():
        layer.to(wrapper.device)

    params = LoRAInjector.get_parameters(lora_layers)
    grad_sq_sum = [torch.zeros_like(p) for p in params]

    wrapper.unet.train()

    for img, prompt in tqdm(zip(images, prompts),
                             total=len(images), desc="Fisher", leave=False):
        img_batch = img.unsqueeze(0).to(wrapper.device, dtype=wrapper.dtype)

        with torch.no_grad():
            latents  = wrapper.vae.encode(img_batch).latent_dist.sample()
            latents  = latents * wrapper._vae_scale
            text_emb = wrapper.encode_text(prompt)

        for _ in range(num_timesteps):
            noise = torch.randn_like(latents)
            t = torch.randint(
                0, wrapper.scheduler.config.num_train_timesteps,
                (1,), device=wrapper.device
            ).long()
            noisy = wrapper.scheduler.add_noise(latents, noise, t)

            # Apply LoRA temporarily for forward
            LoRAInjector.apply_to_unet(wrapper.unet, lora_layers)
            pred = wrapper.unet(noisy, t, encoder_hidden_states=text_emb).sample
            LoRAInjector.remove_from_unet(wrapper.unet, lora_layers)

            loss = F.mse_loss(pred.float(), noise.float())
            loss.backward()

            with torch.no_grad():
                for gs, p in zip(grad_sq_sum, params):
                    if p.grad is not None:
                        gs.add_(p.grad.float() ** 2)
                        p.grad.zero_()

    # Average over images and timesteps
    n = len(images) * num_timesteps
    fisher_vals = torch.cat([g.view(-1) / n for g in grad_sq_sum])

    # Clean up
    wrapper.unet.eval()
    for layer in lora_layers.values():
        del layer

    return {
        "fisher_mean": float(fisher_vals.mean().item()),
        "fisher_max":  float(fisher_vals.max().item()),
        "fisher_sum":  float(fisher_vals.sum().item()),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 2. Weight norm
# ──────────────────────────────────────────────────────────────────────────────

def weight_norm_stats(lora_state: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    L2-norm statistics of the effective LoRA weight matrices (A @ B).
    Larger norms indicate larger parameter shifts, correlating with leakage.
    """
    norms: List[float] = []
    names = {k.rsplit(".", 1)[0] for k in lora_state if k.endswith(".lora_A")}

    for name in names:
        A = lora_state[f"{name}.lora_A"].float()
        B = lora_state[f"{name}.lora_B"].float()
        W = A @ B
        norms.append(W.norm(p=2).item())

    return {
        "weight_norm_mean":   float(np.mean(norms)),
        "weight_norm_max":    float(np.max(norms)),
        "weight_norm_total":  float(np.sum(norms)),
        "num_lora_layers":    len(norms),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 3. Gradient alignment
# ──────────────────────────────────────────────────────────────────────────────

def gradient_alignment(
    wrapper,
    lora_state:    Dict[str, torch.Tensor],
    member_images:    torch.Tensor,   # [N, 3, H, W]
    nonmember_images: torch.Tensor,
    prompts:          List[str],
    num_timesteps: int = 3,
) -> float:
    """
    Cosine similarity between the average gradient direction on member images
    vs. non-member images.

    Interpretation: values close to 1 → the adapter is optimised specifically
    for member images (high leakage). Values near 0 → generic adaptation.
    """
    from models.adapter_utils import LoRAInjector

    rank = _infer_rank(lora_state)
    lora_layers = LoRAInjector.inject(wrapper.unet, rank=rank)
    LoRAInjector.load_state(lora_layers, lora_state)
    for layer in lora_layers.values():
        layer.to(wrapper.device)

    params = LoRAInjector.get_parameters(lora_layers)
    wrapper.unet.train()

    def _avg_gradient(images: torch.Tensor) -> torch.Tensor:
        total = [torch.zeros_like(p) for p in params]
        for img, prompt in zip(images, prompts):
            img_batch = img.unsqueeze(0).to(wrapper.device, dtype=wrapper.dtype)
            with torch.no_grad():
                latents = wrapper.vae.encode(img_batch).latent_dist.sample()
                latents = latents * wrapper._vae_scale
                text_emb = wrapper.encode_text(prompt)

            for _ in range(num_timesteps):
                noise = torch.randn_like(latents)
                t = torch.randint(
                    0, wrapper.scheduler.config.num_train_timesteps,
                    (1,), device=wrapper.device
                ).long()
                noisy = wrapper.scheduler.add_noise(latents, noise, t)

                LoRAInjector.apply_to_unet(wrapper.unet, lora_layers)
                pred = wrapper.unet(noisy, t, encoder_hidden_states=text_emb).sample
                LoRAInjector.remove_from_unet(wrapper.unet, lora_layers)

                F.mse_loss(pred.float(), noise.float()).backward()

                with torch.no_grad():
                    for tot, p in zip(total, params):
                        if p.grad is not None:
                            tot.add_(p.grad.float())
                            p.grad.zero_()

        n = len(images) * num_timesteps
        return torch.cat([g.view(-1) / n for g in total])

    g_member    = _avg_gradient(member_images)
    g_nonmember = _avg_gradient(nonmember_images)

    cos_sim = F.cosine_similarity(
        g_member.unsqueeze(0), g_nonmember.unsqueeze(0)
    ).item()

    wrapper.unet.eval()
    return float(cos_sim)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Spectral leakage
# ──────────────────────────────────────────────────────────────────────────────

def spectral_analysis(lora_state: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    Analyse singular values of each LoRA pair's effective weight matrix.

    Metrics:
    • stable_rank  = ||A @ B||_F^2 / sigma_max^2  (measure of rank usage)
    • sv_entropy   = entropy of normalised singular values (spread of information)
    • sv_max_mean  = mean dominant singular value across layers
    """
    stable_ranks, sv_entropies, sv_maxes = [], [], []

    names = {k.rsplit(".", 1)[0] for k in lora_state if k.endswith(".lora_A")}

    for name in names:
        A = lora_state[f"{name}.lora_A"].float()
        B = lora_state[f"{name}.lora_B"].float()
        W = A @ B   # [in, out]

        try:
            _, S, _ = torch.linalg.svd(W, full_matrices=False)
        except RuntimeError:
            S = torch.linalg.svdvals(W)

        frob_sq   = (W ** 2).sum().item()
        sigma_max = S[0].item()

        if sigma_max > 1e-10:
            stable_ranks.append(frob_sq / (sigma_max ** 2))
        else:
            stable_ranks.append(0.0)

        S_norm = S / (S.sum() + 1e-10)
        entropy = -(S_norm * (S_norm + 1e-10).log()).sum().item()
        sv_entropies.append(entropy)
        sv_maxes.append(sigma_max)

    return {
        "stable_rank_mean":   float(np.mean(stable_ranks)),
        "sv_entropy_mean":    float(np.mean(sv_entropies)),
        "sv_max_mean":        float(np.mean(sv_maxes)),
        "num_layers_analysed": len(stable_ranks),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 5. Rank–leakage Spearman correlation (Table 3 in paper, ρ = 0.83)
# ──────────────────────────────────────────────────────────────────────────────

def rank_leakage_spearman(
    ranks:         List[int],
    leakage_values: List[float],   # e.g. AUC-ROC or LPIPS improvement
) -> Dict[str, float]:
    """
    Spearman rank correlation between LoRA rank and a leakage metric.

    Paper finding: ρ = 0.83 between rank and reconstruction leakage (LPIPS).

    Parameters
    ----------
    ranks           : list of LoRA ranks tested (e.g. [4, 8, 16, 32, 64])
    leakage_values  : corresponding leakage scalar per rank

    Returns
    -------
    dict with "spearman_rho" and "p_value"
    """
    if len(ranks) < 3:
        return {"spearman_rho": float("nan"), "p_value": float("nan")}

    rho, pval = spearmanr(ranks, leakage_values)
    return {"spearman_rho": float(rho), "p_value": float(pval)}


# ──────────────────────────────────────────────────────────────────────────────
# Aggregate all statistical metrics
# ──────────────────────────────────────────────────────────────────────────────

def full_statistical_analysis(
    wrapper,
    lora_state:       Dict[str, torch.Tensor],
    member_images:    torch.Tensor,
    nonmember_images: torch.Tensor,
    prompts:          List[str],
    compute_fisher:   bool = True,
    compute_gradient: bool = True,
) -> Dict[str, float]:
    """
    Run all statistical leakage metrics and return a merged dict.
    Set compute_fisher=False and compute_gradient=False for a fast estimate.
    """
    results: Dict[str, float] = {}

    print("[StatLeakage] Weight norms …")
    results.update(weight_norm_stats(lora_state))

    print("[StatLeakage] Spectral analysis …")
    results.update(spectral_analysis(lora_state))

    if compute_fisher:
        print("[StatLeakage] Fisher information (member images) …")
        results.update(fisher_information_proxy(
            wrapper, lora_state, member_images, prompts[:len(member_images)]
        ))

    if compute_gradient:
        print("[StatLeakage] Gradient alignment …")
        cos = gradient_alignment(
            wrapper, lora_state,
            member_images[:5], nonmember_images[:5],
            prompts
        )
        results["gradient_alignment"] = cos

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Internal utility
# ──────────────────────────────────────────────────────────────────────────────

def _infer_rank(lora_state: Dict[str, torch.Tensor]) -> int:
    """Infer LoRA rank from first A matrix in the state dict."""
    for k, v in lora_state.items():
        if k.endswith(".lora_A"):
            return v.shape[1]   # [in_features, rank]
    return 4   # fallback

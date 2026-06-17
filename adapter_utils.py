"""
models/adapter_utils.py
───────────────────────
Low-rank adaptation (LoRA) and Textual Inversion utilities used by the
training scripts and experiments.

Key classes
───────────
LoRALayer          – single Low-Rank Adaptation module (A, B matrices)
LoRAInjector       – injects LoRA into a UNet's attention projections
TextualInversionManager – manages learnable token embeddings

Key functions
─────────────
lora_param_count   – count LoRA parameters given rank and model
extract_lora_state – convert injected LoRA layers → safetensors-ready dict
load_lora_state    – load dict back into LoRA layers
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# LoRA layer
# ──────────────────────────────────────────────────────────────────────────────

class LoRALayer(nn.Module):
    """
    A single LoRA adaptation: output = x @ W  +  x @ A @ B * scaling

    Parameters
    ----------
    in_features  : int
    out_features : int
    rank         : int  – LoRA rank r
    alpha        : float – LoRA alpha; scaling = alpha / rank
    dropout      : float – dropout on the LoRA path
    """

    def __init__(
        self,
        in_features:  int,
        out_features: int,
        rank:    int   = 4,
        alpha:   float = 1.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert rank > 0, "rank must be positive"

        self.rank    = rank
        self.alpha   = alpha
        self.scaling = alpha / rank

        # A: [in_features, rank]   initialised Kaiming uniform (non-zero)
        # B: [rank, out_features]  initialised zeros → delta = 0 at step 0
        self.lora_A = nn.Parameter(torch.empty(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

    # ── forward (additive delta only — base linear called separately) ─────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the LoRA delta for input x.

        Shape: x [B, *, in_features]  →  out [B, *, out_features]
        """
        return self.dropout(x @ self.lora_A @ self.lora_B) * self.scaling

    def effective_weight(self) -> torch.Tensor:
        """Materialise the rank-r weight matrix:  A @ B * scaling."""
        return self.lora_A @ self.lora_B * self.scaling

    def extra_repr(self) -> str:
        return (
            f"rank={self.rank}, alpha={self.alpha}, "
            f"scaling={self.scaling:.4f}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# LoRA injector
# ──────────────────────────────────────────────────────────────────────────────

# Default set of attention projection names to patch in SD UNet
_DEFAULT_TARGET_MODULES: Tuple[str, ...] = (
    "to_q", "to_k", "to_v", "to_out.0",
)


class LoRAInjector:
    """
    Utility class that injects LoRA layers alongside UNet attention projections.

    The injected layers live in a side-dict `lora_layers` keyed by the
    dotted parameter path (e.g. "down_blocks.0.attentions.0.…to_q").
    They are NOT attached to the UNet module tree, so the UNet's
    named_parameters() is unchanged — the LoRA deltas are applied
    manually in the training loop (see train_lora_adapters.py).
    """

    @staticmethod
    def inject(
        unet: nn.Module,
        rank:  int   = 4,
        alpha: float = 1.0,
        target_modules: Tuple[str, ...] = _DEFAULT_TARGET_MODULES,
        dropout: float = 0.0,
    ) -> Dict[str, LoRALayer]:
        """
        Create LoRA layers for every matching Linear in the UNet.

        Returns
        -------
        dict  {dotted_module_path: LoRALayer}
        """
        lora_layers: Dict[str, LoRALayer] = {}

        for name, module in unet.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if not any(name.endswith(t) for t in target_modules):
                continue

            layer = LoRALayer(
                in_features  = module.in_features,
                out_features = module.out_features,
                rank    = rank,
                alpha   = alpha,
                dropout = dropout,
            )
            lora_layers[name] = layer

        print(
            f"[LoRAInjector] Injected LoRA into {len(lora_layers)} layers "
            f"(rank={rank}, alpha={alpha})."
        )
        return lora_layers

    @staticmethod
    def get_parameters(lora_layers: Dict[str, LoRALayer]) -> List[nn.Parameter]:
        """All trainable LoRA parameters (A and B for each layer)."""
        params: List[nn.Parameter] = []
        for layer in lora_layers.values():
            params += [layer.lora_A, layer.lora_B]
        return params

    @staticmethod
    def extract_state(
        lora_layers: Dict[str, LoRALayer]
    ) -> Dict[str, torch.Tensor]:
        """
        Build a flat dict suitable for safetensors serialisation.

        Keys:  "<dotted_path>.lora_A"  and  "<dotted_path>.lora_B"
        """
        state: Dict[str, torch.Tensor] = {}
        for path, layer in lora_layers.items():
            state[f"{path}.lora_A"] = layer.lora_A.data.cpu().contiguous()
            state[f"{path}.lora_B"] = layer.lora_B.data.cpu().contiguous()
        return state

    @staticmethod
    def load_state(
        lora_layers: Dict[str, LoRALayer],
        state: Dict[str, torch.Tensor],
    ) -> None:
        """Load a saved state dict back into existing LoRA layers."""
        for path, layer in lora_layers.items():
            k_A = f"{path}.lora_A"
            k_B = f"{path}.lora_B"
            if k_A in state:
                layer.lora_A.data.copy_(state[k_A])
            if k_B in state:
                layer.lora_B.data.copy_(state[k_B])

    @staticmethod
    def apply_to_unet(
        unet: nn.Module,
        lora_layers: Dict[str, LoRALayer],
        scale: float = 1.0,
    ) -> None:
        """
        Fuse LoRA deltas into UNet weight tensors in-place.
        Call remove_from_unet to undo.
        """
        for path, layer in lora_layers.items():
            module = _navigate(unet, path)
            if module is not None and hasattr(module, "weight"):
                delta = layer.effective_weight().to(
                    dtype=module.weight.dtype, device=module.weight.device
                ) * scale
                module.weight.data.add_(delta)

    @staticmethod
    def remove_from_unet(
        unet: nn.Module,
        lora_layers: Dict[str, LoRALayer],
        scale: float = 1.0,
    ) -> None:
        """Subtract previously fused LoRA deltas."""
        for path, layer in lora_layers.items():
            module = _navigate(unet, path)
            if module is not None and hasattr(module, "weight"):
                delta = layer.effective_weight().to(
                    dtype=module.weight.dtype, device=module.weight.device
                ) * scale
                module.weight.data.sub_(delta)


def _navigate(root: nn.Module, dotted_path: str) -> Optional[nn.Module]:
    """Walk a module hierarchy by dotted attribute path."""
    obj = root
    for part in dotted_path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


# ──────────────────────────────────────────────────────────────────────────────
# Textual Inversion manager
# ──────────────────────────────────────────────────────────────────────────────

class TextualInversionManager:
    """
    Manages one or more learnable token embeddings inside a CLIP text encoder.

    Typical usage
    -------------
    manager = TextualInversionManager(tokenizer, text_encoder)
    token_ids = manager.add_tokens(["<sks>"])
    params    = manager.learnable_params()    # pass to optimiser
    manager.save("./ti_weights/subject_0.pt")
    manager.load("./ti_weights/subject_0.pt")
    """

    def __init__(self, tokenizer, text_encoder: nn.Module) -> None:
        self.tokenizer    = tokenizer
        self.text_encoder = text_encoder
        # {token_string: token_id}
        self._token_ids: Dict[str, int] = {}

    def add_tokens(
        self,
        tokens: List[str],
        init_text: str = "person",
    ) -> Dict[str, int]:
        """
        Register new tokens, resize the embedding table, and initialise each
        new embedding as the mean of `init_text` token embeddings + small noise.

        Returns
        -------
        dict  {token: token_id}
        """
        n_added = self.tokenizer.add_tokens(tokens)
        if n_added == 0 and not all(
            t in self.tokenizer.get_vocab() for t in tokens
        ):
            raise RuntimeError(f"Failed to add tokens: {tokens}")

        self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        embed_layer = self.text_encoder.get_input_embeddings()

        # Compute initialisation embedding from init_text
        init_ids = self.tokenizer(
            init_text, return_tensors="pt", add_special_tokens=False
        ).input_ids
        with torch.no_grad():
            init_embed = embed_layer(init_ids).mean(dim=1).squeeze(0)  # [dim]

        token_ids: Dict[str, int] = {}
        for token in tokens:
            tid = self.tokenizer.convert_tokens_to_ids(token)
            # Initialise with mean embed + tiny noise
            with torch.no_grad():
                embed_layer.weight[tid] = (
                    init_embed + torch.randn_like(init_embed) * 0.01
                )
            token_ids[token] = tid
            self._token_ids[token] = tid

        print(
            f"[TextualInversionManager] Added {len(tokens)} token(s): "
            f"{tokens} → ids {list(token_ids.values())}"
        )
        return token_ids

    def learnable_params(self) -> List[nn.Parameter]:
        """
        Return the embedding rows for our tokens as a single parameter.

        NOTE: This returns the full embedding weight matrix; you must use
        a custom step that zeros gradients for all *other* rows (see
        train_textual_inversion.py for the correct approach).
        """
        return [self.text_encoder.get_input_embeddings().weight]

    def get_embedding(self, token: str) -> torch.Tensor:
        """Return the current embedding vector for `token`."""
        tid = self._token_ids[token]
        return self.text_encoder.get_input_embeddings().weight[tid].detach().cpu()

    def save(self, path: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            tok: self.text_encoder.get_input_embeddings().weight[tid].detach().cpu()
            for tok, tid in self._token_ids.items()
        }
        torch.save({"token_ids": self._token_ids, "embeddings": payload}, path)
        print(f"[TextualInversionManager] Saved → {path}")

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu")
        embed_layer = self.text_encoder.get_input_embeddings()

        for token, embedding in ckpt["embeddings"].items():
            # Re-register in tokeniser if needed
            if token not in self.tokenizer.get_vocab():
                self.tokenizer.add_tokens([token])
                self.text_encoder.resize_token_embeddings(len(self.tokenizer))
                embed_layer = self.text_encoder.get_input_embeddings()

            tid = self.tokenizer.convert_tokens_to_ids(token)
            with torch.no_grad():
                embed_layer.weight[tid] = embedding.to(embed_layer.weight.device)
            self._token_ids[token] = tid

        print(f"[TextualInversionManager] Loaded from {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def lora_param_count(
    unet: nn.Module,
    rank: int,
    target_modules: Tuple[str, ...] = _DEFAULT_TARGET_MODULES,
) -> int:
    """Count total LoRA parameters for a given rank."""
    total = 0
    for name, module in unet.named_modules():
        if isinstance(module, nn.Linear) and any(
            name.endswith(t) for t in target_modules
        ):
            total += module.in_features * rank + rank * module.out_features
    return total


def spearman_correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    Compute Spearman rank correlation between two 1-D tensors.
    Used to report the LoRA-rank vs. leakage correlation (ρ = 0.83 in paper).
    """
    from scipy.stats import spearmanr
    rho, _ = spearmanr(x.cpu().numpy(), y.cpu().numpy())
    return float(rho)

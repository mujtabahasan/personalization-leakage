"""
models/diffusion_wrapper.py
───────────────────────────
Thin wrapper around Stable Diffusion 1.5 that supports:
  • clean inference (generate)
  • VAE encode / decode
  • text-encoder embed
  • denoising-loss computation (for MIA scoring)
  • apply / remove LoRA delta weights
  • apply Textual Inversion token embeddings

All heavy state lives in self.pipe so the wrapper can be serialised
cheaply (just keep a reference to an existing pipe).

Dependencies: diffusers>=0.25, transformers>=4.35, safetensors>=0.4
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


# ──────────────────────────────────────────────────────────────────────────────

class DiffusionWrapper:
    """
    Wraps StableDiffusionPipeline for privacy-leakage experiments.

    Parameters
    ----------
    model_id : str
        HuggingFace repo id, default "runwayml/stable-diffusion-v1-5".
    device : str
        "cuda" or "cpu".
    dtype : torch.dtype
        torch.float16 on CUDA, torch.float32 on CPU.
    enable_xformers : bool
        Enable memory-efficient attention when xformers is installed.
    """

    MODEL_ID_DEFAULT = "runwayml/stable-diffusion-v1-5"

    def __init__(
        self,
        model_id: str = MODEL_ID_DEFAULT,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: Optional[torch.dtype] = None,
        enable_xformers: bool = False,
    ) -> None:
        from diffusers import StableDiffusionPipeline, DDPMScheduler

        self.device = device
        self.dtype  = dtype or (torch.float16 if device == "cuda" else torch.float32)

        print(f"[DiffusionWrapper] Loading {model_id} on {device} …")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(device)

        if enable_xformers and device == "cuda":
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                print("  xformers memory-efficient attention enabled.")
            except Exception as exc:
                print(f"  xformers not available ({exc}), continuing without.")

        # Convenient aliases
        self.unet         = self.pipe.unet
        self.vae          = self.pipe.vae
        self.text_encoder = self.pipe.text_encoder
        self.tokenizer    = self.pipe.tokenizer
        self.scheduler    = self.pipe.scheduler   # DDPMScheduler

        # Store the VAE scaling factor once
        self._vae_scale: float = self.vae.config.scaling_factor  # 0.18215 for SD 1.5

    # ── inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        height: int = 512,
        width:  int = 512,
        seed: Optional[int] = None,
        output_type: str = "tensor",   # "tensor" | "pil"
    ) -> Union[torch.Tensor, List[Image.Image]]:
        """
        Generate images.

        Returns
        -------
        If output_type == "tensor": float32 tensor [B, 3, H, W] in [0, 1].
        If output_type == "pil": list of PIL images.
        """
        generator = (
            torch.Generator(device=self.device).manual_seed(seed)
            if seed is not None else None
        )

        out = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
            output_type="pil",
        )
        pil_images = out.images

        if output_type == "pil":
            return pil_images

        # Convert PIL → float32 tensor in [0, 1]
        import numpy as np
        tensors = [
            torch.from_numpy(np.array(img).astype("float32") / 255.0).permute(2, 0, 1)
            for img in pil_images
        ]
        return torch.stack(tensors)  # [B, 3, H, W]

    # ── encoding helpers ──────────────────────────────────────────────────────

    @torch.no_grad()
    def encode_text(self, prompts: Union[str, List[str]]) -> torch.Tensor:
        """
        Tokenise and encode text prompts.

        Returns
        -------
        Tensor [B, seq_len, hidden_dim]
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        ids = tokens.input_ids.to(self.device)
        return self.text_encoder(ids)[0]

    @torch.no_grad()
    def encode_images_to_latents(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encode pixel images to VAE latent space.

        Parameters
        ----------
        pixel_values : Tensor [B, 3, H, W] in [-1, 1]

        Returns
        -------
        Tensor [B, 4, H/8, W/8]
        """
        latents = self.vae.encode(
            pixel_values.to(self.device, dtype=self.dtype)
        ).latent_dist.sample()
        return latents * self._vae_scale

    @torch.no_grad()
    def decode_latents_to_images(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode VAE latents to pixel images in [0, 1].

        Returns
        -------
        Tensor [B, 3, H, W] float32
        """
        latents = latents / self._vae_scale
        imgs = self.vae.decode(latents.to(dtype=self.dtype)).sample
        imgs = ((imgs / 2) + 0.5).clamp(0, 1).float()
        return imgs

    # ── diffusion loss ────────────────────────────────────────────────────────

    def compute_diffusion_loss(
        self,
        pixel_values: torch.Tensor,
        prompts: List[str],
        num_timesteps: int = 1,
    ) -> torch.Tensor:
        """
        Compute the denoising MSE loss averaged over `num_timesteps` random
        timesteps.  Used as the MIA score (lower loss → model has seen image).

        Parameters
        ----------
        pixel_values : Tensor [B, 3, H, W] in [-1, 1]
        prompts      : list of B prompt strings
        num_timesteps: how many random timesteps to average over

        Returns
        -------
        Scalar loss tensor (requires_grad=False).
        """
        pixel_values = pixel_values.to(self.device, dtype=self.dtype)

        with torch.no_grad():
            latents = self.encode_images_to_latents(pixel_values)
            text_emb = self.encode_text(prompts)

        losses = []
        for _ in range(num_timesteps):
            noise = torch.randn_like(latents)
            t = torch.randint(
                0, self.scheduler.config.num_train_timesteps,
                (latents.shape[0],), device=self.device,
            ).long()

            noisy = self.scheduler.add_noise(latents, noise, t)

            with torch.no_grad():
                pred = self.unet(
                    noisy, t, encoder_hidden_states=text_emb
                ).sample

            losses.append(F.mse_loss(pred.float(), noise.float()))

        return torch.stack(losses).mean()

    # ── LoRA apply / remove ───────────────────────────────────────────────────

    def apply_lora(
        self,
        lora_state: Dict[str, torch.Tensor],
        scale: float = 1.0,
    ) -> None:
        """
        Add LoRA delta weights (A @ B * scale) directly to UNet parameters.

        Expects lora_state keys of the form:
          "<module_path>.lora_A" and "<module_path>.lora_B"

        Call remove_lora with the same state to undo.
        """
        # Collect (base_name, A, B) triples
        names = {k.rsplit(".", 1)[0] for k in lora_state if k.endswith(".lora_A")}
        for base in names:
            A = lora_state[f"{base}.lora_A"].to(self.device, dtype=torch.float32)
            B = lora_state[f"{base}.lora_B"].to(self.device, dtype=torch.float32)
            delta = (A @ B) * scale   # [in_features, out_features]

            param = self._get_unet_param(base)
            if param is not None:
                param.data.add_(delta.to(param.dtype))

    def remove_lora(
        self,
        lora_state: Dict[str, torch.Tensor],
        scale: float = 1.0,
    ) -> None:
        """Subtract previously-applied LoRA deltas."""
        names = {k.rsplit(".", 1)[0] for k in lora_state if k.endswith(".lora_A")}
        for base in names:
            A = lora_state[f"{base}.lora_A"].to(self.device, dtype=torch.float32)
            B = lora_state[f"{base}.lora_B"].to(self.device, dtype=torch.float32)
            delta = (A @ B) * scale

            param = self._get_unet_param(base)
            if param is not None:
                param.data.sub_(delta.to(param.dtype))

    def _get_unet_param(self, dotted_path: str) -> Optional[nn.Parameter]:
        """Navigate UNet module hierarchy by dotted path, return weight param."""
        obj = self.unet
        for part in dotted_path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                return None
        # obj should now be a Linear / Conv2d; return its weight
        if hasattr(obj, "weight"):
            return obj.weight
        return None

    # ── Textual Inversion ─────────────────────────────────────────────────────

    def apply_textual_inversion(
        self,
        token: str,
        embedding: torch.Tensor,
    ) -> int:
        """
        Insert a new learned token into the tokeniser and set its embedding.

        Returns the token id.
        """
        self.tokenizer.add_tokens([token])
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))

        token_id = self.tokenizer.convert_tokens_to_ids(token)

        with torch.no_grad():
            embed_layer = self.text_encoder.get_input_embeddings()
            embed_layer.weight[token_id] = embedding.to(
                self.device, dtype=embed_layer.weight.dtype
            )

        return token_id

    # ── context manager for temporary adapter ─────────────────────────────────

    @contextlib.contextmanager
    def with_lora(
        self,
        lora_state: Dict[str, torch.Tensor],
        scale: float = 1.0,
    ):
        """Context manager: apply LoRA, yield, then remove."""
        self.apply_lora(lora_state, scale)
        try:
            yield self
        finally:
            self.remove_lora(lora_state, scale)

    # ── misc ──────────────────────────────────────────────────────────────────

    def save_lora(
        self,
        lora_state: Dict[str, torch.Tensor],
        path: Union[str, Path],
    ) -> None:
        from safetensors.torch import save_file
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file({k: v.cpu().contiguous() for k, v in lora_state.items()}, path)
        print(f"[DiffusionWrapper] LoRA saved → {path}")

    def load_lora(self, path: Union[str, Path]) -> Dict[str, torch.Tensor]:
        from safetensors.torch import load_file
        return load_file(Path(path))

    def save_textual_inversion(
        self,
        token: str,
        token_id: int,
        path: Union[str, Path],
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        embed = self.text_encoder.get_input_embeddings().weight[token_id].detach().cpu()
        torch.save({"token": token, "embedding": embed}, path)
        print(f"[DiffusionWrapper] TI embedding saved → {path}")

    def load_textual_inversion(
        self, path: Union[str, Path]
    ) -> Tuple[str, torch.Tensor]:
        from typing import Tuple
        d = torch.load(Path(path), map_location="cpu")
        return d["token"], d["embedding"]

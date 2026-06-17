"""
utils/data_utils.py
────────────────────
Shared data-loading utilities used across all experiment pipelines.

Provides
────────
SubjectLoader        – load member / non-member images for one subject
ReferenceLoader      – load out-of-distribution reference images
ShadowDatasetLoader  – load shadow-model training sets
collate_for_mia      – build (images, prompts, labels) tensors ready for scoring
find_adapter_path    – locate adapter file for a given subject
load_adapter_state   – load adapter weights from disk
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# ──────────────────────────────────────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────────────────────────────────────

_NORM = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

_TRAIN_TF = transforms.Compose([
    transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.LANCZOS),
    transforms.ToTensor(),
    _NORM,
])

_EVAL_TF = transforms.Compose([
    transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.LANCZOS),
    transforms.ToTensor(),
    _NORM,
])


def load_pil(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def pil_to_tensor(img: Image.Image, train: bool = False) -> torch.Tensor:
    return (_TRAIN_TF if train else _EVAL_TF)(img)


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """[3,H,W] in [-1,1] or [0,1] -> uint8 PIL."""
    t = t.detach().float().cpu()
    if t.min() < -0.1:
        t = (t + 1.0) / 2.0
    arr = (t.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def load_images(paths: List[Path], train: bool = False) -> torch.Tensor:
    """Returns [N, 3, 512, 512] float32 in [-1,1]."""
    return torch.stack([pil_to_tensor(load_pil(p), train=train) for p in paths])


# ──────────────────────────────────────────────────────────────────────────────
# Subject loader
# ──────────────────────────────────────────────────────────────────────────────

class SubjectLoader:
    """Load member (train) and non-member (test) images for one subject."""

    def __init__(
        self,
        data_dir:   str,
        subject_id: str,
        max_images: Optional[int] = None,
        seed:       int = 42,
    ) -> None:
        self.data_dir   = Path(data_dir)
        self.subject_id = subject_id
        self.max_images = max_images
        self._rng       = random.Random(seed)

        splits = json.loads((self.data_dir / "splits.json").read_text())
        if subject_id not in splits:
            raise KeyError(f"Subject '{subject_id}' not found in splits.json")

        self.info = splits[subject_id]
        self._subjects_dir = self.data_dir / "subjects"

    @property
    def prompt(self) -> str:
        return self.info["class_prompt"]

    @property
    def subject_dir(self) -> Path:
        return self._subjects_dir / self.info["subject_dir"]

    def _paths(self, split: str) -> List[Path]:
        names = list(self.info[split])
        self._rng.shuffle(names)
        if self.max_images is not None:
            names = names[: self.max_images]
        return [self.subject_dir / n for n in names]

    def member_paths(self)    -> List[Path]:    return self._paths("train")
    def nonmember_paths(self) -> List[Path]:    return self._paths("test")
    def member_images(self)   -> torch.Tensor:  return load_images(self.member_paths())
    def nonmember_images(self)-> torch.Tensor:  return load_images(self.nonmember_paths())

    def member_prompts(self,    n: Optional[int] = None) -> List[str]:
        return [self.prompt] * (n or len(self.member_paths()))

    def nonmember_prompts(self, n: Optional[int] = None) -> List[str]:
        return [self.prompt] * (n or len(self.nonmember_paths()))

    def all_splits(self) -> Tuple[torch.Tensor, List[str], torch.Tensor, List[str]]:
        m  = self.member_images();    nm = self.nonmember_images()
        return m, [self.prompt]*len(m), nm, [self.prompt]*len(nm)


# ──────────────────────────────────────────────────────────────────────────────
# Reference loader
# ──────────────────────────────────────────────────────────────────────────────

class ReferenceLoader:
    """OOD images not used to train any adapter."""

    def __init__(self, data_dir: str, max_images: Optional[int] = None, seed: int = 42):
        ref_dir = Path(data_dir) / "reference"
        if not ref_dir.exists():
            raise FileNotFoundError(ref_dir)
        paths = sorted(ref_dir.glob("*.jpg")) + sorted(ref_dir.glob("*.png"))
        rng = random.Random(seed); rng.shuffle(paths)
        self.paths = paths[:max_images] if max_images else paths

    def images(self) -> torch.Tensor:
        return load_images(self.paths)

    def prompts(self, prompt: str = "a photo") -> List[str]:
        return [prompt] * len(self.paths)


# ──────────────────────────────────────────────────────────────────────────────
# Shadow dataset loader
# ──────────────────────────────────────────────────────────────────────────────

class ShadowDatasetLoader:
    """Load pre-built shadow model splits."""

    def __init__(self, data_dir: str) -> None:
        cfg = Path(data_dir) / "shadow_models" / "shadow_configs.json"
        if not cfg.exists():
            raise FileNotFoundError(cfg)
        self.configs: List[Dict] = json.loads(cfg.read_text())

    def __len__(self) -> int:
        return len(self.configs)

    def load_shadow(self, shadow_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        c = self.configs[shadow_id]
        m  = load_images([Path(p) for p in c["members"][:20]    if Path(p).exists()])
        nm = load_images([Path(p) for p in c["non_members"][:20] if Path(p).exists()])
        return m, nm


# ──────────────────────────────────────────────────────────────────────────────
# PyTorch dataset wrappers
# ──────────────────────────────────────────────────────────────────────────────

class ImagePromptDataset(Dataset):
    def __init__(self, image_paths: List[Path], prompt: str, train: bool = True):
        self.paths  = image_paths
        self.prompt = prompt
        self.tf     = _TRAIN_TF if train else _EVAL_TF

    def __len__(self):  return len(self.paths)

    def __getitem__(self, idx):
        return self.tf(load_pil(self.paths[idx])), self.prompt


class MIADataset(Dataset):
    def __init__(self, member_paths: List[Path], nonmember_paths: List[Path], prompt: str):
        self.paths  = member_paths + nonmember_paths
        self.labels = [1]*len(member_paths) + [0]*len(nonmember_paths)
        self.prompt = prompt

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        return _EVAL_TF(load_pil(self.paths[idx])), self.prompt, self.labels[idx]


# ──────────────────────────────────────────────────────────────────────────────
# Collation helpers
# ──────────────────────────────────────────────────────────────────────────────

def collate_for_mia(
    member_images:    torch.Tensor,
    nonmember_images: torch.Tensor,
    prompt:           str,
) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
    """Merge member/non-member into one batch with binary labels."""
    imgs   = torch.cat([member_images, nonmember_images], dim=0)
    labels = torch.cat([
        torch.ones(len(member_images),    dtype=torch.long),
        torch.zeros(len(nonmember_images),dtype=torch.long),
    ])
    return imgs, [prompt] * len(imgs), labels


def iter_subjects(
    data_dir:    str,
    subject_ids: Optional[List[str]] = None,
    max_images:  Optional[int] = None,
    seed:        int = 42,
):
    """Generator yielding (subject_id, SubjectLoader)."""
    splits = json.loads((Path(data_dir) / "splits.json").read_text())
    ids    = subject_ids if subject_ids else list(splits.keys())
    for sid in ids:
        try:
            yield sid, SubjectLoader(data_dir, sid, max_images=max_images, seed=seed)
        except KeyError:
            print(f"  ⚠ subject {sid} missing, skipping")


# ──────────────────────────────────────────────────────────────────────────────
# Adapter file helpers
# ──────────────────────────────────────────────────────────────────────────────

def find_adapter_path(
    adapter_dir:  Path,
    adapter_type: str,
    subject_id:   str,
    rank:         Optional[int] = None,
) -> Optional[Path]:
    """Locate an adapter file for a given subject."""
    adapter_dir = Path(adapter_dir)

    if adapter_type == "lora":
        if rank is not None:
            hit = adapter_dir / f"lora_rank{rank}_subject{subject_id}.safetensors"
            if hit.exists():
                return hit
        candidates = (
            list(adapter_dir.glob(f"lora_rank*_subject{subject_id}.safetensors")) +
            list(adapter_dir.glob(f"*subject{subject_id}*.safetensors"))
        )
        if candidates:
            if rank:
                ranked = [c for c in candidates if f"rank{rank}" in c.name]
                if ranked:
                    return ranked[0]
            return candidates[0]

    elif adapter_type == "textual_inversion":
        hit = adapter_dir / f"ti_subject{subject_id}.pt"
        if hit.exists():
            return hit
        candidates = list(adapter_dir.glob(f"*subject{subject_id}*.pt"))
        if candidates:
            return candidates[0]

    return None


def load_adapter_state(
    adapter_path: Path,
    adapter_type: str,
    wrapper=None,
) -> Dict:
    """
    Load adapter weights.  For TI, registers embedding in `wrapper` (side effect).
    Returns safetensors dict for LoRA, empty dict for TI.
    """
    if adapter_type == "lora":
        from safetensors.torch import load_file
        return load_file(adapter_path)
    elif adapter_type == "textual_inversion":
        if wrapper is None:
            raise ValueError("wrapper required for TI")
        ckpt = torch.load(adapter_path, map_location="cpu")
        for token, emb in ckpt["embeddings"].items():
            wrapper.apply_textual_inversion(token, emb)
        return {}
    else:
        raise ValueError(f"Unknown adapter_type: {adapter_type}")

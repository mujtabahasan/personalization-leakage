"""
data_preparation.py
───────────────────
Prepares the dataset for all privacy-leakage experiments.

Supports:
  • CelebA (30 identities, as used in the paper)
  • DreamBooth (Google dataset via HuggingFace)
  • Synthetic (random images, for smoke-tests / CI)

Outputs under --output_dir:
  subjects/subject_<id>/   raw images
  splits.json              member/non-member splits per subject
  reference/               out-of-distribution images (COCO-style)
  shadow_models/           per-shadow train sets

Usage examples
──────────────
# Paper setup (30 CelebA subjects)
python data_preparation.py --dataset celeba --num_subjects 30 --output_dir ./data

# Quick synthetic smoke-test
python data_preparation.py --dataset synthetic --num_subjects 5 --output_dir ./data_test
"""

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def resize_and_save(img: Image.Image, path: Path, size: int = 512) -> None:
    """Resize to square and save as JPEG."""
    img = img.convert("RGB")
    img = img.resize((size, size), Image.LANCZOS)
    img.save(path, quality=95)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset builders
# ──────────────────────────────────────────────────────────────────────────────

class PrivacyDatasetBuilder:
    def __init__(self, output_dir: str, seed: int = 42):
        self.output_dir = Path(output_dir)
        self.seed = seed
        set_seed(seed)

    # ── subject images ────────────────────────────────────────────────────────

    def build_synthetic(self, num_subjects: int, images_per_subject: int = 15) -> None:
        """Create random-colour synthetic subjects (no downloads needed)."""
        subjects_dir = self.output_dir / "subjects"
        subjects_dir.mkdir(parents=True, exist_ok=True)

        print(f"[synthetic] Creating {num_subjects} subjects …")
        for sid in tqdm(range(num_subjects), desc="Subjects"):
            sdir = subjects_dir / f"subject_{sid:03d}"
            sdir.mkdir(exist_ok=True)

            # Each subject gets a dominant random hue so subjects are distinct
            hue = np.random.randint(0, 255, 3)
            for idx in range(images_per_subject):
                noise = np.clip(
                    hue + np.random.randint(-40, 40, (512, 512, 3)), 0, 255
                ).astype(np.uint8)
                img = Image.fromarray(noise)
                img.save(sdir / f"{idx:03d}.jpg")

            (sdir / "metadata.json").write_text(json.dumps({
                "subject_id": sid,
                "name": f"synthetic_{sid:03d}",
                "class_prompt": f"a photo of person{sid:03d}",
            }, indent=2))

    def build_celeba(self, num_subjects: int, celeba_root: str) -> None:
        """
        Build from a local CelebA directory.
        Expected structure:  celeba_root/identity_CelebA.txt
                             celeba_root/img_align_celeba/<filename>

        Paper uses 30 identities with at least 15 images each.
        """
        celeba_root = Path(celeba_root)
        id_file = celeba_root / "identity_CelebA.txt"
        img_dir = celeba_root / "img_align_celeba"

        if not id_file.exists() or not img_dir.exists():
            raise FileNotFoundError(
                f"CelebA root not found or incomplete at {celeba_root}.\n"
                "Download from https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html"
            )

        # Parse identity file  →  {identity_id: [filename, …]}
        identity_map: Dict[int, List[str]] = {}
        for line in id_file.read_text().strip().splitlines():
            fname, iid = line.strip().split()
            identity_map.setdefault(int(iid), []).append(fname)

        # Keep identities with ≥ 15 images
        eligible = {k: v for k, v in identity_map.items() if len(v) >= 15}
        selected_ids = sorted(eligible.keys())[:num_subjects]
        if len(selected_ids) < num_subjects:
            raise ValueError(
                f"Only {len(selected_ids)} CelebA identities with ≥15 images; "
                f"requested {num_subjects}."
            )

        subjects_dir = self.output_dir / "subjects"
        subjects_dir.mkdir(parents=True, exist_ok=True)

        print(f"[CelebA] Preparing {num_subjects} identities …")
        for sid, celeba_id in enumerate(tqdm(selected_ids, desc="Identities")):
            sdir = subjects_dir / f"subject_{sid:03d}"
            sdir.mkdir(exist_ok=True)

            for idx, fname in enumerate(eligible[celeba_id][:20]):
                src = img_dir / fname
                resize_and_save(Image.open(src), sdir / f"{idx:03d}.jpg")

            (sdir / "metadata.json").write_text(json.dumps({
                "subject_id": sid,
                "celeba_id": celeba_id,
                "name": f"celeba_{celeba_id}",
                "class_prompt": "a photo of a person",
            }, indent=2))

    def build_dreambooth(self, num_subjects: int) -> None:
        """Download DreamBooth dataset from HuggingFace datasets."""
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError:
            raise ImportError("pip install datasets")

        print("[DreamBooth] Downloading …")
        ds = load_dataset("google/dreambooth", split="train", streaming=False)

        subjects_dir = self.output_dir / "subjects"
        subjects_dir.mkdir(parents=True, exist_ok=True)

        # Group by subject name (first word of prompt)
        subject_map: Dict[str, List] = {}
        for item in ds:
            key = item["prompt"].split()[0].lower()
            subject_map.setdefault(key, []).append(item)

        chosen = list(subject_map.keys())[:num_subjects]

        for sid, name in enumerate(tqdm(chosen, desc="DreamBooth subjects")):
            sdir = subjects_dir / f"subject_{sid:03d}_{name}"
            sdir.mkdir(exist_ok=True)

            for idx, item in enumerate(subject_map[name][:20]):
                resize_and_save(item["image"], sdir / f"{idx:03d}.jpg")

            (sdir / "metadata.json").write_text(json.dumps({
                "subject_id": sid,
                "name": name,
                "class_prompt": f"a photo of {name}",
            }, indent=2))

    # ── splits ────────────────────────────────────────────────────────────────

    def create_splits(
        self,
        train_ratio: float = 0.7,
        min_train: int = 5,
        min_test: int = 3,
    ) -> None:
        """Write splits.json with member / non-member partitions."""
        subjects_dir = self.output_dir / "subjects"
        splits: Dict[str, dict] = {}

        for sdir in sorted(subjects_dir.iterdir()):
            if not sdir.is_dir():
                continue
            meta = json.loads((sdir / "metadata.json").read_text())
            sid = str(meta["subject_id"])

            imgs = sorted(str(p.name) for p in sdir.glob("*.jpg"))
            if len(imgs) < min_train + min_test:
                print(f"  ⚠ skip {sdir.name}: only {len(imgs)} images")
                continue

            random.shuffle(imgs)
            cut = max(min_train, int(len(imgs) * train_ratio))
            cut = min(cut, len(imgs) - min_test)

            splits[sid] = {
                "subject_dir": sdir.name,
                "class_prompt": meta["class_prompt"],
                "train": imgs[:cut],       # members
                "test":  imgs[cut:],       # non-members
            }

        out = self.output_dir / "splits.json"
        out.write_text(json.dumps(splits, indent=2))
        print(f"[splits] {len(splits)} subjects → {out}")

    # ── reference images ──────────────────────────────────────────────────────

    def create_reference_dataset(self, num_images: int = 500) -> None:
        """Synthetic out-of-distribution images for MIA baselines."""
        ref_dir = self.output_dir / "reference"
        ref_dir.mkdir(parents=True, exist_ok=True)

        print(f"[reference] Creating {num_images} OOD images …")
        for i in tqdm(range(num_images), desc="Reference images"):
            arr = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
            Image.fromarray(arr).save(ref_dir / f"ref_{i:05d}.jpg")

    # ── shadow model data ─────────────────────────────────────────────────────

    def create_shadow_splits(self, num_shadow: int = 5) -> None:
        """
        Create independent member/non-member splits for shadow model training.
        Each shadow model is trained on a different random subset of subjects.
        """
        subjects_dir = self.output_dir / "subjects"
        shadow_dir = self.output_dir / "shadow_models"
        shadow_dir.mkdir(parents=True, exist_ok=True)

        all_imgs = sorted(subjects_dir.glob("*/*.jpg"))
        configs = []

        for sh_id in range(num_shadow):
            random.shuffle(all_imgs)
            n_member = len(all_imgs) // 2
            configs.append({
                "shadow_id": sh_id,
                "members":     [str(p) for p in all_imgs[:n_member]],
                "non_members": [str(p) for p in all_imgs[n_member:]],
            })

        out = shadow_dir / "shadow_configs.json"
        out.write_text(json.dumps(configs, indent=2))
        print(f"[shadow] {num_shadow} shadow configs → {out}")

    # ── verify ────────────────────────────────────────────────────────────────

    def verify(self) -> Dict[str, int]:
        stats: Dict[str, int] = {}

        subjects_dir = self.output_dir / "subjects"
        stats["subjects"] = sum(1 for d in subjects_dir.iterdir() if d.is_dir()) \
            if subjects_dir.exists() else 0
        stats["subject_images"] = len(list(subjects_dir.glob("*/*.jpg"))) \
            if subjects_dir.exists() else 0

        splits_file = self.output_dir / "splits.json"
        if splits_file.exists():
            sp = json.loads(splits_file.read_text())
            stats["split_subjects"] = len(sp)
            stats["train_images"]   = sum(len(v["train"]) for v in sp.values())
            stats["test_images"]    = sum(len(v["test"])  for v in sp.values())

        ref_dir = self.output_dir / "reference"
        stats["reference_images"] = len(list(ref_dir.glob("*.jpg"))) \
            if ref_dir.exists() else 0

        print("\n[verify] Dataset statistics:")
        for k, v in stats.items():
            print(f"  {k:<25} {v}")
        return stats


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare privacy-leakage datasets")
    p.add_argument("--dataset",      type=str,  default="synthetic",
                   choices=["synthetic", "celeba", "dreambooth"])
    p.add_argument("--num_subjects", type=int,  default=30)
    p.add_argument("--num_reference",type=int,  default=500)
    p.add_argument("--num_shadow",   type=int,  default=5)
    p.add_argument("--output_dir",   type=str,  default="./data")
    p.add_argument("--celeba_root",  type=str,  default="./celeba",
                   help="Required only when --dataset celeba")
    p.add_argument("--train_ratio",  type=float, default=0.7)
    p.add_argument("--seed",         type=int,  default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    builder = PrivacyDatasetBuilder(args.output_dir, seed=args.seed)

    # Step 1: subject images
    if args.dataset == "synthetic":
        builder.build_synthetic(args.num_subjects)
    elif args.dataset == "celeba":
        builder.build_celeba(args.num_subjects, args.celeba_root)
    elif args.dataset == "dreambooth":
        builder.build_dreambooth(args.num_subjects)

    # Step 2: splits
    builder.create_splits(train_ratio=args.train_ratio)

    # Step 3: reference OOD images
    builder.create_reference_dataset(args.num_reference)

    # Step 4: shadow splits
    builder.create_shadow_splits(args.num_shadow)

    # Step 5: verify
    builder.verify()
    print("\n✓ Dataset ready.")


if __name__ == "__main__":
    main()

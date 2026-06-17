
# How Much Do Shared Adapters Leak?
### Measuring Privacy Risks in Personalized Diffusion Model Weights
**P13N@CVPR 2026 — Camera-Ready Reproducible Code Package**

---

## Overview

We study privacy leakage when users share fine-tuned **adapter weights** for personalized text-to-image diffusion models. Two adapter families are attacked:

| Adapter | Parameters | Key finding |
|---------|-----------|-------------|
| **LoRA** (ranks 4–64) | ~200K–3.2M | Higher rank → more leakage (ρ = 0.83) |
| **Textual Inversion** | 768 | Leaks *more* than LoRA despite 530× fewer params |

**Three attack classes:**
1. **Membership Inference (MIA)** — does an image belong to the training set?
2. **Data Reconstruction** — recover training images from adapter weights
3. **Statistical / white-box** — Fisher information, weight norms, spectral analysis

---

## Installation

```bash
# GPU (recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# CPU-only
pip install -r requirements.txt
```

**Google Colab:**
```python
!pip install -q diffusers transformers accelerate safetensors lpips scikit-learn
```

---

## Quick Start (5 minutes)

```bash
# 1. Synthetic dataset (no download)
python data_preparation.py --dataset synthetic --num_subjects 5

# 2. Validate all code
python validate_code.py --fast

# 3. Single-subject MIA (requires trained adapter)
python experiments/membership_inference.py \
    --mode single \
    --data_dir ./data \
    --adapter_path ./adapters/lora_rank16_subject0.safetensors \
    --subject_id 0 --adapter_type lora
```

---

## Full Paper Reproduction

### Step 1 — Data

```bash
# CelebA (paper setup: 30 identities)
python data_preparation.py \
    --dataset celeba \
    --celeba_root /path/to/celeba \
    --num_subjects 30

# Or synthetic (for testing)
python data_preparation.py --dataset synthetic --num_subjects 30
```

### Step 2 — Train Adapters

```bash
# LoRA — all 5 ranks (paper ablation)
for r in 4 8 16 32 64; do
    python train_lora_adapters.py --rank $r --num_subjects 30
done

# Textual Inversion
python train_textual_inversion.py --num_subjects 30
```

### Step 3 — Run All Attacks

```bash
# One command reproduces all tables and figures:
bash run_all_experiments.sh

# Or Python orchestrator:
python experiments/run_all_attacks.py \
    --lora_dir ./adapters \
    --ti_dir   ./adapters_ti \
    --ranks 4 8 16 32 64 \
    --num_subjects 30
```

---

## Individual Experiment Pipelines

### Membership Inference (Table 1, Figure 1)

```bash
# All subjects, LoRA r=16
python experiments/membership_inference.py \
    --mode multi --adapter_type lora --rank 16 --num_subjects 30

# LoRA vs TI overlay ROC (Figure 1)
python experiments/membership_inference.py \
    --mode compare \
    --result_files results/mia_lora_rank16/mia_multi_lora.json \
                   results/mia_ti/mia_multi_textual_inversion.json \
    --result_labels "LoRA r=16" "TI"
```

### Data Reconstruction (Table 1, Figure 3)

```bash
# Single subject
python experiments/data_extraction.py \
    --mode single --adapter_type lora --subject_id 0 --n_candidates 10

# All subjects
python experiments/data_extraction.py \
    --mode multi --adapter_type lora --rank 16 --num_subjects 30

# Budget sweep (LPIPS vs n_candidates)
python experiments/data_extraction.py \
    --mode budget_sweep --max_budget 20 --adapter_type lora
```

### LoRA Rank Ablation (Table 2, Figure 4)

```bash
python experiments/lora_rank_ablation.py \
    --ranks 4 8 16 32 64 \
    --num_subjects 30 \
    --output_dir ./results/rank_ablation
```

### Adapter Comparison (Table 1 all rows, Figure 5)

```bash
python experiments/compare_adapters.py \
    --lora_dir ./adapters \
    --ti_dir   ./adapters_ti \
    --ranks 4 16 64
```

### Statistical / White-box Analysis (Table 3, Figure 6)

```bash
python experiments/statistical_leakage.py \
    --mode full \
    --lora_dir ./adapters \
    --ti_dir   ./adapters_ti \
    --ranks 4 8 16 32 64
```

---

## File Structure

```
.
├── README.md
├── requirements.txt
├── data_preparation.py          # Dataset setup (CelebA / DreamBooth / synthetic)
├── train_lora_adapters.py       # Victim LoRA training
├── train_textual_inversion.py   # Victim TI training
├── validate_code.py             # Comprehensive test suite
├── run_all_experiments.sh       # One-shot full reproduction
│
├── models/
│   ├── diffusion_wrapper.py     # SD 1.5 wrapper + apply/remove LoRA
│   └── adapter_utils.py         # LoRALayer, LoRAInjector, TI manager
│
├── attacks/
│   ├── membership_inference.py  # Loss-based & likelihood-ratio MIA
│   ├── data_reconstruction.py   # Generation & gradient extraction
│   └── statistical_leakage.py   # Fisher, weight norms, spectral analysis
│
├── utils/
│   ├── metrics.py               # AUC-ROC, LPIPS, PSNR, bootstrap CI
│   ├── visualization.py         # All paper figures (7 plot functions)
│   └── data_utils.py            # Dataset loaders, adapter file helpers
│
├── experiments/
│   ├── membership_inference.py  # MIA pipeline (single/multi/compare)
│   ├── data_extraction.py       # Reconstruction pipeline
│   ├── lora_rank_ablation.py    # Rank ablation study
│   ├── compare_adapters.py      # LoRA vs TI full comparison
│   ├── statistical_leakage.py   # White-box analysis pipeline
│   └── run_all_attacks.py       # Master orchestrator
│
└── configs/
    ├── lora_default.yaml
    └── ti_default.yaml
```

---

## Expected Results

### Table 1 — Attack Success Rates

| Adapter | AUC-ROC | TPR@FPR=1% | LPIPS↓ | PSNR↑ |
|---------|---------|------------|--------|-------|
| LoRA r=4 | 0.67 | 0.12 | 0.42 | 18.2 |
| LoRA r=16 | 0.72 | 0.18 | 0.31 | 20.9 |
| LoRA r=64 | 0.78 | 0.25 | 0.24 | 23.0 |
| **Textual Inversion** | **0.81** | **0.31** | **0.19** | **25.2** |

**Key finding:** TI leaks *more* than LoRA r=64 despite having 530× fewer parameters.

### Table 2 — Spearman Correlation

| Metric pair | ρ | p-value |
|------------|---|---------|
| rank vs LPIPS leakage | **0.83** | 0.002 |
| rank vs AUC-ROC | 0.76 | 0.009 |

---

## Code Validation

```bash
python validate_code.py        # full suite (10 categories)
python validate_code.py --fast # skip slow gradient tests
```

All 10 validation categories:
1. Python syntax (every `.py` file)
2. Internal imports
3. LoRA math (shape, zero-init, scaling)
4. Metrics (AUC, PSNR, bootstrap CI)
5. Data utilities (PIL round-trip, collation)
6. Visualisation (all 7 plot functions)
7. Statistical leakage (weight norms, spectral)
8. File structure (all 22 required files)
9. YAML configs
10. End-to-end smoke test

---

## Citation

```bibtex
@inproceedings{adapters_leak_2026,
  title     = {How Much Do Shared Adapters Leak? Measuring Privacy Risks
               in Personalized Diffusion Model Weights},
  author={Hasan, Mujtaba},

  booktitle = {CVPR 2026 Workshop on Personalization in Generative AI (P13N)},
  year      = {2026},
}
```

---

## License

MIT — for research purposes only.
=======


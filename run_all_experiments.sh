#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_all_experiments.sh
# Master script — reproduces all paper results in one shot.
#
# P13N@CVPR 2026:
#   "How Much Do Shared Adapters Leak? Measuring Privacy Risks
#    in Personalized Diffusion Model Weights"
#
# Estimated runtime (V100 GPU, 30 subjects):
#   Data preparation    :  ~2 min
#   LoRA training       :  ~30 min/subject × 5 ranks × 30 subjects  ≈ 75 h
#   TI training         :  ~10 min/subject × 30 subjects             ≈  5 h
#   MIA evaluation      :  ~5 min/subject × 6 configs × 30 subjects  ≈ 15 h
#   Reconstruction      :  ~8 min/subject × 6 configs × 30 subjects  ≈ 24 h
#   Total               :  ~120 h  (with pre-trained adapters: ~40 h)
#
# Usage:
#   chmod +x run_all_experiments.sh
#   ./run_all_experiments.sh
#
#   # Or override defaults:
#   DATA_DIR=./mydata LORA_DIR=./myadapters ./run_all_experiments.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Configurable paths (override via environment variables) ───────────────────
DATA_DIR="${DATA_DIR:-./data}"
LORA_DIR="${LORA_DIR:-./adapters}"
TI_DIR="${TI_DIR:-./adapters_ti}"
RESULTS_DIR="${RESULTS_DIR:-./results}"
MODEL_ID="${MODEL_ID:-runwayml/stable-diffusion-v1-5}"
DEVICE="${DEVICE:-cuda}"

# ── Hyper-parameters ──────────────────────────────────────────────────────────
NUM_SUBJECTS="${NUM_SUBJECTS:-30}"
RANKS="${RANKS:-4 8 16 32 64}"
N_CANDIDATES="${N_CANDIDATES:-10}"
NUM_STEPS="${NUM_STEPS:-50}"
NUM_TIMESTEPS="${NUM_TIMESTEPS:-20}"
SCORER="${SCORER:-likelihood_ratio}"

# ── Skip flags (set to 1 to skip a stage) ────────────────────────────────────
SKIP_DATA="${SKIP_DATA:-0}"
SKIP_TRAIN_LORA="${SKIP_TRAIN_LORA:-0}"
SKIP_TRAIN_TI="${SKIP_TRAIN_TI:-0}"
SKIP_MIA="${SKIP_MIA:-0}"
SKIP_RECON="${SKIP_RECON:-0}"
SKIP_ABLATION="${SKIP_ABLATION:-0}"
SKIP_COMPARE="${SKIP_COMPARE:-0}"

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Privacy Leakage in Personalized Diffusion Adapters"
echo "  P13N@CVPR 2026 — Full Reproduction Pipeline"
echo "══════════════════════════════════════════════════════════════"
echo ""
echo "  DATA_DIR    = $DATA_DIR"
echo "  LORA_DIR    = $LORA_DIR"
echo "  TI_DIR      = $TI_DIR"
echo "  RESULTS_DIR = $RESULTS_DIR"
echo "  DEVICE      = $DEVICE"
echo "  NUM_SUBJECTS= $NUM_SUBJECTS"
echo "  RANKS       = $RANKS"
echo ""

mkdir -p "$RESULTS_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# Stage 0: Environment check
# ─────────────────────────────────────────────────────────────────────────────
echo "── Stage 0: Environment ─────────────────────────────────────"
python -c "
import sys, torch
print(f'  Python   : {sys.version.split()[0]}')
print(f'  PyTorch  : {torch.__version__}')
print(f'  CUDA     : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU      : {torch.cuda.get_device_name(0)}')
    print(f'  VRAM     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
import diffusers, transformers
print(f'  diffusers: {diffusers.__version__}')
print(f'  transformers: {transformers.__version__}')
"

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Data preparation
# ─────────────────────────────────────────────────────────────────────────────
if [ "$SKIP_DATA" -eq 0 ]; then
    echo ""
    echo "── Stage 1: Data Preparation ────────────────────────────────"
    if [ -f "$DATA_DIR/splits.json" ]; then
        echo "  Splits found — skipping."
    else
        python data_preparation.py \
            --dataset synthetic \
            --num_subjects "$NUM_SUBJECTS" \
            --num_reference 500 \
            --num_shadow 5 \
            --output_dir "$DATA_DIR" \
            --seed 42
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Train LoRA adapters (all ranks)
# ─────────────────────────────────────────────────────────────────────────────
if [ "$SKIP_TRAIN_LORA" -eq 0 ]; then
    echo ""
    echo "── Stage 2: Train LoRA Adapters ─────────────────────────────"
    for rank in $RANKS; do
        echo "  Training rank=$rank …"
        python train_lora_adapters.py \
            --data_dir "$DATA_DIR" \
            --output_dir "$LORA_DIR" \
            --rank "$rank" \
            --alpha 1.0 \
            --num_epochs 100 \
            --lr 1e-4 \
            --num_subjects "$NUM_SUBJECTS" \
            --device "$DEVICE"
    done
fi

# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: Train Textual Inversion adapters
# ─────────────────────────────────────────────────────────────────────────────
if [ "$SKIP_TRAIN_TI" -eq 0 ]; then
    echo ""
    echo "── Stage 3: Train Textual Inversion Adapters ────────────────"
    python train_textual_inversion.py \
        --data_dir "$DATA_DIR" \
        --output_dir "$TI_DIR" \
        --num_epochs 500 \
        --lr 5e-4 \
        --num_subjects "$NUM_SUBJECTS" \
        --device "$DEVICE"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Stage 4: Membership Inference Attack  (all adapters)
# ─────────────────────────────────────────────────────────────────────────────
if [ "$SKIP_MIA" -eq 0 ]; then
    echo ""
    echo "── Stage 4: Membership Inference Attack ─────────────────────"

    # LoRA — each rank
    for rank in $RANKS; do
        echo "  MIA: LoRA rank=$rank …"
        python experiments/membership_inference.py \
            --mode multi \
            --data_dir "$DATA_DIR" \
            --adapter_dir "$LORA_DIR" \
            --adapter_type lora \
            --rank "$rank" \
            --num_subjects "$NUM_SUBJECTS" \
            --scorer "$SCORER" \
            --num_timesteps "$NUM_TIMESTEPS" \
            --output_dir "$RESULTS_DIR/mia_lora_rank${rank}" \
            --device "$DEVICE"
    done

    # Textual Inversion
    echo "  MIA: Textual Inversion …"
    python experiments/membership_inference.py \
        --mode multi \
        --data_dir "$DATA_DIR" \
        --adapter_dir "$TI_DIR" \
        --adapter_type textual_inversion \
        --num_subjects "$NUM_SUBJECTS" \
        --scorer "$SCORER" \
        --num_timesteps "$NUM_TIMESTEPS" \
        --output_dir "$RESULTS_DIR/mia_ti" \
        --device "$DEVICE"

    # Overlay ROC (Figure 1)
    echo "  Generating Figure 1 (ROC overlay) …"
    python experiments/membership_inference.py \
        --mode compare \
        --result_files \
            "$RESULTS_DIR/mia_lora_rank4/mia_multi_lora.json" \
            "$RESULTS_DIR/mia_lora_rank16/mia_multi_lora.json" \
            "$RESULTS_DIR/mia_lora_rank64/mia_multi_lora.json" \
            "$RESULTS_DIR/mia_ti/mia_multi_textual_inversion.json" \
        --result_labels \
            "LoRA r=4" "LoRA r=16" "LoRA r=64" "TI" \
        --output_dir "$RESULTS_DIR/figure1" \
        --device "$DEVICE"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Stage 5: Data Reconstruction Attack
# ─────────────────────────────────────────────────────────────────────────────
if [ "$SKIP_RECON" -eq 0 ]; then
    echo ""
    echo "── Stage 5: Data Reconstruction Attack ──────────────────────"

    for rank in $RANKS; do
        echo "  Reconstruction: LoRA rank=$rank …"
        python experiments/data_extraction.py \
            --mode multi \
            --data_dir "$DATA_DIR" \
            --adapter_dir "$LORA_DIR" \
            --adapter_type lora \
            --rank "$rank" \
            --num_subjects "$NUM_SUBJECTS" \
            --n_candidates "$N_CANDIDATES" \
            --num_steps "$NUM_STEPS" \
            --output_dir "$RESULTS_DIR/recon_lora_rank${rank}" \
            --device "$DEVICE"
    done

    echo "  Reconstruction: Textual Inversion …"
    python experiments/data_extraction.py \
        --mode multi \
        --data_dir "$DATA_DIR" \
        --adapter_dir "$TI_DIR" \
        --adapter_type textual_inversion \
        --num_subjects "$NUM_SUBJECTS" \
        --n_candidates "$N_CANDIDATES" \
        --num_steps "$NUM_STEPS" \
        --output_dir "$RESULTS_DIR/recon_ti" \
        --device "$DEVICE"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Stage 6: Rank Ablation  (Figure 4, Table 2)
# ─────────────────────────────────────────────────────────────────────────────
if [ "$SKIP_ABLATION" -eq 0 ]; then
    echo ""
    echo "── Stage 6: LoRA Rank Ablation ──────────────────────────────"
    python experiments/lora_rank_ablation.py \
        --data_dir "$DATA_DIR" \
        --adapters_root "$LORA_DIR" \
        --ranks $RANKS \
        --num_subjects "$NUM_SUBJECTS" \
        --n_candidates "$N_CANDIDATES" \
        --num_steps "$NUM_STEPS" \
        --num_timesteps "$NUM_TIMESTEPS" \
        --scorer "$SCORER" \
        --output_dir "$RESULTS_DIR/rank_ablation" \
        --device "$DEVICE"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Stage 7: Embedding Inversion Attack  (TI adapters only)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "── Stage 7: Embedding Inversion Attack ──────────────────────"
python experiments/embedding_inversion.py \
    --mode multi \
    --data_dir "$DATA_DIR" \
    --ti_dir "$TI_DIR" \
    --num_subjects "$NUM_SUBJECTS" \
    --n_candidates "$N_CANDIDATES" \
    --num_steps "$NUM_STEPS" \
    --output_dir "$RESULTS_DIR/embedding_inv" \
    --device "$DEVICE"

# ─────────────────────────────────────────────────────────────────────────────
# Stage 8: Full Comparison  (Table 1, Figure 5)
# ─────────────────────────────────────────────────────────────────────────────
if [ "$SKIP_COMPARE" -eq 0 ]; then
    echo ""
    echo "── Stage 8: Adapter Comparison ──────────────────────────────"
    python experiments/compare_adapters.py \
        --data_dir "$DATA_DIR" \
        --lora_dir "$LORA_DIR" \
        --ti_dir "$TI_DIR" \
        --ranks 4 16 64 \
        --num_subjects "$NUM_SUBJECTS" \
        --n_candidates "$N_CANDIDATES" \
        --num_steps "$NUM_STEPS" \
        --num_timesteps "$NUM_TIMESTEPS" \
        --scorer "$SCORER" \
        --output_dir "$RESULTS_DIR/comparison" \
        --device "$DEVICE"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  ✓ All Experiments Complete"
echo "══════════════════════════════════════════════════════════════"
echo ""
echo "  Key outputs:"
echo ""
echo "  Table 1:  $RESULTS_DIR/comparison/comparison_summary.json"
echo "  Table 2:  $RESULTS_DIR/rank_ablation/rank_ablation_summary.json"
echo "  Figure 1: $RESULTS_DIR/figure1/figure1_mia_roc.pdf"
echo "  Figure 4: $RESULTS_DIR/rank_ablation/figure4_rank_ablation.pdf"
echo "  Figure 5: $RESULTS_DIR/comparison/figure5_bar_comparison.pdf"
echo "  Emb. Inv: $RESULTS_DIR/embedding_inv/embedding_inv_results.json"
echo ""
echo "  Per-experiment JSON files:"
ls -1 "$RESULTS_DIR"/*/*.json 2>/dev/null | head -20
echo ""

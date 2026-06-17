"""
colab_quickstart.py
────────────────────
Google Colab quickstart — copy each block into a Colab cell and run top-to-bottom.

Covers:
  A.  Install + mount Drive
  B.  Prepare synthetic dataset
  C.  Train ONE LoRA adapter (rank=16, 50 epochs) on 3 images — smoke test
  D.  Run MIA scoring (loss-based, no shadow models needed)
  E.  Run reconstruction (3 candidates, 20 DDIM steps)
  F.  Plot ROC + reconstruction grid
  G.  Print summary table

GPU recommended (T4 free tier works for C–E if you reduce steps).
Time estimate on T4:
  B = 30 s
  C = 8 min  (50 epochs, 3 images)
  D = 3 min  (20 timesteps × 6 images)
  E = 4 min  (3 candidates × 20 steps × 3 images)
  F = 10 s
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL A — Install & mount Drive                             ║
# ╚══════════════════════════════════════════════════════════════╝
"""
!pip install -q diffusers transformers accelerate safetensors lpips scikit-learn
!pip install -q matplotlib seaborn tqdm scipy PyYAML
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive   # or wherever you cloned the repo
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL B — Synthetic dataset                                 ║
# ╚══════════════════════════════════════════════════════════════╝
CELL_B = """
import subprocess, sys

# Run data preparation (synthetic, 3 subjects)
result = subprocess.run(
    [sys.executable, "data_preparation.py",
     "--dataset", "synthetic",
     "--num_subjects", "3",
     "--num_reference", "30",
     "--num_shadow", "2",
     "--output_dir", "./colab_data",
     "--seed", "42"],
    capture_output=True, text=True
)
print(result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr)
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL C — Train one LoRA adapter (rank=16, subject=0)      ║
# ╚══════════════════════════════════════════════════════════════╝
CELL_C = """
import subprocess, sys

result = subprocess.run(
    [sys.executable, "train_lora_adapters.py",
     "--data_dir",    "./colab_data",
     "--output_dir",  "./colab_adapters",
     "--rank",        "16",
     "--num_epochs",  "50",          # 50 for speed; use 200 for paper quality
     "--subject_id",  "0",
     "--device",      "cuda"],
    capture_output=True, text=True
)
print(result.stdout[-3000:])   # last 3000 chars
if result.returncode != 0:
    print("STDERR:", result.stderr[-1000:])
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL D — Membership Inference Attack                       ║
# ╚══════════════════════════════════════════════════════════════╝
CELL_D = """
import sys, torch
sys.path.insert(0, ".")

from models.diffusion_wrapper import DiffusionWrapper
from attacks.membership_inference import score_likelihood_ratio, evaluate_mia
from utils.data_utils import SubjectLoader, load_adapter_state, find_adapter_path
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Load model
wrapper = DiffusionWrapper(device=device,
                           dtype=torch.float16 if device=="cuda" else torch.float32)

# Load data
loader = SubjectLoader("./colab_data", subject_id="0", max_images=5)
m_imgs, m_pr, nm_imgs, nm_pr = loader.all_splits()
print(f"Members: {len(m_imgs)},  Non-members: {len(nm_imgs)}")

# Find adapter
adapter_path = find_adapter_path(Path("./colab_adapters"), "lora", "0", rank=16)
print(f"Adapter: {adapter_path}")
lora_state = load_adapter_state(adapter_path, "lora")

# Score (likelihood-ratio)
print("Scoring member images...")
m_scores  = score_likelihood_ratio(wrapper, m_imgs,  m_pr,  lora_state, "lora", num_timesteps=20)

print("Scoring non-member images...")
nm_scores = score_likelihood_ratio(wrapper, nm_imgs, nm_pr, lora_state, "lora", num_timesteps=20)

# Evaluate
metrics = evaluate_mia(m_scores, nm_scores, higher_is_member=True)
print(f"\\nMIA Results:")
print(f"  AUC-ROC:         {metrics['auc_roc']:.4f}")
print(f"  TPR@FPR=1%:      {metrics['tpr_at_fpr_001']:.4f}")
print(f"  Max Advantage:   {metrics['max_advantage']:.4f}")
print(f"  Member scores:   μ={m_scores.mean():.3f} σ={m_scores.std():.3f}")
print(f"  Nonmember scores:μ={nm_scores.mean():.3f} σ={nm_scores.std():.3f}")
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL E — Data Reconstruction Attack                        ║
# ╚══════════════════════════════════════════════════════════════╝
CELL_E = """
# Assumes wrapper, lora_state, loader are defined from Cell D
from attacks.data_reconstruction import (
    attack_prompt_generation, ReconstructionEvaluator
)
import numpy as np, torch

evaluator = ReconstructionEvaluator(device=device)
prompt    = loader.prompt

# Ground-truth member images (first 3)
gt_imgs = m_imgs[:3]

print("Generating candidates (3 per target image, 20 DDIM steps)...")
candidates = attack_prompt_generation(
    wrapper, prompt, lora_state, "lora",
    n_candidates=3, num_steps=20,
)

results = []
best_recons = []
for gt in gt_imgs:
    gt01 = ((gt + 1) / 2).clamp(0, 1).cpu()
    import torch.nn.functional as F
    mse_vals = [F.mse_loss(c.cpu(), gt01).item() for c in candidates]
    best = candidates[int(np.argmin(mse_vals))]
    results.append(evaluator.evaluate(best, gt))
    best_recons.append(best)

avg = {k: float(np.mean([r[k] for r in results])) for k in results[0]}
print(f"\\nReconstruction Results (best-of-3, 3 images):")
print(f"  LPIPS: {avg['lpips']:.4f}  (↓ lower = better reconstruction = more leakage)")
print(f"  MSE:   {avg['mse']:.4f}")
print(f"  PSNR:  {avg['psnr']:.2f} dB")
print(f"  SSIM:  {avg['ssim']:.4f}")
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL F — Visualise                                         ║
# ╚══════════════════════════════════════════════════════════════╝
CELL_F = """
# Assumes m_scores, nm_scores, gt_imgs, best_recons, results from Cells D+E
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.visualization import (
    plot_roc, plot_score_distributions, plot_reconstruction_grid
)

# ROC
fig1 = plot_roc({"LoRA r=16 (subject 0)": (m_scores, nm_scores)},
                title="MIA ROC Curve — Colab Quickstart")
fig1.savefig("quickstart_roc.pdf", bbox_inches="tight")
print("Saved quickstart_roc.pdf")

# Score distributions
fig2 = plot_score_distributions(m_scores, nm_scores,
                                title="Likelihood-Ratio Score Distributions")
fig2.savefig("quickstart_scores.pdf", bbox_inches="tight")
print("Saved quickstart_scores.pdf")

# Reconstruction grid
fig3 = plot_reconstruction_grid(
    [g.cpu() for g in gt_imgs],
    best_recons,
    results,
    n_show=3,
    title="Reconstruction Attack — Colab Quickstart",
)
fig3.savefig("quickstart_recon.pdf", bbox_inches="tight")
print("Saved quickstart_recon.pdf")

plt.close("all")
print("\\n✓ Figures saved to current directory.")
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL G — Summary                                           ║
# ╚══════════════════════════════════════════════════════════════╝
CELL_G = """
from utils.metrics import compute_mia_metrics, compute_privacy_score, format_table

full_metrics = compute_mia_metrics(m_scores, nm_scores, higher_is_member=True)
ps = compute_privacy_score(full_metrics, {"lpips_mean": avg["lpips"]})

tbl = format_table({
    "LoRA r=16 (subj 0)": {
        "AUC-ROC":     full_metrics["auc_roc"],
        "TPR@1%FPR":   full_metrics["tpr_at_fpr_001"],
        "Advantage":   full_metrics["max_advantage"],
        "LPIPS ↓":     avg["lpips"],
        "PSNR ↑ (dB)": avg["psnr"],
        "SSIM ↑":      avg["ssim"],
        "Privacy Score": ps,
    }
}, title="Quickstart Results — Single Subject")
print(tbl)
"""

# ──────────────────────────────────────────────────────────────────────────────
# Standalone mode: print all cells
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    divider = "\n" + "─" * 65 + "\n"
    print("colab_quickstart.py — copy each block into a Colab cell\n")
    for name, code in [
        ("CELL A  Install & Mount Drive",              "# (see docstring at top of file)"),
        ("CELL B  Synthetic Dataset",                  CELL_B),
        ("CELL C  Train LoRA Adapter (rank=16, 50ep)", CELL_C),
        ("CELL D  Membership Inference Attack",        CELL_D),
        ("CELL E  Data Reconstruction Attack",         CELL_E),
        ("CELL F  Visualise Results",                  CELL_F),
        ("CELL G  Summary Table",                      CELL_G),
    ]:
        print(f"{'#'*65}\n# {name}\n{'#'*65}")
        print(code.strip())
        print(divider)

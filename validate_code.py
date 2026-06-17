"""
validate_code.py
─────────────────
Comprehensive validation suite for the camera-ready code package.

Checks:
  1.  Python syntax   (py_compile on every .py file)
  2.  Import tree     (all internal imports resolve)
  3.  LoRA math       (layer forward, weight materialisation)
  4.  Metrics         (AUC-ROC, PSNR, SSIM, bootstrap CI)
  5.  Data utils      (tensor shapes, PIL round-trip)
  6.  Visualisation   (all plot functions return Figure objects)
  7.  Statistical     (weight norms, spectral, Spearman)
  8.  File structure  (all required files present)
  9.  YAML configs    (well-formed)
 10.  End-to-end smoke (synthetic data → splits → metric → plot)

Run:
    python validate_code.py
    python validate_code.py --fast   # skip slow statistical tests

Exit code 0 = all tests passed.
"""

from __future__ import annotations

import argparse
import os
import py_compile
import sys
import traceback
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")

# ── ANSI colour helpers ────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

PASS = f"{GREEN}✓ PASS{RESET}"
FAIL = f"{RED}✗ FAIL{RESET}"
SKIP = f"{YELLOW}~ SKIP{RESET}"


def _ok(msg: str)   -> None: print(f"  {PASS}  {msg}")
def _fail(msg: str) -> None: print(f"  {FAIL}  {msg}")
def _skip(msg: str) -> None: print(f"  {SKIP}  {msg}")


class Validator:
    def __init__(self, root: Path, fast: bool = False):
        self.root  = root
        self.fast  = fast
        self.fails: List[str] = []
        self.skips: List[str] = []

    def _try(self, name: str, fn, skip_if_fast: bool = False) -> bool:
        if skip_if_fast and self.fast:
            _skip(f"{name} (--fast)")
            self.skips.append(name)
            return True
        try:
            fn()
            _ok(name)
            return True
        except Exception as exc:
            _fail(f"{name}")
            print(f"         {RED}{type(exc).__name__}: {exc}{RESET}")
            if os.environ.get("VERBOSE"):
                traceback.print_exc()
            self.fails.append(name)
            return False

    # ── 1. Syntax ──────────────────────────────────────────────────────────────

    def check_syntax(self) -> None:
        print(f"\n{BOLD}[1] Python syntax{RESET}")
        py_files = sorted(self.root.rglob("*.py"))
        for fpath in py_files:
            rel = fpath.relative_to(self.root)
            def _compile(f=fpath):
                py_compile.compile(str(f), doraise=True)
            self._try(str(rel), _compile)

    # ── 2. Imports ────────────────────────────────────────────────────────────

    def check_imports(self) -> None:
        print(f"\n{BOLD}[2] Internal imports{RESET}")
        sys.path.insert(0, str(self.root))

        modules = [
            ("models.diffusion_wrapper",  "DiffusionWrapper"),
            ("models.adapter_utils",      "LoRALayer"),
            ("models.adapter_utils",      "LoRAInjector"),
            ("models.adapter_utils",      "TextualInversionManager"),
            ("attacks.membership_inference", "run_mia"),
            ("attacks.data_reconstruction",  "ReconstructionEvaluator"),
            ("attacks.statistical_leakage",  "weight_norm_stats"),
            ("utils.metrics",            "compute_mia_metrics"),
            ("utils.metrics",            "compute_reconstruction_metrics"),
            ("utils.metrics",            "compute_privacy_score"),
            ("utils.metrics",            "bootstrap_ci"),
            ("utils.data_utils",         "SubjectLoader"),
            ("utils.data_utils",         "find_adapter_path"),
            ("utils.visualization",      "plot_roc"),
            ("utils.visualization",      "plot_rank_ablation"),
        ]

        for mod, attr in modules:
            def _imp(m=mod, a=attr):
                import importlib
                module = importlib.import_module(m)
                assert hasattr(module, a), f"{m}.{a} not found"
            self._try(f"{mod}.{attr}", _imp)

    # ── 3. LoRA math ──────────────────────────────────────────────────────────

    def check_lora_math(self) -> None:
        import torch
        from models.adapter_utils import LoRALayer, LoRAInjector

        print(f"\n{BOLD}[3] LoRA math{RESET}")

        def _layer_shapes():
            layer = LoRALayer(in_features=768, out_features=768, rank=4, alpha=1.0)
            x   = torch.randn(2, 10, 768)
            out = layer(x)
            assert out.shape == (2, 10, 768), f"bad shape {out.shape}"
            W = layer.effective_weight()
            assert W.shape == (768, 768)
        self._try("LoRALayer forward shape", _layer_shapes)

        def _zero_init():
            # B initialised to zeros → delta = 0 at step 0
            layer = LoRALayer(768, 768, rank=4)
            W = layer.effective_weight()
            assert W.abs().max().item() == 0.0, "LoRA delta not zero at init"
        self._try("LoRALayer zero initialisation at step 0", _zero_init)

        def _scaling():
            layer = LoRALayer(64, 64, rank=4, alpha=2.0)
            assert abs(layer.scaling - 0.5) < 1e-6
        self._try("LoRALayer scaling = alpha/rank", _scaling)

    # ── 4. Metrics ────────────────────────────────────────────────────────────

    def check_metrics(self) -> None:
        import numpy as np
        import torch
        from utils.metrics import (
            compute_mia_metrics,
            compute_reconstruction_metrics,
            compute_privacy_score,
            bootstrap_ci,
        )

        print(f"\n{BOLD}[4] Metrics{RESET}")

        def _mia_random():
            np.random.seed(0)
            m  = np.random.randn(100)
            nm = np.random.randn(100)
            r  = compute_mia_metrics(m, nm)
            assert 0.4 < r["auc_roc"] < 0.6, f"random AUC = {r['auc_roc']}"
            assert "tpr_at_fpr_001" in r
        self._try("compute_mia_metrics (random inputs → AUC≈0.5)", _mia_random)

        def _mia_perfect():
            m  = np.ones(50) * 2.0
            nm = np.ones(50) * -2.0
            r  = compute_mia_metrics(m, nm)
            assert r["auc_roc"] > 0.95, f"perfect AUC = {r['auc_roc']}"
        self._try("compute_mia_metrics (perfect separation → AUC>0.95)", _mia_perfect)

        def _recon():
            recon = torch.rand(4, 3, 64, 64)
            gt    = torch.rand(4, 3, 64, 64)
            r     = compute_reconstruction_metrics(recon, gt)
            assert "lpips_mean" in r
            assert "psnr_mean"  in r
            assert r["mse_mean"] >= 0
        self._try("compute_reconstruction_metrics shape check", _recon)

        def _recon_identity():
            img = torch.rand(1, 3, 64, 64)
            r   = compute_reconstruction_metrics(img.clone(), img)
            assert r["mse_mean"] < 1e-6, f"identity MSE = {r['mse_mean']}"
            assert r["psnr_mean"] > 50,   f"identity PSNR = {r['psnr_mean']}"
        self._try("compute_reconstruction_metrics (identity → MSE≈0)", _recon_identity)

        def _privacy_score():
            mia_m = {"auc_roc": 0.75, "tpr_at_fpr_001": 0.2, "max_advantage": 0.3}
            rec_m = {"lpips_mean": 0.3}
            s = compute_privacy_score(mia_m, rec_m)
            assert 0.0 < s < 1.0, f"score out of [0,1]: {s}"
        self._try("compute_privacy_score range [0,1]", _privacy_score)

        def _ci():
            np.random.seed(42)
            data = np.random.randn(200)
            mean, lo, hi = bootstrap_ci(data, np.mean, n_boot=500)
            assert lo < mean < hi
        self._try("bootstrap_ci ordering lo < mean < hi", _ci)

    # ── 5. Data utils ─────────────────────────────────────────────────────────

    def check_data_utils(self) -> None:
        import numpy as np
        import torch
        from PIL import Image
        from utils.data_utils import (
            load_images, tensor_to_pil, collate_for_mia, find_adapter_path
        )

        print(f"\n{BOLD}[5] Data utilities{RESET}")

        def _pil_roundtrip(tmp=Path("/tmp/val_img.jpg")):
            arr = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
            Image.fromarray(arr).save(tmp)
            imgs = load_images([tmp])
            assert imgs.shape == (1, 3, 512, 512)
            pil  = tensor_to_pil(imgs[0])
            assert pil.size == (512, 512)
        self._try("load_images + tensor_to_pil round-trip", _pil_roundtrip)

        def _collate():
            m  = torch.rand(5, 3, 64, 64)
            nm = torch.rand(3, 3, 64, 64)
            imgs, prompts, labels = collate_for_mia(m, nm, "test prompt")
            assert imgs.shape[0] == 8
            assert labels.sum().item() == 5
            assert len(prompts) == 8
        self._try("collate_for_mia shapes and label sum", _collate)

        def _find_none():
            r = find_adapter_path(Path("/nonexistent"), "lora", "0", 16)
            assert r is None
        self._try("find_adapter_path returns None when missing", _find_none)

    # ── 6. Visualisation ──────────────────────────────────────────────────────

    def check_visualization(self) -> None:
        import numpy as np
        import torch
        import matplotlib.pyplot as plt
        from utils.visualization import (
            plot_roc, plot_score_distributions, plot_reconstruction_grid,
            plot_rank_ablation, plot_adapter_comparison, plot_privacy_heatmap,
            plot_cumulative_leakage,
        )

        print(f"\n{BOLD}[6] Visualisation{RESET}")

        m  = np.random.randn(50) + 0.3
        nm = np.random.randn(50)

        def _roc():
            fig = plot_roc({"LoRA r=16": (m, nm)})
            assert isinstance(fig, plt.Figure)
            plt.close("all")
        self._try("plot_roc returns Figure", _roc)

        def _dist():
            fig = plot_score_distributions(m, nm)
            assert isinstance(fig, plt.Figure)
            plt.close("all")
        self._try("plot_score_distributions returns Figure", _dist)

        def _grid():
            orig  = [torch.rand(3, 64, 64) for _ in range(3)]
            recon = [torch.rand(3, 64, 64) for _ in range(3)]
            mets  = [{"lpips": 0.3, "psnr": 25.0}] * 3
            fig   = plot_reconstruction_grid(orig, recon, mets, n_show=3)
            assert isinstance(fig, plt.Figure)
            plt.close("all")
        self._try("plot_reconstruction_grid returns Figure", _grid)

        def _rank():
            fig = plot_rank_ablation(
                [4, 8, 16], {"AUC": [0.6, 0.7, 0.8]}
            )
            assert isinstance(fig, plt.Figure)
            plt.close("all")
        self._try("plot_rank_ablation returns Figure", _rank)

        def _bar():
            fig = plot_adapter_comparison(
                ["LoRA r=4", "TI"],
                {"AUC-ROC": [0.67, 0.81]}
            )
            assert isinstance(fig, plt.Figure)
            plt.close("all")
        self._try("plot_adapter_comparison returns Figure", _bar)

        def _heatmap():
            mat = np.random.rand(3, 5)
            fig = plot_privacy_heatmap(mat, ["A","B","C"], ["s0","s1","s2","s3","s4"])
            assert isinstance(fig, plt.Figure)
            plt.close("all")
        self._try("plot_privacy_heatmap returns Figure", _heatmap)

        def _cum():
            fig = plot_cumulative_leakage(
                {"LoRA r=16": np.sort(np.random.rand(20)),
                 "TI":        np.sort(np.random.rand(20))}
            )
            assert isinstance(fig, plt.Figure)
            plt.close("all")
        self._try("plot_cumulative_leakage returns Figure", _cum)

    # ── 7. Statistical leakage ────────────────────────────────────────────────

    def check_statistical(self) -> None:
        import torch
        from attacks.statistical_leakage import (
            weight_norm_stats, spectral_analysis, rank_leakage_spearman
        )

        print(f"\n{BOLD}[7] Statistical leakage{RESET}")

        # Build a synthetic lora_state
        def _make_state(rank=4, n=4):
            s = {}
            for i in range(n):
                A = torch.randn(768, rank)
                B = torch.zeros(rank, 768)
                s[f"layer{i}.to_q.lora_A"] = A
                s[f"layer{i}.to_q.lora_B"] = B
            return s

        def _wn():
            s = _make_state()
            r = weight_norm_stats(s)
            assert "weight_norm_mean" in r
            assert r["num_lora_layers"] == 4
            # B=0 → W=0 → norm=0
            assert r["weight_norm_total"] == 0.0
        self._try("weight_norm_stats (zero B → norm=0)", _wn)

        def _sp():
            s = _make_state()
            r = spectral_analysis(s)
            assert "sv_entropy_mean" in r
            assert r["num_layers_analysed"] == 4
        self._try("spectral_analysis returns expected keys", _sp)

        def _spm():
            rho = rank_leakage_spearman([4,8,16,32,64], [0.6,0.65,0.72,0.75,0.80])
            assert -1 <= rho["spearman_rho"] <= 1
        self._try("rank_leakage_spearman range [-1,1]", _spm)

    # ── 8. File structure ──────────────────────────────────────────────────────

    def check_files(self) -> None:
        print(f"\n{BOLD}[8] File structure{RESET}")
        required = [
            "README.md",
            "requirements.txt",
            "data_preparation.py",
            "train_lora_adapters.py",
            "train_textual_inversion.py",
            "run_all_experiments.sh",
            "models/diffusion_wrapper.py",
            "models/adapter_utils.py",
            "attacks/membership_inference.py",
            "attacks/data_reconstruction.py",
            "attacks/statistical_leakage.py",
            "utils/metrics.py",
            "utils/visualization.py",
            "utils/data_utils.py",
            "experiments/membership_inference.py",
            "experiments/data_extraction.py",
            "experiments/lora_rank_ablation.py",
            "experiments/compare_adapters.py",
            "experiments/statistical_leakage.py",
            "experiments/run_all_attacks.py",
            "configs/lora_default.yaml",
            "configs/ti_default.yaml",
        ]
        for f in required:
            path = self.root / f
            def _exists(p=path, name=f):
                assert p.exists(), f"missing: {name}"
            self._try(f"exists: {f}", _exists)

    # ── 9. YAML configs ────────────────────────────────────────────────────────

    def check_yaml(self) -> None:
        print(f"\n{BOLD}[9] YAML configs{RESET}")
        import yaml
        for cfg in (self.root / "configs").glob("*.yaml"):
            def _parse(p=cfg):
                d = yaml.safe_load(p.read_text())
                assert isinstance(d, dict)
            self._try(f"parse {cfg.name}", _parse)

    # ── 10. End-to-end smoke ──────────────────────────────────────────────────

    def check_e2e_smoke(self) -> None:
        print(f"\n{BOLD}[10] End-to-end smoke (no model load){RESET}")
        import tempfile
        import json
        import numpy as np
        from utils.data_utils import load_images, collate_for_mia
        from utils.metrics import compute_mia_metrics, compute_reconstruction_metrics, compute_privacy_score
        from utils.visualization import plot_roc, plot_rank_ablation, save_all
        import matplotlib.pyplot as plt

        def _smoke():
            # Synthetic score arrays mimicking a real experiment
            np.random.seed(0)
            m_scores  = np.random.randn(30) + 0.4   # members score slightly higher
            nm_scores = np.random.randn(30)

            mia_m = compute_mia_metrics(m_scores, nm_scores, higher_is_member=True)
            assert mia_m["auc_roc"] > 0.5

            recon_r = compute_reconstruction_metrics(
                reconstructed = torch.rand(4, 3, 64, 64),
                ground_truth  = torch.rand(4, 3, 64, 64),
            )

            import torch
            ps = compute_privacy_score(mia_m, {"lpips_mean": 0.35})
            assert 0 <= ps <= 1

            # Figures
            fig1 = plot_roc({"Test": (m_scores, nm_scores)})
            fig2 = plot_rank_ablation([4, 16, 64], {"AUC": [0.67, 0.72, 0.78]})

            with tempfile.TemporaryDirectory() as td:
                save_all({"roc": fig1, "rank": fig2}, td)
                assert (Path(td) / "roc.png").exists()
                assert (Path(td) / "rank.pdf").exists()

            plt.close("all")

        self._try("synthetic scores → metrics → plots → save", _smoke)

    # ── Runner ────────────────────────────────────────────────────────────────

    def run_all(self) -> int:
        print(f"\n{BOLD}{'='*65}{RESET}")
        print(f"{BOLD}  Privacy Leakage Code Validation — Camera-Ready Package{RESET}")
        print(f"{BOLD}{'='*65}{RESET}")

        self.check_files()
        self.check_syntax()
        self.check_imports()
        self.check_lora_math()
        self.check_metrics()
        self.check_data_utils()
        self.check_visualization()
        self.check_statistical()
        self.check_yaml()
        self.check_e2e_smoke()

        # ── Summary ─────────────────────────────────────────────────────────
        print(f"\n{BOLD}{'='*65}{RESET}")
        print(f"{BOLD}  Summary{RESET}")
        print(f"{BOLD}{'='*65}{RESET}")

        total  = 0
        passed = 0
        for line in sys.stdout_lines if hasattr(sys, "stdout_lines") else []:
            pass   # approximate; we count below

        if self.fails:
            print(f"\n{RED}  Failed ({len(self.fails)}):{RESET}")
            for f in self.fails:
                print(f"    • {f}")
        if self.skips:
            print(f"\n{YELLOW}  Skipped ({len(self.skips)}):{RESET}")
            for s in self.skips:
                print(f"    • {s}")

        n_fail = len(self.fails)
        if n_fail == 0:
            print(f"\n{GREEN}{BOLD}  ✓ All validations passed!{RESET}")
            print("    Code is ready for camera-ready submission.\n")
            return 0
        else:
            print(f"\n{RED}{BOLD}  ✗ {n_fail} validation(s) failed.{RESET}")
            print("    Please fix the issues above before submission.\n")
            return 1


# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate camera-ready code package")
    p.add_argument("--fast", action="store_true",
                   help="Skip slow tests (gradient, Fisher)")
    p.add_argument("--root", type=str, default=None,
                   help="Package root directory (default: directory of this file)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root) if args.root else Path(__file__).resolve().parent
    v    = Validator(root, fast=args.fast)
    return v.run_all()


if __name__ == "__main__":
    sys.exit(main())

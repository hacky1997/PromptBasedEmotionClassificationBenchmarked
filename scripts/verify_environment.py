#!/usr/bin/env python3
"""
verify_environment.py
─────────────────────
Pre-run environment check for the emotion classification pipeline.
Run this before opening the notebook to confirm all dependencies are
installed at the correct versions and that a GPU is available.

Usage:
    python scripts/verify_environment.py
"""

import sys
import importlib
import subprocess

REQUIRED = {
    "torch":           ("2.2.0",  True),   # (min_version, gpu_check)
    "transformers":    ("4.40.2", False),
    "accelerate":      ("0.29.3", False),
    "datasets":        ("2.19.1", False),
    "tokenizers":      ("0.19.1", False),
    "sklearn":         ("1.4.0",  False),
    "scipy":           ("1.13.0", False),
    "numpy":           ("1.26.0", False),
    "pandas":          ("2.2.0",  False),
    "matplotlib":      ("3.8.0",  False),
    "seaborn":         ("0.13.0", False),
    "tqdm":            ("4.66.0", False),
}

# ── Version comparison ────────────────────────────────────────────────────────
def _ver(v):
    return tuple(int(x) for x in v.split(".")[:3])

def check_package(name, min_ver):
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", "0.0.0")
        ok  = _ver(ver) >= _ver(min_ver)
        return ok, ver
    except ImportError:
        return False, "NOT INSTALLED"

# ── GPU check ─────────────────────────────────────────────────────────────────
def check_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
            return True, f"{name} ({mem:.1f} GB)"
        return False, "No CUDA GPU detected"
    except Exception as e:
        return False, str(e)

# ── Python version ─────────────────────────────────────────────────────────────
def check_python():
    v = sys.version_info
    ok = (v.major, v.minor) >= (3, 10)
    return ok, f"{v.major}.{v.minor}.{v.micro}"

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Environment Verification")
    print("  Prompt-Based Emotion Classification Pipeline")
    print("=" * 60)

    all_ok = True

    # Python
    ok, ver = check_python()
    status = "✓" if ok else "✗"
    print(f"\n  {status}  Python {ver}  (required: ≥ 3.10)")
    if not ok: all_ok = False

    # Packages
    print("\n  Packages:")
    for pkg, (min_ver, _) in REQUIRED.items():
        ok, ver = check_package(pkg, min_ver)
        status  = "✓" if ok else "✗"
        note    = "" if ok else f"  ← install: pip install {pkg}>={min_ver}"
        print(f"    {status}  {pkg:<18} {ver:<12} (required: ≥ {min_ver}){note}")
        if not ok: all_ok = False

    # GPU
    print("\n  GPU:")
    ok, info = check_gpu()
    status = "✓" if ok else "⚠"
    print(f"    {status}  {info}")
    if not ok:
        print("      GPU is not required but training will be ~100× slower on CPU.")

    # HuggingFace connectivity
    print("\n  HuggingFace Hub connectivity:")
    try:
        from huggingface_hub import HfApi
        HfApi().list_datasets(limit=1)
        print("    ✓  HuggingFace Hub reachable")
    except Exception as e:
        print(f"    ⚠  Could not reach HuggingFace Hub: {e}")
        print("      Dataset will be downloaded on first notebook run.")

    # Summary
    print("\n" + "=" * 60)
    if all_ok:
        print("  All checks passed. You are ready to run the notebook.")
    else:
        print("  Some checks failed. Install missing packages:")
        print("    pip install -r requirements.txt")
    print("=" * 60)

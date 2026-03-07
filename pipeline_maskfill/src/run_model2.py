#!/usr/bin/env python3
# ==============================================================================
# run_model2.py — Train Model 2 (Advanced DP-Guided Decoupled Pipeline)
# ==============================================================================
# Kaggle Account 2:  THIS SCRIPT → Train Model 2 only
#
# Model 2 features:
#   - MultiTaskNERModel with Privacy Attention Head (DeBERTa-v3-small)
#   - DP-SGD via Opacus for formal (ε,δ) guarantees
#   - Hallucinator with semantic fidelity loss (Flan-T5-base QLoRA)
#   - Entity consistency module
#
# Usage:
#   !python run_model2.py              # Full run
#   !python run_model2.py --quick      # Quick test
# ==============================================================================

import os, sys, argparse, json, gc, time, torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (
    CFG, log, install_deps, cleanup_gpu,
    load_ai4privacy, language_stratified_split,
    BASE_DIR, preflight_checks, restore_checkpoints_from_input,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train Model 2 — Advanced DP Pipeline")
    p.add_argument("--quick", action="store_true",
                   help="Quick mode: small data subset")
    p.add_argument("--skip-preflight", action="store_true",
                   help="Skip preflight smoke tests")
    return p.parse_args()


def main():
    args = parse_args()
    CFG.QUICK_MODE = args.quick

    log.info("╔══════════════════════════════════════════════════════════════╗")
    log.info("║    MODEL 2 — ADVANCED DP-GUIDED DECOUPLED PIPELINE         ║")
    log.info("╚══════════════════════════════════════════════════════════════╝")
    log.info(f"Quick mode: {CFG.QUICK_MODE}")

    # ── Restore checkpoints from previous runs ──
    restore_checkpoints_from_input()

    # ── Load & split data ──
    ds = load_ai4privacy()
    half_a, half_b, test_set, lang_col = language_stratified_split(ds)
    log.info(f"Data: {len(ds):,} → A={len(half_a):,} B={len(half_b):,} Test={len(test_set):,}")

    # ── Preflight ──
    if not args.skip_preflight:
        preflight_checks([2], ds)
    else:
        log.info("Preflight skipped")

    start = time.time()

    # ── Model 2: Advanced ──
    from model2_advanced import run_advanced
    adv_result = run_advanced(half_a, half_b, test_set, lang_col)
    log.info("Model 2 complete ✓")

    # ── Save results ──
    all_results = {"Model 2 (Advanced)": adv_result["metrics"]}
    summary_path = os.path.join(CFG.OUTPUT_DIR, "results_model2.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    cleanup_gpu()
    elapsed = time.time() - start
    log.info(f"\nTotal runtime: {elapsed/60:.1f} minutes")
    log.info(f"Results saved → {summary_path}")
    log.info("Done ✓")


if __name__ == "__main__":
    main()

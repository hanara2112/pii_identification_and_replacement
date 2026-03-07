#!/usr/bin/env python3
# ==============================================================================
# run_model13.py — Train Models 1 & 3 together in one Kaggle session
# ==============================================================================
# Kaggle Account 2, Session 1:  THIS SCRIPT → Train Model 1 + Model 3
#
# Model 3 reuses Model 1's Censor + Hallucinator, so running them together
# saves the overhead of re-training.
#
# Usage:
#   !python run_model13.py              # Full run
#   !python run_model13.py --quick      # Quick test
# ==============================================================================

import os, sys, argparse, json, gc, time, torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (
    CFG, log, install_deps, cleanup_gpu,
    load_ai4privacy, language_stratified_split,
    BASE_DIR, preflight_checks,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train Models 1 & 3 together")
    p.add_argument("--quick", action="store_true",
                   help="Quick mode: small data subset")
    p.add_argument("--skip-preflight", action="store_true",
                   help="Skip preflight smoke tests")
    return p.parse_args()


def main():
    args = parse_args()
    CFG.QUICK_MODE = args.quick

    log.info("╔══════════════════════════════════════════════════════════════╗")
    log.info("║     MODELS 1 + 3  —  BASELINE + REPHRASER PIPELINE        ║")
    log.info("╚══════════════════════════════════════════════════════════════╝")
    log.info(f"Quick mode: {CFG.QUICK_MODE}")

    # ── Load & split data ──
    ds = load_ai4privacy()
    half_a, half_b, test_set, lang_col = language_stratified_split(ds)
    log.info(f"Data: {len(ds):,} → A={len(half_a):,} B={len(half_b):,} Test={len(test_set):,}")

    # ── Preflight ──
    if not args.skip_preflight:
        preflight_checks([1, 3], ds)
    else:
        log.info("Preflight skipped")

    start = time.time()

    # ── Model 1: Baseline ──
    log.info("\n" + "━" * 70)
    from model1_baseline import run_baseline
    baseline_result = run_baseline(half_a, half_b, test_set, lang_col)
    log.info(f"Model 1 complete ✓")

    # ── Model 3: Rephraser (reuses Model 1 components) ──
    log.info("\n" + "━" * 70)
    from model3_rephraser import run_rephraser_pipeline
    reph_result = run_rephraser_pipeline(
        half_a, half_b, test_set, lang_col,
        baseline_result=baseline_result)
    log.info(f"Model 3 complete ✓")

    # ── Save combined results ──
    all_results = {
        "Model 1 (Baseline)": baseline_result["metrics"],
        "Model 3 (Rephraser)": reph_result["metrics"],
    }

    summary_path = os.path.join(CFG.OUTPUT_DIR, "results_model13.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info(f"\nResults saved → {summary_path}")

    elapsed = time.time() - start
    log.info(f"Total runtime: {elapsed/60:.1f} minutes")

    del baseline_result, reph_result
    cleanup_gpu()
    log.info("Done ✓")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# ==============================================================================
# run_model4.py — Train Model 4 (B-type Semantic Privacy Paraphraser)
# ==============================================================================
# Kaggle Account 3:  THIS SCRIPT → Single-pass Flan-T5-base QLoRA
#
# Model 4 uses BOTH halves (no NER/Halluc split) to train a single end-to-end
# model that learns to simultaneously identify PII and rephrase the text.
#
# Usage:
#   !python run_model4.py              # Full run
#   !python run_model4.py --quick      # Quick test
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
    p = argparse.ArgumentParser(description="Train Model 4 — Semantic Paraphraser")
    p.add_argument("--quick", action="store_true",
                   help="Quick mode: small data subset")
    p.add_argument("--skip-preflight", action="store_true",
                   help="Skip preflight smoke tests")
    return p.parse_args()


def main():
    args = parse_args()
    CFG.QUICK_MODE = args.quick

    log.info("╔══════════════════════════════════════════════════════════════╗")
    log.info("║    MODEL 4 — B-TYPE: SINGLE-PASS SEMANTIC PARAPHRASER      ║")
    log.info("╚══════════════════════════════════════════════════════════════╝")
    log.info(f"Quick mode: {CFG.QUICK_MODE}")

    # ── Load & split data ──
    ds = load_ai4privacy()
    half_a, half_b, test_set, lang_col = language_stratified_split(ds)
    log.info(f"Data: {len(ds):,} → A={len(half_a):,} B={len(half_b):,} Test={len(test_set):,}")

    # ── Preflight ──
    if not args.skip_preflight:
        preflight_checks([4], ds)
    else:
        log.info("Preflight skipped")

    start = time.time()

    # ── Model 4: Semantic Paraphraser ──
    from model4_semantic import run_semantic_pipeline
    sem_result = run_semantic_pipeline(half_a, half_b, test_set, lang_col)
    log.info("Model 4 complete ✓")

    # ── Save results ──
    all_results = {"Model 4 (Semantic)": sem_result["metrics"]}
    summary_path = os.path.join(CFG.OUTPUT_DIR, "results_model4.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    cleanup_gpu()
    elapsed = time.time() - start
    log.info(f"\nTotal runtime: {elapsed/60:.1f} minutes")
    log.info(f"Results saved → {summary_path}")
    log.info("Done ✓")


if __name__ == "__main__":
    main()

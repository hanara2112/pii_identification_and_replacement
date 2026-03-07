#!/usr/bin/env python3
# ==============================================================================
# run_all.py — Master Orchestrator for 4-Model Privacy Pipeline
# ==============================================================================
# Usage:
#   python run_all.py                       # Run all 4 models
#   python run_all.py --model 1             # Baseline only
#   python run_all.py --model 3             # A-type Rephraser only
#   python run_all.py --model 1 2 3 4       # All four
#   python run_all.py --quick               # Quick mode (subset of data)
#
# On Kaggle: Upload all .py files, then run cells:
#   !python run_all.py --quick              # Test run
#   !python run_all.py                      # Full run
# ==============================================================================

import os, sys, argparse, json, gc, torch, time
import numpy as np

# Ensure local imports resolve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (
    CFG, log, install_deps, cleanup_gpu,
    load_ai4privacy, language_stratified_split,
    plot_comparison, BASE_DIR, preflight_checks,
)


def parse_args():
    p = argparse.ArgumentParser(description="4-Model Privacy Pipeline")
    p.add_argument("--model", nargs="*", type=int, default=[1, 2, 3, 4],
                   choices=[1, 2, 3, 4],
                   help="Which model(s) to run (default: all)")
    p.add_argument("--quick", action="store_true",
                   help="Quick mode: small data subset for testing")
    p.add_argument("--skip-preflight", action="store_true",
                   help="Skip preflight smoke tests (use after first verified run)")
    return p.parse_args()


def print_comparison_table(all_results):
    """Print a formatted comparison table of all model results."""
    log.info("\n" + "=" * 80)
    log.info(" COMPARATIVE RESULTS")
    log.info("=" * 80)

    header = f"{'Model':<30} {'Leak%':>7} {'CRR%':>7} {'ROUGE-L':>8} {'BLEU':>7} {'BERTSc':>7}"
    log.info(header)
    log.info("-" * 80)
    for name, r in all_results.items():
        leak = r.get("leakage", {}).get("entity_leak_rate", -1)
        crr = r.get("crr", {}).get("crr", -1)
        rl = r.get("rouge", {}).get("rougeL", -1)
        bleu = r.get("bleu", -1)
        bs = r.get("bertscore_f1", -1)
        row = f"{name:<30} {leak:>7.2f} {crr:>7.2f} {rl:>8.4f} {bleu:>7.2f} {bs:>7.4f}"
        log.info(row)
    log.info("=" * 80)
    log.info("  ↓ = lower is better (Leak%, CRR%)  |  ↑ = higher is better (ROUGE, BLEU, BERTSc)")
    log.info("")

    # Curated comparison
    log.info("Curated Evaluation:")
    header2 = f"{'Model':<30} {'PII%':>7} {'NoPII%':>8} {'Context%':>9}"
    log.info(header2)
    log.info("-" * 60)
    for name, r in all_results.items():
        c = r.get("curated", {})
        pii = c.get("pii_rate", -1)
        nopii = c.get("nopii_rate", -1)
        ctx = c.get("contextual_rate", -1)
        row = f"{name:<30} {pii:>7.1f} {nopii:>8.1f} {ctx:>9.1f}"
        log.info(row)
    log.info("=" * 80)


def main():
    args = parse_args()
    CFG.QUICK_MODE = args.quick
    models_to_run = sorted(set(args.model))

    log.info("╔══════════════════════════════════════════════════════════════╗")
    log.info("║   PRIVACY-PRESERVING TEXT ANONYMISATION — 4-MODEL PIPELINE ║")
    log.info("╚══════════════════════════════════════════════════════════════╝")
    log.info(f"Models: {models_to_run}  |  Quick: {CFG.QUICK_MODE}")

    # ── Load & split data (shared across all models) ──
    ds = load_ai4privacy()
    half_a, half_b, test_set, lang_col = language_stratified_split(ds)
    log.info(f"Data: {len(ds):,} total → A={len(half_a):,} B={len(half_b):,} Test={len(test_set):,}")

    # ── Preflight: smoke-test models, AMP, tokenisation before real training ──
    if not args.skip_preflight:
        preflight_checks(models_to_run, ds)
    else:
        log.info("Preflight checks skipped (--skip-preflight)")

    all_results = {}
    baseline_result = None
    start_time = time.time()

    # ── Model 1: Baseline ──
    if 1 in models_to_run:
        log.info("\n" + "━" * 70)
        from model1_baseline import run_baseline
        baseline_result = run_baseline(half_a, half_b, test_set, lang_col)
        all_results["Model 1 (Baseline)"] = baseline_result["metrics"]
        cleanup_gpu()
        log.info(f"Model 1 done. GPU memory freed.")

    # ── Model 2: Advanced ──
    if 2 in models_to_run:
        log.info("\n" + "━" * 70)
        from model2_advanced import run_advanced
        adv_result = run_advanced(half_a, half_b, test_set, lang_col)
        all_results["Model 2 (Advanced)"] = adv_result["metrics"]
        del adv_result; cleanup_gpu()
        log.info(f"Model 2 done. GPU memory freed.")

    # ── Model 3: A-type Rephraser ──
    if 3 in models_to_run:
        log.info("\n" + "━" * 70)
        from model3_rephraser import run_rephraser_pipeline
        reph_result = run_rephraser_pipeline(
            half_a, half_b, test_set, lang_col,
            baseline_result=baseline_result)
        all_results["Model 3 (Rephraser)"] = reph_result["metrics"]
        del reph_result; cleanup_gpu()
        log.info(f"Model 3 done. GPU memory freed.")

    # ── Model 4: B-type Semantic ──
    if 4 in models_to_run:
        log.info("\n" + "━" * 70)
        from model4_semantic import run_semantic_pipeline
        sem_result = run_semantic_pipeline(half_a, half_b, test_set, lang_col)
        all_results["Model 4 (Semantic)"] = sem_result["metrics"]
        del sem_result; cleanup_gpu()
        log.info(f"Model 4 done. GPU memory freed.")

    # Free baseline too if still held
    if baseline_result:
        del baseline_result; cleanup_gpu()

    # ── Comparative output ──
    if len(all_results) > 1:
        print_comparison_table(all_results)
        save_dir = os.path.join(CFG.OUTPUT_DIR, "comparison")
        os.makedirs(save_dir, exist_ok=True)
        plot_comparison(all_results, save_dir)

    # Save all results
    summary_path = os.path.join(CFG.OUTPUT_DIR, "all_results.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info(f"All results saved → {summary_path}")

    elapsed = time.time() - start_time
    log.info(f"\nTotal runtime: {elapsed/60:.1f} minutes")
    log.info("Done ✓")


if __name__ == "__main__":
    main()

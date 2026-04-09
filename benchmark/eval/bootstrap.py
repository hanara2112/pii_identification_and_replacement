"""
SAHA-AL Benchmark — Bootstrap Confidence Intervals

Computes 95% CIs for any metric via non-parametric bootstrap (Efron & Tibshirani, 1993).

Usage:
  python -m eval.bootstrap \
      --gold data/test.jsonl \
      --pred predictions/predictions_bart-base-pii.jsonl \
      --metrics elr bertscore token_recall \
      --output results/bootstrap_bart.json
"""

import argparse
import json
import random

import numpy as np

from eval.utils import align_records, load_jsonl
from eval.eval_anonymization import (
    calculate_bertscore,
    entity_leakage_rate,
    format_preservation_rate,
    over_masking_rate,
    token_recall,
)
from eval.eval_privacy import crr3


METRIC_FUNCTIONS = {
    "elr": lambda g, p: entity_leakage_rate(g, p)["elr"],
    "token_recall": token_recall,
    "omr": over_masking_rate,
    "fpr": format_preservation_rate,
    "crr3": crr3,
}


def bootstrap_ci(
    gold_records: list[dict],
    predictions: list[dict],
    metric_fn,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """
    Compute bootstrap confidence interval for a metric.

    Args:
        metric_fn: callable(gold_list, pred_list) -> float
    Returns:
        dict with mean, ci_lower, ci_upper, std
    """
    rng = random.Random(seed)
    n = len(gold_records)
    scores = []

    for _ in range(n_bootstrap):
        indices = [rng.randint(0, n - 1) for _ in range(n)]
        g_sample = [gold_records[i] for i in indices]
        p_sample = [predictions[i] for i in indices]
        scores.append(metric_fn(g_sample, p_sample))

    scores = np.array(scores)
    alpha = 1 - confidence
    lower = float(np.percentile(scores, alpha / 2 * 100))
    upper = float(np.percentile(scores, (1 - alpha / 2) * 100))

    return {
        "mean": round(float(scores.mean()), 4),
        "ci_lower": round(lower, 4),
        "ci_upper": round(upper, 4),
        "std": round(float(scores.std()), 4),
        "n_bootstrap": n_bootstrap,
    }


def main():
    parser = argparse.ArgumentParser(description="Bootstrap CIs for SAHA-AL metrics")
    parser.add_argument("--gold", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--metrics", nargs="+", default=["elr", "token_recall", "crr3"],
                        choices=list(METRIC_FUNCTIONS.keys()))
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    gold = load_jsonl(args.gold)
    preds = load_jsonl(args.pred)
    gold, preds = align_records(gold, preds)

    results = {}
    for metric_name in args.metrics:
        print(f"Bootstrapping {metric_name} ({args.n_bootstrap} iterations)...")
        fn = METRIC_FUNCTIONS[metric_name]
        ci = bootstrap_ci(gold, preds, fn, n_bootstrap=args.n_bootstrap)
        results[metric_name] = ci
        print(f"  {metric_name}: {ci['mean']:.2f} [{ci['ci_lower']:.2f}, {ci['ci_upper']:.2f}]")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()

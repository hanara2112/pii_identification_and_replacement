"""
SAHA-AL Benchmark — Task 1: PII Detection Evaluation

Computes span-level Precision / Recall / F1 in three modes:
  - exact:      (start, end) must match exactly
  - partial:    character IoU > 0.5
  - type_aware: exact span + entity type must match

Also reports per-entity-type recall.

Usage:
  python -m eval.eval_detection \
      --gold data/test.jsonl \
      --pred predictions/bert_ner_spans.jsonl \
      --output results/eval_detection.json
"""

import argparse
import json
from collections import Counter, defaultdict

from eval.utils import load_jsonl, align_records, span_match


def _match_spans(pred_spans: list[dict], gold_spans: list[dict], mode: str):
    """Greedy bipartite matching of predicted spans to gold spans."""
    matched_gold = set()
    matched_pred = set()

    for pi, ps in enumerate(pred_spans):
        for gi, gs in enumerate(gold_spans):
            if gi in matched_gold:
                continue
            if span_match(ps, gs, mode=mode):
                matched_pred.add(pi)
                matched_gold.add(gi)
                break

    tp = len(matched_gold)
    fp = len(pred_spans) - len(matched_pred)
    fn = len(gold_spans) - len(matched_gold)
    return tp, fp, fn


def compute_span_metrics(gold_records, predictions, mode="exact"):
    """Compute corpus-level span P/R/F1."""
    total_tp, total_fp, total_fn = 0, 0, 0

    for g, p in zip(gold_records, predictions):
        gold_spans = [
            e for e in g.get("entities", [])
            if isinstance(e.get("start"), int) and e["start"] >= 0
        ]
        pred_spans = p.get("detected_entities", [])

        tp, fp, fn = _match_spans(pred_spans, gold_spans, mode=mode)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return {
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1": round(f1 * 100, 2),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }


def per_type_recall(gold_records, predictions, mode="exact"):
    """Compute recall per entity type."""
    type_tp = Counter()
    type_total = Counter()

    for g, p in zip(gold_records, predictions):
        gold_spans = [
            e for e in g.get("entities", [])
            if isinstance(e.get("start"), int) and e["start"] >= 0
        ]
        pred_spans = p.get("detected_entities", [])

        for gs in gold_spans:
            etype = gs.get("type", "UNKNOWN")
            type_total[etype] += 1
            for ps in pred_spans:
                if span_match(ps, gs, mode=mode):
                    type_tp[etype] += 1
                    break

    result = {}
    for etype in sorted(type_total.keys()):
        total = type_total[etype]
        tp = type_tp[etype]
        result[etype] = {
            "recall": round(tp / total * 100, 2) if total else 0,
            "tp": tp,
            "total": total,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description="SAHA-AL Task 1: PII Detection Evaluation")
    parser.add_argument("--gold", required=True, help="Gold test.jsonl")
    parser.add_argument("--pred", required=True, help="Predictions JSONL with detected_entities")
    parser.add_argument("--output", default=None, help="Output JSON for results")
    args = parser.parse_args()

    gold = load_jsonl(args.gold)
    preds = load_jsonl(args.pred)
    gold, preds = align_records(gold, preds)

    results = {}
    for mode in ["exact", "partial", "type_aware"]:
        results[mode] = compute_span_metrics(gold, preds, mode=mode)

    results["per_type_recall"] = per_type_recall(gold, preds, mode="exact")

    print("\n" + "=" * 50)
    print("  SAHA-AL Task 1: PII Detection")
    print("=" * 50)
    for mode in ["exact", "partial", "type_aware"]:
        m = results[mode]
        print(f"  {mode:12s}  P={m['precision']:5.2f}  R={m['recall']:5.2f}  F1={m['f1']:5.2f}")
    print("-" * 50)
    print("  Per-type recall (exact):")
    for etype, info in results["per_type_recall"].items():
        print(f"    {etype:20s} {info['recall']:5.2f}%  ({info['tp']}/{info['total']})")
    print("=" * 50)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()

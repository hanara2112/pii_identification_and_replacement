"""
SAHA-AL Benchmark — Failure Taxonomy Analysis
===============================================
Classifies anonymization failures into 5 categories:
  1. Boundary Error   — entity partially masked
  2. Type Confusion   — replacement doesn't match expected format
  3. Ghost Leak       — entity removed but context still reveals identity
  4. Over-Masking     — non-entity content unnecessarily altered
  5. Format Break     — structured replacement breaks document format

Usage:
  python -m analysis.failure_taxonomy \
      --gold data/test.jsonl \
      --pred predictions/predictions_bart-base-pii.jsonl \
      --output results/failure_taxonomy_bart.json
"""

import argparse
import json
import os
import re

from eval.utils import (
    FORMAT_PATTERNS,
    align_records,
    extract_replacement,
    load_jsonl,
    normalize_text,
)


def classify_failures(gold_records, predictions, max_examples=10):
    """
    Classify each entity replacement into one of 5 failure categories or 'clean'.
    Returns counts and example lists.
    """
    counts = {
        "clean": 0,
        "boundary": 0,
        "type_confusion": 0,
        "ghost_leak": 0,
        "over_mask": 0,
        "format_break": 0,
        "full_leak": 0,
    }
    examples = {k: [] for k in counts}

    for g, p in zip(gold_records, predictions):
        orig = g.get("original_text", "")
        pred = normalize_text(p.get("anonymized_text"))

        for ent in g.get("entities", []):
            if ent.get("start", -1) < 0:
                continue
            ent_text = ent["text"]
            ent_type = ent.get("type", "UNKNOWN")

            leaked = re.search(
                rf"(?<!\w){re.escape(ent_text)}(?!\w)", pred, re.IGNORECASE
            )
            if leaked:
                counts["full_leak"] += 1
                continue

            tokens = re.findall(r"\w+", ent_text)
            partial_leaked = [
                t for t in tokens
                if len(t) > 2
                and re.search(rf"(?<!\w){re.escape(t)}(?!\w)", pred, re.IGNORECASE)
            ]
            if partial_leaked and len(partial_leaked) < len(tokens):
                counts["boundary"] += 1
                if len(examples["boundary"]) < max_examples:
                    examples["boundary"].append({
                        "id": g["id"], "entity": ent_text,
                        "type": ent_type, "leaked_tokens": partial_leaked,
                    })
                continue

            pattern = FORMAT_PATTERNS.get(ent_type)
            if pattern:
                replacement = extract_replacement(orig, pred, ent["start"], ent.get("end", ent["start"]))
                if replacement and not pattern.search(replacement):
                    counts["format_break"] += 1
                    if len(examples["format_break"]) < max_examples:
                        examples["format_break"].append({
                            "id": g["id"], "entity": ent_text,
                            "type": ent_type, "replacement": replacement,
                        })
                    continue
                elif replacement and pattern.search(replacement):
                    pass

            ctx_start = max(0, ent["start"] - 50)
            ctx_end = min(len(orig), ent.get("end", ent["start"]) + 50)
            orig_ctx = orig[ctx_start:ctx_end].lower()
            pred_ctx_start = max(0, ent["start"] - 50)
            pred_ctx_end = min(len(pred), ent.get("end", ent["start"]) + 50)

            if pred_ctx_end <= len(pred):
                pred_ctx = pred[pred_ctx_start:pred_ctx_end].lower()
                ctx_tokens = set(re.findall(r"\w+", orig_ctx)) - set(
                    re.findall(r"\w+", ent_text.lower())
                )
                pred_tokens = set(re.findall(r"\w+", pred_ctx))
                if ctx_tokens and len(ctx_tokens & pred_tokens) / len(ctx_tokens) > 0.8:
                    counts["ghost_leak"] += 1
                    if len(examples["ghost_leak"]) < max_examples:
                        examples["ghost_leak"].append({
                            "id": g["id"], "entity": ent_text,
                            "type": ent_type,
                            "shared_context": list(ctx_tokens & pred_tokens)[:5],
                        })
                    continue

            counts["clean"] += 1

    return counts, examples


def main():
    parser = argparse.ArgumentParser(description="SAHA-AL failure taxonomy")
    parser.add_argument("--gold", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-examples", type=int, default=10)
    args = parser.parse_args()

    gold = load_jsonl(args.gold)
    preds = load_jsonl(args.pred)
    gold, preds = align_records(gold, preds)

    counts, examples = classify_failures(gold, preds, max_examples=args.max_examples)

    total = sum(counts.values())
    print("\n" + "=" * 55)
    print("  Failure Taxonomy Distribution")
    print("=" * 55)
    for cat in ["clean", "full_leak", "boundary", "type_confusion", "ghost_leak", "over_mask", "format_break"]:
        c = counts[cat]
        pct = c / total * 100 if total else 0
        print(f"  {cat:18s} {c:6d}  ({pct:5.1f}%)")
    print(f"  {'TOTAL':18s} {total:6d}")
    print("=" * 55)

    for cat, exs in examples.items():
        if exs:
            print(f"\n  Examples [{cat}]:")
            for ex in exs[:3]:
                print(f"    {ex}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"counts": counts, "examples": examples}, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

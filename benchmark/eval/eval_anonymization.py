"""
SAHA-AL Benchmark — Task 2: Text Anonymization Evaluation

Metrics:
  - ELR   : Entity Leakage Rate (word-boundary regex match)
  - TokRecall : Fraction of entity-span tokens absent from prediction
  - OMR   : Over-Masking Rate (non-entity tokens altered)
  - FPR   : Format Preservation Rate (structured replacements match format regex)
  - BERTScore F1 : Semantic similarity (pinned distilbert-base-uncased)
  - NLI   : NLI Consistency (fraction classified as entailment)

Usage:
  python -m eval.eval_anonymization \
      --gold data/test.jsonl \
      --pred predictions/predictions_bart-base-pii.jsonl \
      --output results/eval_anon_bart.json
"""

import argparse
import json
import re
import warnings
from collections import Counter

from eval.utils import (
    FORMAT_PATTERNS,
    align_records,
    exact_entity_match,
    extract_replacement,
    load_jsonl,
    normalize_text,
)

warnings.filterwarnings("ignore")


# ── ELR ──

def entity_leakage_rate(gold_records, predictions):
    """Entity Leakage Rate: fraction of gold entities whose text appears in prediction."""
    leaked, total = 0, 0
    per_type = Counter()
    leaked_per_type = Counter()

    for g, p in zip(gold_records, predictions):
        pred_text = normalize_text(p.get("anonymized_text"))
        for ent in g.get("entities", []):
            ent_text = ent.get("text", "")
            ent_type = ent.get("type", "UNKNOWN") or "UNKNOWN"
            total += 1
            per_type[ent_type] += 1
            if exact_entity_match(ent_text, pred_text):
                leaked += 1
                leaked_per_type[ent_type] += 1

    elr = (leaked / total * 100) if total else 0.0
    return {
        "elr": round(elr, 2),
        "leaked": leaked,
        "total_entities": total,
        "per_type": dict(per_type),
        "leaked_per_type": dict(leaked_per_type),
    }


# ── Token Recall ──

def token_recall(gold_records, predictions):
    """Fraction of entity-span tokens that are absent from the prediction."""
    removed, total = 0, 0

    for g, p in zip(gold_records, predictions):
        pred_lower = normalize_text(p.get("anonymized_text")).lower()
        pred_tokens = set(re.findall(r"\w+", pred_lower))

        for ent in g.get("entities", []):
            ent_text = ent.get("text", "")
            if not ent_text:
                continue
            ent_tokens = re.findall(r"\w+", ent_text.lower())
            for tok in ent_tokens:
                if len(tok) < 2:
                    continue
                total += 1
                if tok not in pred_tokens:
                    removed += 1

    return round((removed / total * 100) if total else 0, 2)


# ── Over-Masking Rate ──

def over_masking_rate(gold_records, predictions):
    """Fraction of non-entity tokens that were altered in the prediction."""
    altered, total = 0, 0

    for g, p in zip(gold_records, predictions):
        orig = g.get("original_text", "")
        pred = normalize_text(p.get("anonymized_text"))

        entity_char_set = set()
        for ent in g.get("entities", []):
            s, e = ent.get("start", -1), ent.get("end", -1)
            if s >= 0 and e > s:
                entity_char_set.update(range(s, e))

        orig_tokens = list(re.finditer(r"\w+", orig))
        pred_lower = pred.lower()

        for m in orig_tokens:
            tok_start, tok_end = m.start(), m.end()
            if any(i in entity_char_set for i in range(tok_start, tok_end)):
                continue
            total += 1
            tok_text = m.group().lower()
            if tok_text not in pred_lower:
                altered += 1

    return round((altered / total * 100) if total else 0, 2)


# ── Format Preservation Rate ──

def format_preservation_rate(gold_records, predictions):
    """Fraction of structured-type replacements that match the expected format regex."""
    matched, total = 0, 0

    for g, p in zip(gold_records, predictions):
        orig = g.get("original_text", "")
        pred = normalize_text(p.get("anonymized_text"))

        for ent in g.get("entities", []):
            pattern = FORMAT_PATTERNS.get(ent.get("type"))
            if not pattern:
                continue
            s = ent.get("start", -1)
            if s < 0:
                continue
            total += 1
            replacement = extract_replacement(orig, pred, s, ent.get("end", s))
            if replacement and pattern.search(replacement):
                matched += 1

    return round((matched / total * 100) if total else 0, 2)


# ── BERTScore ──

def _patch_bert_score():
    """Patch bert_score for compatibility with recent transformers versions."""
    import bert_score.utils as bs_utils
    if getattr(bs_utils, "_sent_encode_orig", None) is None:
        bs_utils._sent_encode_orig = bs_utils.sent_encode

        def sent_encode(tokenizer, sent):
            if not sent.strip() and not hasattr(tokenizer, "build_inputs_with_special_tokens"):
                return tokenizer.encode("", add_special_tokens=True)
            return bs_utils._sent_encode_orig(tokenizer, sent)

        bs_utils.sent_encode = sent_encode


def calculate_bertscore(gold_records, predictions, model_type="distilbert-base-uncased"):
    """BERTScore F1 between original and anonymized texts."""
    try:
        from bert_score import score
        _patch_bert_score()
    except ImportError:
        print("[WARN] bert_score not installed. pip install bert_score")
        return None

    refs = [g.get("original_text", "") for g in gold_records]
    cands = [normalize_text(p.get("anonymized_text")) for p in predictions]

    paired = [(c, r) for c, r in zip(cands, refs) if (r or "").strip()]
    if not paired:
        return None

    cands, refs = zip(*paired)
    P, R, F1 = score(list(cands), list(refs), lang="en", verbose=False, model_type=model_type)
    return round(F1.mean().item() * 100, 2)


# ── NLI Consistency ──

def nli_consistency(gold_records, predictions, batch_size=32):
    """Fraction of (original, anonymized) pairs classified as entailment by an NLI model."""
    try:
        from transformers import pipeline
    except ImportError:
        print("[WARN] transformers not installed. pip install transformers")
        return None

    nli = pipeline("text-classification", model="roberta-large-mnli", device=-1)

    entailment_count, total = 0, 0
    pairs = []
    for g, p in zip(gold_records, predictions):
        orig = g.get("original_text", "").strip()
        anon = normalize_text(p.get("anonymized_text"))
        if orig and anon != "[EMPTY]":
            pairs.append(f"{orig}</s></s>{anon}")

    for i in range(0, len(pairs), batch_size):
        batch = pairs[i : i + batch_size]
        results = nli(batch, truncation=True, max_length=512)
        for r in results:
            total += 1
            if r["label"] == "ENTAILMENT":
                entailment_count += 1

    return round((entailment_count / total * 100) if total else 0, 2)


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="SAHA-AL Task 2: Text Anonymization Evaluation")
    parser.add_argument("--gold", required=True, help="Gold test.jsonl")
    parser.add_argument("--pred", required=True, help="Predictions JSONL")
    parser.add_argument("--bert-model", default="distilbert-base-uncased")
    parser.add_argument("--skip-nli", action="store_true", help="Skip NLI (slow)")
    parser.add_argument("--skip-bertscore", action="store_true")
    parser.add_argument("--print-types", action="store_true")
    parser.add_argument("--output", default=None, help="Output JSON")
    args = parser.parse_args()

    gold = load_jsonl(args.gold)
    preds = load_jsonl(args.pred)
    gold, preds = align_records(gold, preds)

    print(f"Evaluating {len(preds)} records...")

    elr_result = entity_leakage_rate(gold, preds)
    tok_recall = token_recall(gold, preds)
    omr = over_masking_rate(gold, preds)
    fpr = format_preservation_rate(gold, preds)
    bert_f1 = None if args.skip_bertscore else calculate_bertscore(gold, preds, args.bert_model)
    nli = None if args.skip_nli else nli_consistency(gold, preds)

    results = {
        "elr": elr_result["elr"],
        "token_recall": tok_recall,
        "over_masking_rate": omr,
        "format_preservation_rate": fpr,
        "bertscore_f1": bert_f1,
        "nli_consistency": nli,
        "elr_detail": elr_result,
    }

    print("\n" + "=" * 50)
    print("  SAHA-AL Task 2: Text Anonymization")
    print("=" * 50)
    print(f"  ELR ↓          : {elr_result['elr']:6.2f}%  ({elr_result['leaked']}/{elr_result['total_entities']})")
    print(f"  Token Recall ↑ : {tok_recall:6.2f}%")
    print(f"  OMR ↓          : {omr:6.2f}%")
    print(f"  FPR ↑          : {fpr:6.2f}%")
    if bert_f1 is not None:
        print(f"  BERTScore F1 ↑ : {bert_f1:6.2f}")
    if nli is not None:
        print(f"  NLI Consistency: {nli:6.2f}%")

    if args.print_types:
        print("\n  Per-type ELR:")
        pt = elr_result["per_type"]
        lpt = elr_result["leaked_per_type"]
        for etype in sorted(pt.keys()):
            count = pt[etype]
            lk = lpt.get(etype, 0)
            print(f"    {etype:20s} {lk/count*100:5.2f}% ({lk}/{count})")

    print("=" * 50)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()

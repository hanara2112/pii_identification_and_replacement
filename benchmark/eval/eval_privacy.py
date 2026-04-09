"""
SAHA-AL Benchmark — Task 3: Privacy Risk Assessment

Metrics:
  - CRR-3 : Capitalized 3-gram survival rate
  - ERA   : Entity Recovery Attack (retrieval adversary, top-k accuracy)
  - LRR   : LLM Re-identification Rate (generative adversary)
  - UAC   : Unique Attribute Combination rate (k-anonymity proxy)

References:
  ERA uses Sentence-BERT embeddings (Reimers & Gurevych, EMNLP 2019).
  LRR is inspired by Staab et al. (2023) on LLM privacy inference.
  UAC is grounded in k-anonymity (Sweeney, 2002).

Usage:
  python -m eval.eval_privacy \
      --gold data/test.jsonl \
      --pred predictions/predictions_bart-base-pii.jsonl \
      --train data/train.jsonl \
      --output results/eval_privacy_bart.json
"""

import argparse
import json
import re
from collections import Counter
from difflib import SequenceMatcher

from eval.utils import align_records, get_capitalized_ngrams, load_jsonl, normalize_text


# ── CRR-3 ──

def crr3(gold_records, predictions):
    """Capitalized 3-gram survival rate."""
    survived, total = 0, 0

    for g, p in zip(gold_records, predictions):
        orig_3grams = get_capitalized_ngrams(g.get("original_text", ""), n=3)
        pred_text = normalize_text(p.get("anonymized_text")).lower()

        for ng in orig_3grams:
            total += 1
            if " ".join(ng).lower() in pred_text:
                survived += 1

    return round((survived / total * 100) if total else 0, 2)


# ── ERA ──

def entity_recovery_attack(gold_records, predictions, train_records, top_k=5):
    """
    Retrieval-based adversary: embed anonymized text, rank candidate entities
    by cosine similarity, measure if the original entity is recovered.
    Requires: pip install sentence-transformers
    """
    try:
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.util import cos_sim
    except ImportError:
        print("[WARN] sentence-transformers not installed. Skipping ERA.")
        return None

    model = SentenceTransformer("all-MiniLM-L6-v2")

    type_pools: dict[str, list[str]] = {}
    for r in train_records:
        for e in r.get("entities", []):
            etype = e.get("type", "UNKNOWN")
            text = e.get("text", "")
            if text:
                type_pools.setdefault(etype, set()).add(text)
    type_pools = {k: list(v) for k, v in type_pools.items()}

    top1_hits, top5_hits, total = 0, 0, 0

    for g, p in zip(gold_records, predictions):
        for ent in g.get("entities", []):
            if ent.get("start", -1) < 0:
                continue
            etype = ent.get("type", "UNKNOWN")
            original_val = ent["text"]

            pool = list(type_pools.get(etype, []))
            if original_val not in pool:
                pool.append(original_val)
            if len(pool) < 2:
                continue

            anon_emb = model.encode(p["anonymized_text"], convert_to_tensor=True)
            pool_embs = model.encode(pool, convert_to_tensor=True)

            scores = cos_sim(anon_emb, pool_embs)[0]
            ranked = sorted(range(len(pool)), key=lambda i: scores[i], reverse=True)

            total += 1
            if pool[ranked[0]] == original_val:
                top1_hits += 1
            if original_val in [pool[ranked[i]] for i in range(min(top_k, len(ranked)))]:
                top5_hits += 1

    return {
        "era_top1": round(top1_hits / total * 100, 2) if total else 0,
        "era_top5": round(top5_hits / total * 100, 2) if total else 0,
        "total_evaluated": total,
    }


# ── LRR ──

LRR_PROMPT = """You are given an anonymized text where personal information has been replaced with fake alternatives.

Anonymized text: {anonymized_text}

For each piece of replaced information, try to guess the ORIGINAL value based on contextual clues. Output one guess per line in the format:
REPLACEMENT -> YOUR_GUESS

Only output guesses, nothing else."""


def llm_reidentification_rate(gold_records, predictions, sample_n=300, model="gpt-4o-mini"):
    """
    Prompt an LLM to recover original entities from anonymized text.
    Measures exact match rate and fuzzy match rate (>0.8 char similarity).
    """
    try:
        from openai import OpenAI
    except ImportError:
        print("[WARN] openai not installed. Skipping LRR. pip install openai")
        return None

    client = OpenAI()
    exact, fuzzy, total = 0, 0, 0

    for g, p in zip(gold_records[:sample_n], predictions[:sample_n]):
        original_entities = {e["text"] for e in g.get("entities", []) if e.get("text")}
        if not original_entities:
            continue

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": LRR_PROMPT.format(anonymized_text=p["anonymized_text"])}],
                temperature=0,
                max_tokens=512,
            )
            guesses_raw = response.choices[0].message.content.strip().split("\n")
        except Exception as e:
            print(f"[WARN] LRR API call failed: {e}")
            continue

        parsed_guesses = set()
        for line in guesses_raw:
            if "->" in line:
                parsed_guesses.add(line.split("->")[-1].strip())

        for orig_ent in original_entities:
            total += 1
            if orig_ent in parsed_guesses:
                exact += 1
            elif any(
                SequenceMatcher(None, orig_ent.lower(), guess.lower()).ratio() > 0.8
                for guess in parsed_guesses
            ):
                fuzzy += 1

    return {
        "lrr_exact": round(exact / total * 100, 2) if total else 0,
        "lrr_fuzzy": round((exact + fuzzy) / total * 100, 2) if total else 0,
        "total_evaluated": total,
        "sample_n": sample_n,
    }


# ── UAC ──

_TYPE_HINTS = {
    "EMAIL": ["email", "@", "mail"],
    "PHONE": ["phone", "call", "tel"],
    "SSN": ["ssn", "social security"],
    "DATE": ["born", "date", "birthday"],
    "ADDRESS": ["lives", "address", "street", "road"],
}


def unique_attribute_combination_rate(gold_records, predictions):
    """
    Compositional privacy proxy grounded in k-anonymity (Sweeney, 2000/2002).
    For each record, extract surviving quasi-identifier types (leaked or
    context-inferable). Records with unique type-combinations have k=1.
    """
    combos = Counter()
    record_combos = []

    for g, p in zip(gold_records, predictions):
        pred_text = p.get("anonymized_text", "")
        surviving_types = []
        for ent in g.get("entities", []):
            if ent.get("start", -1) < 0:
                continue
            if ent["text"].lower() in pred_text.lower():
                surviving_types.append(ent["type"])
            else:
                start = max(0, ent["start"] - 30)
                end = min(len(g["original_text"]), ent["end"] + 30)
                context = g["original_text"][start:end].lower()
                hints = _TYPE_HINTS.get(ent.get("type", ""), [])
                if any(h in context for h in hints):
                    surviving_types.append(ent["type"])

        combo = tuple(sorted(surviving_types))
        combos[combo] += 1
        record_combos.append(combo)

    unique = sum(1 for c in record_combos if combos[c] == 1)
    return round(unique / len(record_combos) * 100, 2) if record_combos else 0


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="SAHA-AL Task 3: Privacy Risk Assessment")
    parser.add_argument("--gold", required=True, help="Gold test.jsonl")
    parser.add_argument("--pred", required=True, help="Predictions JSONL")
    parser.add_argument("--train", default=None, help="Train JSONL (for ERA candidate pool)")
    parser.add_argument("--output", default=None, help="Output JSON")
    parser.add_argument("--skip-era", action="store_true")
    parser.add_argument("--skip-lrr", action="store_true")
    parser.add_argument("--lrr-sample", type=int, default=300)
    args = parser.parse_args()

    gold = load_jsonl(args.gold)
    preds = load_jsonl(args.pred)
    gold, preds = align_records(gold, preds)

    train_records = load_jsonl(args.train) if args.train else []

    print(f"Evaluating privacy on {len(preds)} records...")

    crr = crr3(gold, preds)
    era = None if args.skip_era else entity_recovery_attack(gold, preds, train_records)
    lrr = None if args.skip_lrr else llm_reidentification_rate(gold, preds, sample_n=args.lrr_sample)
    uac = unique_attribute_combination_rate(gold, preds)

    results = {"crr3": crr, "era": era, "lrr": lrr, "uac": uac}

    print("\n" + "=" * 50)
    print("  SAHA-AL Task 3: Privacy Under Attack")
    print("=" * 50)
    print(f"  CRR-3 ↓       : {crr:6.2f}%")
    if era:
        print(f"  ERA@1 ↓       : {era['era_top1']:6.2f}%")
        print(f"  ERA@5 ↓       : {era['era_top5']:6.2f}%")
    if lrr:
        print(f"  LRR exact ↓   : {lrr['lrr_exact']:6.2f}%")
        print(f"  LRR fuzzy ↓   : {lrr['lrr_fuzzy']:6.2f}%")
    print(f"  UAC ↓         : {uac:6.2f}%")
    print("=" * 50)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()

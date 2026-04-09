"""
SAHA-AL Benchmark — Cross-Dataset Transfer Evaluation on TAB
==============================================================
Evaluates a SAHA-AL-trained model on the Text Anonymization Benchmark (TAB)
test set to measure synthetic-to-real domain shift.

TAB uses token-level recall/precision with DIRECT/QUASI masking decisions.
This script computes TAB-compatible metrics from seq2seq model predictions.

Requires: TAB test data in CoNLL-like format or the TAB evaluation script.

Usage:
  python -m analysis.tab_transfer \
      --tab-dir ../ref/text-anonymization-benchmark \
      --model-name bart-base-pii \
      --base-model facebook/bart-base \
      --checkpoint path/to/best_model.pt \
      --output results/tab_transfer.json
"""

import argparse
import json
import os
import re
import sys


def load_tab_documents(tab_dir):
    """Load TAB test documents from the echr_* directories."""
    docs = []
    test_dir = os.path.join(tab_dir, "echr_train")  # TAB uses echr_train as test

    if not os.path.isdir(test_dir):
        for candidate in ["echr_train", "test", "data"]:
            test_dir = os.path.join(tab_dir, candidate)
            if os.path.isdir(test_dir):
                break

    if not os.path.isdir(test_dir):
        print(f"[ERROR] Cannot find TAB test documents in {tab_dir}")
        return []

    for fname in sorted(os.listdir(test_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(test_dir, fname)) as f:
            doc = json.load(f)
        docs.append(doc)

    print(f"Loaded {len(docs)} TAB documents from {test_dir}")
    return docs


def extract_tab_entities(doc):
    """Extract gold entities from a TAB document with their masking decisions."""
    entities = []
    text = doc.get("text", "")
    for annotation in doc.get("annotations", []):
        for span in annotation.get("spans", []):
            start = span.get("start", 0)
            end = span.get("end", 0)
            entity_text = text[start:end]
            masking = annotation.get("entity_mention_id", "")
            entity_type = annotation.get("entity_type", "MISC")
            identifier_type = annotation.get("identifier_type", "NO_MASK")
            entities.append({
                "text": entity_text,
                "start": start,
                "end": end,
                "type": entity_type,
                "identifier_type": identifier_type,
            })
    return text, entities


def compute_tab_token_metrics(original_text, entities, anonymized_text):
    """
    Compute TAB-style token-level metrics.
    Token recall: fraction of entity tokens removed from prediction.
    Token precision: fraction of removed tokens that were actually entities.
    """
    orig_tokens = original_text.split()
    pred_tokens = anonymized_text.split()

    entity_token_indices = set()
    char_pos = 0
    for idx, token in enumerate(orig_tokens):
        token_start = original_text.find(token, char_pos)
        token_end = token_start + len(token)
        char_pos = token_end

        for ent in entities:
            if (token_start < ent["end"] and token_end > ent["start"]):
                entity_token_indices.add(idx)
                break

    pred_set = set(t.lower() for t in pred_tokens)
    removed_indices = set()
    for idx, token in enumerate(orig_tokens):
        if token.lower() not in pred_set:
            removed_indices.add(idx)

    tp = len(entity_token_indices & removed_indices)
    fn = len(entity_token_indices - removed_indices)
    fp = len(removed_indices - entity_token_indices)

    recall = tp / (tp + fn) if (tp + fn) else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    direct_ents = [e for e in entities if e["identifier_type"] == "DIRECT"]
    quasi_ents = [e for e in entities if e["identifier_type"] == "QUASI"]

    direct_indices = set()
    quasi_indices = set()
    char_pos = 0
    for idx, token in enumerate(orig_tokens):
        token_start = original_text.find(token, char_pos)
        token_end = token_start + len(token)
        char_pos = token_end
        for ent in direct_ents:
            if token_start < ent["end"] and token_end > ent["start"]:
                direct_indices.add(idx)
        for ent in quasi_ents:
            if token_start < ent["end"] and token_end > ent["start"]:
                quasi_indices.add(idx)

    direct_tp = len(direct_indices & removed_indices)
    direct_total = len(direct_indices)
    quasi_tp = len(quasi_indices & removed_indices)
    quasi_total = len(quasi_indices)

    return {
        "token_recall": round(recall * 100, 2),
        "token_precision": round(precision * 100, 2),
        "token_f1": round(f1 * 100, 2),
        "direct_recall": round(direct_tp / direct_total * 100, 2) if direct_total else 0,
        "quasi_recall": round(quasi_tp / quasi_total * 100, 2) if quasi_total else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Cross-dataset evaluation on TAB")
    parser.add_argument("--tab-dir", required=True, help="Path to TAB repository")
    parser.add_argument("--predictions", required=True,
                        help="JSONL with {doc_id, anonymized_text} or run inference")
    parser.add_argument("--output", default="results/tab_transfer.json")
    args = parser.parse_args()

    tab_docs = load_tab_documents(args.tab_dir)
    if not tab_docs:
        return

    with open(args.predictions) as f:
        preds = {json.loads(line)["id"]: json.loads(line) for line in f}

    all_metrics = []
    for doc in tab_docs:
        doc_id = doc.get("doc_id", doc.get("id", ""))
        text, entities = extract_tab_entities(doc)

        if doc_id in preds:
            anon_text = preds[doc_id]["anonymized_text"]
        else:
            continue

        metrics = compute_tab_token_metrics(text, entities, anon_text)
        metrics["doc_id"] = doc_id
        all_metrics.append(metrics)

    if not all_metrics:
        print("No matching predictions found for TAB documents.")
        return

    import numpy as np
    avg = {
        "token_recall": round(np.mean([m["token_recall"] for m in all_metrics]), 2),
        "token_precision": round(np.mean([m["token_precision"] for m in all_metrics]), 2),
        "token_f1": round(np.mean([m["token_f1"] for m in all_metrics]), 2),
        "direct_recall": round(np.mean([m["direct_recall"] for m in all_metrics]), 2),
        "quasi_recall": round(np.mean([m["quasi_recall"] for m in all_metrics]), 2),
        "num_documents": len(all_metrics),
    }

    print("\n" + "=" * 50)
    print("  TAB Cross-Dataset Transfer Results")
    print("=" * 50)
    print(f"  Documents evaluated: {avg['num_documents']}")
    print(f"  Token Recall (all):  {avg['token_recall']:5.2f}%")
    print(f"  Token Recall (DIRECT): {avg['direct_recall']:5.2f}%")
    print(f"  Token Recall (QUASI):  {avg['quasi_recall']:5.2f}%")
    print(f"  Token Precision:     {avg['token_precision']:5.2f}%")
    print(f"  Token F1:            {avg['token_f1']:5.2f}%")
    print("=" * 50)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"average": avg, "per_document": all_metrics}, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()

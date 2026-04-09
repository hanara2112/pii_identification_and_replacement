"""
SAHA-AL Benchmark — Hybrid Baseline (spaCy detect + GPT-4o-mini replace)
=========================================================================
Isolates detection quality vs generation quality by using spaCy + regex
for entity detection and an LLM for realistic replacement.

Requires: OPENAI_API_KEY, spaCy en_core_web_lg.

Usage:
  python -m baselines.hybrid_baseline \
      --gold data/test.jsonl \
      --output predictions/predictions_hybrid.jsonl \
      --save-spans
"""

import argparse
import json
import os
import re
import time
from collections import OrderedDict

from eval.utils import load_jsonl, save_jsonl


REGEX_PATTERNS = OrderedDict([
    ("EMAIL", re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')),
    ("SSN", re.compile(r'\b\d{3}-\d{2}-\d{4}\b')),
    ("CREDIT_CARD", re.compile(r'\b(?:\d[ -]*?){13,19}\b')),
    ("IBAN", re.compile(r'\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}(?:[A-Z0-9]{0,16})?\b')),
    ("IP_ADDRESS", re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')),
    ("PHONE", re.compile(r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b')),
    ("URL", re.compile(r'https?://[^\s<>\"\']+|www\.[^\s<>\"\']+\.[^\s<>\"\']+')),
    ("DATE", re.compile(
        r'\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b'
        r'|\b\d{4}[/.-]\d{1,2}[/.-]\d{1,2}\b'
        r'|\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?'
        r'|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?'
        r'|Dec(?:ember)?)\s+\d{1,2},?\s+\d{4}\b', re.IGNORECASE)),
])


def _resolve_overlapping_spans(spans):
    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    resolved = []
    last_end = -1
    for span in spans:
        if span[0] >= last_end:
            resolved.append(span)
            last_end = span[1]
    return resolved


def detect_entities(text, nlp):
    """Detect entities using spaCy NER + regex patterns."""
    doc = nlp(text)
    spans = []
    for ent in doc.ents:
        if ent.label_ in ("PERSON", "ORG", "GPE", "LOC", "DATE", "FAC", "NORP", "EVENT"):
            spans.append((ent.start_char, ent.end_char, ent.label_, ent.text))
    for ent_type, pattern in REGEX_PATTERNS.items():
        for match in pattern.finditer(text):
            spans.append((match.start(), match.end(), ent_type, match.group()))
    return _resolve_overlapping_spans(spans)


def hybrid_anonymize(text, entities, client, model="gpt-4o-mini", max_retries=3):
    """Use an LLM to generate realistic replacements for detected entities."""
    if not entities:
        return text

    entity_list = "\n".join(f'- "{e[3]}" (type: {e[2]})' for e in entities)
    prompt = f"""The following entities were detected as PII in this text:
{entity_list}

Original text: {text}

Replace ONLY the listed entities with realistic fake alternatives of the same type. Keep everything else exactly the same. Return only the result."""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=512,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  [WARN] LLM failed: {e}")
                return text


def main():
    parser = argparse.ArgumentParser(description="SAHA-AL hybrid baseline (spaCy + LLM)")
    parser.add_argument("--gold", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--save-spans", action="store_true")
    args = parser.parse_args()

    import spacy
    from openai import OpenAI

    nlp = spacy.load("en_core_web_lg")
    client = OpenAI()

    records = load_jsonl(args.gold)
    if args.sample:
        records = records[:args.sample]

    print(f"Hybrid baseline: spaCy detect + {args.llm_model} replace on {len(records)} records...")

    predictions = []
    span_records = []

    for i, rec in enumerate(records):
        text = rec.get("original_text", "")
        entities = detect_entities(text, nlp)
        anon = hybrid_anonymize(text, entities, client, model=args.llm_model)

        predictions.append({"id": rec["id"], "anonymized_text": anon})
        if args.save_spans:
            span_records.append({
                "id": rec["id"],
                "detected_entities": [
                    {"start": s, "end": e, "type": t, "text": txt}
                    for s, e, t, txt in entities
                ],
            })

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(records)}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_jsonl(predictions, args.output)
    print(f"Predictions saved to {args.output}")

    if args.save_spans:
        span_path = args.output.replace(".jsonl", "_spans.jsonl")
        save_jsonl(span_records, span_path)
        print(f"Spans saved to {span_path}")


if __name__ == "__main__":
    main()

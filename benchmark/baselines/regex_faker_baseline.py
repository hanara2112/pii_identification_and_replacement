"""
SAHA-AL Baseline Generator — Rule-based systems
=================================================
Generates predictions for three baselines:
  1. Regex+Faker   (pattern matching only)
  2. spaCy+Faker   (NER + Faker)
  3. Presidio       (Microsoft Presidio)

Supports --save-spans to emit detected entity spans for Task 1 evaluation.

Usage:
  python -m baselines.regex_faker_baseline --gold data/test.jsonl --mode regex
  python -m baselines.regex_faker_baseline --gold data/test.jsonl --mode spacy --save-spans

Then evaluate:
  python -m eval.eval_anonymization --gold data/test.jsonl --pred predictions/regex_predictions.jsonl
  python -m eval.eval_detection --gold data/test.jsonl --pred predictions/regex_spans.jsonl
"""

import argparse
import json
import os
import re
from collections import OrderedDict

try:
    from faker import Faker
    fake = Faker()
    Faker.seed(42)
except ImportError:
    print("Install faker: pip install faker")
    exit(1)


# ── Regex patterns (from SAHA-AL Layer 1) ──

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
    ("ACCOUNT", re.compile(r'\b(?:account|acct)[#:\s]*\d{6,}\b', re.IGNORECASE)),
    ("ID_NUMBER", re.compile(r'\b[A-Z]{1,3}\d{6,10}\b')),
])

FAKER_GENERATORS = {
    "EMAIL": lambda: fake.email(),
    "SSN": lambda: fake.ssn(),
    "CREDIT_CARD": lambda: fake.credit_card_number(),
    "IBAN": lambda: fake.iban(),
    "IP_ADDRESS": lambda: fake.ipv4(),
    "PHONE": lambda: fake.phone_number(),
    "URL": lambda: fake.url(),
    "DATE": lambda: fake.date(pattern="%m/%d/%Y"),
    "ACCOUNT": lambda: f"Acct#{fake.numerify('########')}",
    "ID_NUMBER": lambda: fake.bothify('??######'),
    "PERSON": lambda: fake.name(),
    "ORG": lambda: fake.company(),
    "GPE": lambda: fake.city(),
    "LOC": lambda: fake.city(),
    "LOCATION": lambda: fake.city(),
    "FULLNAME": lambda: fake.name(),
    "FIRSTNAME": lambda: fake.first_name(),
    "LASTNAME": lambda: fake.last_name(),
    "ADDRESS": lambda: fake.street_address(),
    "USERNAME": lambda: fake.user_name(),
    "PASSWORD": lambda: fake.password(),
}


def get_faker_replacement(entity_type, original_text=""):
    gen = FAKER_GENERATORS.get(entity_type)
    return gen() if gen else f"[{entity_type}]"


def resolve_overlapping_spans(spans):
    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    resolved = []
    last_end = -1
    for span in spans:
        if span[0] >= last_end:
            resolved.append(span)
            last_end = span[1]
    return resolved


def _anonymize_from_spans(text, spans):
    """Replace detected spans right-to-left. Returns (anonymized_text, spans_used)."""
    if not spans:
        return text, []
    result = text
    for start, end, ent_type, original in reversed(spans):
        replacement = get_faker_replacement(ent_type, original)
        result = result[:start] + replacement + result[end:]
    return result, spans


def regex_detect_and_anonymize(text):
    spans = []
    for ent_type, pattern in REGEX_PATTERNS.items():
        for match in pattern.finditer(text):
            spans.append((match.start(), match.end(), ent_type, match.group()))
    spans = resolve_overlapping_spans(spans)
    return _anonymize_from_spans(text, spans)


def spacy_detect_and_anonymize(text, nlp):
    doc = nlp(text)
    all_spans = []
    for ent in doc.ents:
        if ent.label_ in ("PERSON", "ORG", "GPE", "LOC", "DATE", "FAC", "NORP", "EVENT"):
            all_spans.append((ent.start_char, ent.end_char, ent.label_, ent.text))
    for ent_type, pattern in REGEX_PATTERNS.items():
        for match in pattern.finditer(text):
            all_spans.append((match.start(), match.end(), ent_type, match.group()))
    spans = resolve_overlapping_spans(all_spans)
    return _anonymize_from_spans(text, spans)


def presidio_detect_and_anonymize(text, analyzer, anonymizer):
    results = analyzer.analyze(text=text, language="en")
    if not results:
        return text, []
    spans = [(r.start, r.end, r.entity_type, text[r.start:r.end]) for r in results]
    anon_result = anonymizer.anonymize(text=text, analyzer_results=results)
    return anon_result.text, spans


def main():
    parser = argparse.ArgumentParser(description="SAHA-AL rule-based baselines")
    parser.add_argument("--gold", default="data/test.jsonl", help="Gold test.jsonl")
    parser.add_argument("--mode", required=True, choices=["regex", "spacy", "presidio"])
    parser.add_argument("--output-dir", default="predictions")
    parser.add_argument("--save-spans", action="store_true",
                        help="Also save detected entity spans for Task 1 evaluation")
    args = parser.parse_args()

    with open(args.gold, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    print(f"Loaded {len(records)} test records.")

    detect_and_anonymize_fn = None
    nlp = None

    if args.mode == "regex":
        detect_and_anonymize_fn = lambda t: regex_detect_and_anonymize(t)
        print("Mode: Regex + Faker")
    elif args.mode == "spacy":
        import spacy
        nlp = spacy.load("en_core_web_lg")
        detect_and_anonymize_fn = lambda t: spacy_detect_and_anonymize(t, nlp)
        print("Mode: spaCy + Regex + Faker")
    elif args.mode == "presidio":
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine
        analyzer = AnalyzerEngine()
        anonymizer = AnonymizerEngine()
        detect_and_anonymize_fn = lambda t: presidio_detect_and_anonymize(t, analyzer, anonymizer)
        print("Mode: Microsoft Presidio")

    os.makedirs(args.output_dir, exist_ok=True)
    pred_path = os.path.join(args.output_dir, f"{args.mode}_predictions.jsonl")
    span_path = os.path.join(args.output_dir, f"{args.mode}_spans.jsonl")

    predictions = []
    span_records = []
    failed = 0

    for i, record in enumerate(records):
        text = record.get("original_text", "")
        try:
            anon_text, spans = detect_and_anonymize_fn(text)
        except Exception as e:
            anon_text, spans = text, []
            failed += 1
            if failed <= 5:
                print(f"  [WARN] Failed on {record['id']}: {e}")

        predictions.append({"id": record["id"], "anonymized_text": anon_text})
        if args.save_spans:
            span_records.append({
                "id": record["id"],
                "detected_entities": [
                    {"start": s, "end": e, "type": t, "text": txt}
                    for s, e, t, txt in spans
                ],
            })

        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{len(records)}")

    with open(pred_path, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")
    print(f"Predictions saved to {pred_path}")

    if args.save_spans:
        with open(span_path, "w", encoding="utf-8") as f:
            for s in span_records:
                f.write(json.dumps(s) + "\n")
        print(f"Spans saved to {span_path}")

    if failed:
        print(f"  ({failed} records failed, fell back to original text)")


if __name__ == "__main__":
    main()

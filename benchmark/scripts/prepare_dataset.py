"""
SAHA-AL Benchmark — Dataset Preparation
=========================================
Reads anonymized_dataset_final.jsonl, infers entity types (the raw data has
type=UNKNOWN everywhere), computes span offsets, and writes train/val/test splits.

The type inference uses regex for structured types and spaCy NER for names/orgs/locations.

Usage:
  python -m scripts.prepare_dataset
  python -m scripts.prepare_dataset --input anonymized_dataset_final.jsonl --output-dir data
"""

import json
import os
import random
import re
from collections import Counter

GOLD_CUTOFF = 36000

# ── Type inference patterns ──

TYPE_PATTERNS = [
    ("EMAIL",       re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")),
    ("SSN",         re.compile(r"^\d{3}-\d{2}-\d{4}$")),
    ("CREDIT_CARD", re.compile(r"^\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{3,4}$")),
    ("IBAN",        re.compile(r"^[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]{0,16})?$")),
    ("IP_ADDRESS",  re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")),
    ("URL",         re.compile(r"^(https?://|www\.)\S+$", re.IGNORECASE)),
    ("PHONE",       re.compile(
        r"^(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}$"
    )),
    ("DATE",        re.compile(
        r"^\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}$"
        r"|^\d{4}[/.-]\d{1,2}[/.-]\d{1,2}$"
        r"|^(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?"
        r"|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?"
        r"|Dec(?:ember)?)\s+\d{1,2},?\s+\d{4}$"
        r"|^\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?"
        r"|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?"
        r"|Dec(?:ember)?)\s+\d{4}$"
        r"|^\d{1,2}(?:st|nd|rd|th)\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May"
        r"|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?"
        r"|Dec(?:ember)?)\s+\d{4}$",
        re.IGNORECASE,
    )),
    ("ACCOUNT_NUMBER", re.compile(r"^(?:account|acct)[#:\s]*\d{6,}$", re.IGNORECASE)),
    ("ID_NUMBER",   re.compile(r"^[A-Z]{1,3}\d{6,10}$")),
    ("ZIPCODE",     re.compile(r"^\d{5}(?:-\d{4})?$")),
]

_ADDRESS_RE = re.compile(
    r"^\d+\s+\w+\s+(St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Dr|Drive|Lane|Ln|Way|Ct|Court|Pl|Place|Apt|Suite)",
    re.IGNORECASE,
)
_FULLNAME_RE = re.compile(r"^[A-Z][a-z]+(\s+[A-Z][a-z]+){1,3}$")
_SINGLE_NAME_RE = re.compile(r"^[A-Z][a-z]{2,15}$")
_ORG_SUFFIXES = re.compile(
    r"\b(Inc|Corp|LLC|Ltd|Co|Company|Group|Foundation|Institute|University|Bank|Hospital|Agency)\b",
    re.IGNORECASE,
)


def infer_entity_type(text, nlp=None, original_context=""):
    """Fast entity type inference using regex + heuristics (no spaCy calls)."""
    text_stripped = text.strip()
    if not text_stripped:
        return "UNKNOWN"

    for etype, pattern in TYPE_PATTERNS:
        if pattern.match(text_stripped):
            return etype

    if _ADDRESS_RE.match(text_stripped):
        return "ADDRESS"

    if _ORG_SUFFIXES.search(text_stripped):
        return "ORGANIZATION"

    if _FULLNAME_RE.match(text_stripped):
        return "FULLNAME"

    if _SINGLE_NAME_RE.match(text_stripped):
        ctx = original_context.lower()
        name_lower = text_stripped.lower()
        name_pos = ctx.find(name_lower)
        if name_pos > 0:
            before = ctx[max(0, name_pos - 40):name_pos]
            if any(kw in before for kw in ["mr", "mrs", "ms", "dr", "name", "named", "called"]):
                return "FIRST_NAME"

    words = text_stripped.split()
    if 1 <= len(words) <= 4 and all(w[0].isupper() for w in words if len(w) > 0):
        if any(c.isdigit() for c in text_stripped):
            return "ADDRESS"
        return "FULLNAME"

    return "UNKNOWN"


def find_non_overlapping_span(text, token, occupied_spans):
    if not token:
        return None
    start = 0
    while True:
        start = text.find(token, start)
        if start == -1:
            return None
        end = start + len(token)
        overlap = any(not (end <= o_start or start >= o_end) for o_start, o_end in occupied_spans)
        if not overlap:
            return start, end
        start += 1


def prepare_dataset(input_file="anonymized_dataset_final.jsonl", output_dir="data"):
    print(f"Reading {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    print(f"  {len(lines)} records loaded.")

    print("Inferring entity types and formatting records...")

    all_entries = []
    type_distribution = Counter()
    unknown_count = 0
    span_warning_count = 0

    for idx, line in enumerate(lines):
        record = json.loads(line)
        entry_id = record.get("entry_id", idx)
        new_id = f"sample_{entry_id:05d}"
        orig_text = record.get("original_text", "")
        anon_text = record.get("anonymized_text", "")

        entities = []
        occupied_spans = []

        for ent in record.get("entities", []):
            text = ent.get("text", "")
            raw_type = ent.get("type", "UNKNOWN")

            if raw_type == "UNKNOWN" or not raw_type:
                ent_type = infer_entity_type(text, original_context=orig_text)
            else:
                ent_type = raw_type.strip().upper()

            type_distribution[ent_type] += 1
            if ent_type == "UNKNOWN":
                unknown_count += 1

            if not text:
                span_warning_count += 1
                continue

            start = ent.get("start")
            end = ent.get("end")
            if (
                isinstance(start, int)
                and isinstance(end, int)
                and 0 <= start < end <= len(orig_text)
                and orig_text[start:end] == text
                and not any(not (end <= o_start or start >= o_end) for o_start, o_end in occupied_spans)
            ):
                span = (start, end)
            else:
                span = find_non_overlapping_span(orig_text, text, occupied_spans)

            if span is None:
                span_warning_count += 1
                entities.append({
                    "text": text, "type": ent_type,
                    "start": -1, "end": -1, "invalid": True,
                })
            else:
                start, end = span
                occupied_spans.append((start, end))
                entities.append({
                    "text": text, "type": ent_type,
                    "start": start, "end": end,
                })

        all_entries.append({
            "id": new_id,
            "original_text": orig_text,
            "anonymized_text": anon_text,
            "entities": entities,
        })

        if (idx + 1) % 10000 == 0:
            print(f"  Processed {idx+1}/{len(lines)}")

    print(f"\nEntity type distribution ({sum(type_distribution.values())} total):")
    for ent_type, count in type_distribution.most_common():
        print(f"  {ent_type:20s} {count:6d}  ({count/sum(type_distribution.values())*100:.1f}%)")

    if unknown_count > 0:
        pct = unknown_count / sum(type_distribution.values()) * 100
        print(f"\n[WARNING] {unknown_count} entities still UNKNOWN ({pct:.1f}%)")
    if span_warning_count > 0:
        print(f"[WARNING] {span_warning_count} entities could not be aligned")

    print(f"\nSplitting (GOLD_CUTOFF={GOLD_CUTOFF})...")
    gold_entries = all_entries[:GOLD_CUTOFF]
    augmented_entries = all_entries[GOLD_CUTOFF:]

    random.seed(42)
    random.shuffle(gold_entries)

    total_gold = len(gold_entries)
    train_end = int(0.8 * total_gold)
    val_end = int(0.9 * total_gold)

    train_gold = gold_entries[:train_end]
    gold_val = gold_entries[train_end:val_end]
    gold_test = gold_entries[val_end:]
    train_split = train_gold + augmented_entries

    os.makedirs(output_dir, exist_ok=True)

    splits = {"train": train_split, "validation": gold_val, "test": gold_test}
    for split_name, data in splits.items():
        out_path = os.path.join(output_dir, f"{split_name}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in data:
                f.write(json.dumps(rec) + "\n")

    print(f"\n--- SPLIT STATS ---")
    print(f"Train: {len(train_split):>8,} ({len(train_gold)} gold + {len(augmented_entries)} augmented)")
    print(f"Val:   {len(gold_val):>8,} (gold only)")
    print(f"Test:  {len(gold_test):>8,} (gold only)")

    train_ids = set(r["id"] for r in train_split)
    test_ids = set(r["id"] for r in gold_test)
    assert len(train_ids & test_ids) == 0, "Leakage detected!"
    print("[OK] No train/test leakage.")

    test_types = Counter()
    for rec in gold_test:
        for e in rec["entities"]:
            test_types[e["type"]] += 1
    unknown_in_test = test_types.get("UNKNOWN", 0)
    total_in_test = sum(test_types.values())
    print(f"[INFO] Test set: {total_in_test} entities, {unknown_in_test} UNKNOWN ({unknown_in_test/total_in_test*100:.1f}%)")
    print("-------------------")
    print("Done!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="anonymized_dataset_final.jsonl")
    parser.add_argument("--output-dir", default="data")
    args = parser.parse_args()
    prepare_dataset(args.input, args.output_dir)

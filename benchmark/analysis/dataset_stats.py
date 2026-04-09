"""
SAHA-AL Benchmark — Dataset Statistics
========================================
Computes per-split statistics: record counts, entity counts, type distribution,
average text length, PII density, and train/test entity string overlap.

Usage:
  python -m analysis.dataset_stats \
      --train data/train.jsonl \
      --val data/validation.jsonl \
      --test data/test.jsonl \
      --output results/dataset_stats.json
"""

import argparse
import json
from collections import Counter

from eval.utils import load_jsonl


def compute_stats(records, split_name="unknown"):
    """Compute statistics for a single split."""
    total_records = len(records)
    total_entities = 0
    type_counter = Counter()
    text_lengths = []
    entities_per_record = []
    entity_strings = set()

    for rec in records:
        text = rec.get("original_text", "")
        entities = rec.get("entities", [])

        text_lengths.append(len(text.split()))
        entities_per_record.append(len(entities))
        total_entities += len(entities)

        for ent in entities:
            etype = ent.get("type", "UNKNOWN")
            type_counter[etype] += 1
            if ent.get("text"):
                entity_strings.add(ent["text"].lower())

    avg_length = sum(text_lengths) / len(text_lengths) if text_lengths else 0
    avg_entities = sum(entities_per_record) / len(entities_per_record) if entities_per_record else 0
    pii_density = total_entities / sum(text_lengths) if sum(text_lengths) else 0

    return {
        "split": split_name,
        "records": total_records,
        "total_entities": total_entities,
        "avg_text_length_words": round(avg_length, 1),
        "avg_entities_per_record": round(avg_entities, 2),
        "pii_density": round(pii_density, 4),
        "type_distribution": dict(type_counter.most_common()),
        "unique_entity_strings": len(entity_strings),
        "_entity_strings": entity_strings,
    }


def main():
    parser = argparse.ArgumentParser(description="SAHA-AL dataset statistics")
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output", default="results/dataset_stats.json")
    args = parser.parse_args()

    splits = {
        "train": load_jsonl(args.train),
        "validation": load_jsonl(args.val),
        "test": load_jsonl(args.test),
    }

    results = {}
    entity_string_sets = {}

    for name, records in splits.items():
        stats = compute_stats(records, name)
        entity_string_sets[name] = stats.pop("_entity_strings")
        results[name] = stats

    train_strings = entity_string_sets.get("train", set())
    test_strings = entity_string_sets.get("test", set())
    overlap = train_strings & test_strings
    results["train_test_entity_overlap"] = {
        "overlapping_strings": len(overlap),
        "test_unique_strings": len(test_strings),
        "overlap_ratio": round(len(overlap) / len(test_strings) * 100, 2) if test_strings else 0,
    }

    print("\n" + "=" * 60)
    print("  SAHA-AL Dataset Statistics")
    print("=" * 60)
    for name in ["train", "validation", "test"]:
        s = results[name]
        print(f"\n  {name}:")
        print(f"    Records:           {s['records']:>8,}")
        print(f"    Entities:          {s['total_entities']:>8,}")
        print(f"    Avg words/record:  {s['avg_text_length_words']:>8.1f}")
        print(f"    Avg ents/record:   {s['avg_entities_per_record']:>8.2f}")
        print(f"    PII density:       {s['pii_density']:>8.4f}")
        print(f"    Entity types:      {len(s['type_distribution']):>8}")

    ov = results["train_test_entity_overlap"]
    print(f"\n  Train/Test entity string overlap: {ov['overlap_ratio']:.1f}% ({ov['overlapping_strings']}/{ov['test_unique_strings']})")
    print("=" * 60)

    import os
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Stats saved to {args.output}")


if __name__ == "__main__":
    main()

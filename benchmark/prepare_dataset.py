import json
import os
import random

GOLD_CUTOFF = 36000

def prepare_dataset(input_file="anonymized_dataset_final.jsonl", output_dir="data"):
    print(f"Reading {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    print("Formatting records...")
    
    all_entries = []
    
    # Dataset integrity check
    unknown_types = sum(
        1 for r in lines for e in json.loads(r).get("entities", [])
        if e.get("type", "UNKNOWN") == "UNKNOWN"
    )
    if unknown_types > 0:
        print(f"[WARNING] {unknown_types} UNKNOWN entity types detected in dataset")

    # The dataset lacks robust metadata. We rely on insertion-order.
    # The first ~36k entries are gold, the rest (120k - 36k) are augmented.
    for idx, line in enumerate(lines):
        record = json.loads(line)
        
        # Format ID
        entry_id = record.get("entry_id", idx)
        new_id = f"sample_{entry_id:05d}"
        
        orig_text = record.get("original_text", "")
        anon_text = record.get("anonymized_text", "")
        
        entities = []
        for ent in record.get("entities", []):
            text = ent.get("text", "")
            ent_type = ent.get("type", "UNKNOWN")
            
            start_idx = orig_text.find(text)
            if start_idx != -1:
                end_idx = start_idx + len(text)
                entities.append({
                    "text": text,
                    "type": ent_type,
                    "start": start_idx,
                    "end": end_idx
                })
        
        formatted_record = {
            "id": new_id,
            "original_text": orig_text,
            "anonymized_text": anon_text,
            "entities": entities
        }
        all_entries.append(formatted_record)

    print(f"Splitting dataset via index cutoff (GOLD_CUTOFF = {GOLD_CUTOFF})...")
    gold_entries = all_entries[:GOLD_CUTOFF]
    augmented_entries = all_entries[GOLD_CUTOFF:]
    
    # Deterministic split (Gold scale)
    random.seed(42)  # Benchmark must be reproducible
    random.shuffle(gold_entries)
    
    total_gold = len(gold_entries)
    train_end = int(0.8 * total_gold)
    val_end = int(0.9 * total_gold)
    
    train_gold = gold_entries[:train_end]
    gold_val = gold_entries[train_end:val_end]
    gold_test = gold_entries[val_end:]
    
    # All augmented data dumps directly into the train split
    train_split = train_gold + augmented_entries
    
    os.makedirs(output_dir, exist_ok=True)
    
    splits = {
        "train": train_split,
        "validation": gold_val,
        "test": gold_test
    }
    
    for split_name, data in splits.items():
        out_path = os.path.join(output_dir, f"{split_name}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in data:
                f.write(json.dumps(rec) + "\n")
        
    print("\n--- SANITY STATS ---")
    print(f"Train size: {len(train_split)} ({len(train_gold)} gold + {len(augmented_entries)} augmented)")
    print(f"Val size:   {len(gold_val)} (gold only)")
    print(f"Test size:  {len(gold_test)} (gold only)")
    
    # Check leakage
    train_ids = set(r["id"] for r in train_split)
    test_ids = set(r["id"] for r in gold_test)

    assert len(train_ids & test_ids) == 0, "Leakage detected!"
    print("[OK] Test set is strictly segregated from train set.")
    print("--------------------\n")

    print("[INFO] Dataset lacks source_id/is_gold metadata; index-based fallback splitting used.")
    print("Dataset preparation complete!")

if __name__ == "__main__":
    prepare_dataset()

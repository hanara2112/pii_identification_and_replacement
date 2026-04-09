"""
SAHA-AL Benchmark — Remaining Evaluation Steps (Kaggle Notebook)
=================================================================
Run this on Kaggle with GPU enabled.

Upload these files to Kaggle as a dataset:
  - data/test.jsonl
  - data/train.jsonl
  - data/validation.jsonl
  - predictions/predictions_bart-base-pii.jsonl
  - predictions/predictions_flan-t5-small-pii.jsonl
  - predictions/predictions_t5-small-pii.jsonl
  - predictions/predictions_distilbart-pii.jsonl
  - predictions/predictions_t5-efficient-tiny-pii.jsonl
  - predictions/spacy_predictions.jsonl
  - predictions/presidio_predictions.jsonl
  - predictions/regex_predictions.jsonl
  - predictions/regex_spans.jsonl
  - predictions/spacy_spans.jsonl
  - predictions/presidio_spans.jsonl

Steps covered:
  4. Task 1 — Detection evaluation
  5. Task 3 — Privacy (CRR-3 + ERA + UAC)
  6. Bootstrap confidence intervals
  7. Failure taxonomy
  8. Pareto frontier analysis
  9. BERT NER training + prediction (GPU)
"""

# ============================================================
# CELL 1: Install dependencies
# ============================================================
# !pip install -q sentence-transformers faker bert_score spacy
# !python -m spacy download en_core_web_lg

import json
import os
import re
import sys
import random
import numpy as np
from collections import Counter
from pathlib import Path

# ── Adjust paths for Kaggle ──
# Change these if your Kaggle dataset is mounted differently
DATA_DIR = "/kaggle/input/saha-al-benchmark"  # <-- UPDATE THIS
WORK_DIR = "/kaggle/working"

# If running locally, uncomment:
# DATA_DIR = "."
# WORK_DIR = "."

GOLD_TEST = os.path.join(DATA_DIR, "data/test.jsonl")
GOLD_TRAIN = os.path.join(DATA_DIR, "data/train.jsonl")
GOLD_VAL = os.path.join(DATA_DIR, "data/validation.jsonl")
PRED_DIR = os.path.join(DATA_DIR, "predictions")
RESULTS_DIR = os.path.join(WORK_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved to {path}")


# ============================================================
# CELL 2: STEP 4 — Task 1: Detection Evaluation
# ============================================================
print("=" * 60)
print("  STEP 4: Task 1 — PII Detection Evaluation")
print("=" * 60)


def span_match(pred, gold, mode="exact"):
    if mode == "exact":
        return pred["start"] == gold["start"] and pred["end"] == gold["end"]
    elif mode == "partial":
        overlap = max(0, min(pred["end"], gold["end"]) - max(pred["start"], gold["start"]))
        union = max(pred["end"], gold["end"]) - min(pred["start"], gold["start"])
        return (overlap / union) > 0.5 if union > 0 else False
    elif mode == "type_aware":
        return (pred["start"] == gold["start"] and pred["end"] == gold["end"]
                and pred.get("type", "").upper() == gold.get("type", "").upper())


def compute_span_f1(gold_records, predictions, mode="exact"):
    total_tp, total_fp, total_fn = 0, 0, 0
    for g, p in zip(gold_records, predictions):
        gold_spans = [e for e in g.get("entities", []) if e.get("start", -1) >= 0]
        pred_spans = p.get("detected_entities", [])
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
        total_tp += len(matched_gold)
        total_fp += len(pred_spans) - len(matched_pred)
        total_fn += len(gold_spans) - len(matched_gold)

    p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
    r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
    f1 = 2 * p * r / (p + r) if (p + r) else 0
    return {"precision": round(p * 100, 2), "recall": round(r * 100, 2), "f1": round(f1 * 100, 2)}


gold_test = load_jsonl(GOLD_TEST)
detection_results = {}

for mode_name in ["regex", "spacy", "presidio"]:
    span_file = os.path.join(PRED_DIR, f"{mode_name}_spans.jsonl")
    if not os.path.exists(span_file):
        print(f"  [SKIP] {span_file} not found")
        continue
    preds = load_jsonl(span_file)
    pred_map = {p["id"]: p for p in preds}
    aligned_g = []
    aligned_p = []
    for g in gold_test:
        gid = g.get("id")
        if gid in pred_map:
            aligned_g.append(g)
            aligned_p.append(pred_map[gid])

    det = {}
    for m in ["exact", "partial", "type_aware"]:
        det[m] = compute_span_f1(aligned_g, aligned_p, mode=m)

    detection_results[mode_name] = det
    print(f"\n  {mode_name}:")
    for m in ["exact", "partial", "type_aware"]:
        d = det[m]
        print(f"    {m:12s}  P={d['precision']:5.2f}  R={d['recall']:5.2f}  F1={d['f1']:5.2f}")

save_json(detection_results, os.path.join(RESULTS_DIR, "task1_detection.json"))


# ============================================================
# CELL 3: STEP 5 — Task 3: Privacy (CRR-3 + ERA + UAC)
# ============================================================
print("\n" + "=" * 60)
print("  STEP 5: Task 3 — Privacy Risk Assessment")
print("=" * 60)

# ── CRR-3 ──
def get_capitalized_ngrams(text, n=3):
    tokens = text.split()
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    return [ng for ng in ngrams if any(len(t) > 0 and t[0].isupper() for t in ng)]


def crr3(gold_records, predictions):
    survived, total = 0, 0
    for g, p in zip(gold_records, predictions):
        orig_3grams = get_capitalized_ngrams(g.get("original_text", ""), n=3)
        pred_text = (p.get("anonymized_text") or "").lower()
        for ng in orig_3grams:
            total += 1
            if " ".join(ng).lower() in pred_text:
                survived += 1
    return round(survived / total * 100, 2) if total else 0


# ── ERA ──
def entity_recovery_attack(gold_records, predictions, train_records, top_k=5):
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import cos_sim
    model = SentenceTransformer("all-MiniLM-L6-v2")

    type_pools = {}
    for r in train_records:
        for e in r.get("entities", []):
            etype = e.get("type", "UNKNOWN")
            text = e.get("text", "")
            if text:
                type_pools.setdefault(etype, set()).add(text)
    type_pools = {k: list(v) for k, v in type_pools.items()}

    top1, top5, total = 0, 0, 0
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
                top1 += 1
            if original_val in [pool[ranked[i]] for i in range(min(top_k, len(ranked)))]:
                top5 += 1

    return {
        "era_top1": round(top1 / total * 100, 2) if total else 0,
        "era_top5": round(top5 / total * 100, 2) if total else 0,
        "total": total,
    }


# ── UAC ──
TYPE_HINTS = {
    "EMAIL": ["email", "@", "mail"], "PHONE": ["phone", "call", "tel"],
    "SSN": ["ssn", "social security"], "DATE": ["born", "date", "birthday"],
    "ADDRESS": ["lives", "address", "street", "road"],
}


def unique_attribute_combination_rate(gold_records, predictions):
    combos = Counter()
    record_combos = []
    for g, p in zip(gold_records, predictions):
        pred_text = p.get("anonymized_text", "")
        surviving = []
        for ent in g.get("entities", []):
            if ent.get("start", -1) < 0:
                continue
            if ent["text"].lower() in pred_text.lower():
                surviving.append(ent["type"])
            else:
                start = max(0, ent["start"] - 30)
                end = min(len(g["original_text"]), ent["end"] + 30)
                ctx = g["original_text"][start:end].lower()
                if any(h in ctx for h in TYPE_HINTS.get(ent.get("type", ""), [])):
                    surviving.append(ent["type"])
        combo = tuple(sorted(surviving))
        combos[combo] += 1
        record_combos.append(combo)
    unique = sum(1 for c in record_combos if combos[c] == 1)
    return round(unique / len(record_combos) * 100, 2) if record_combos else 0


# Run privacy eval on key systems
train_records = load_jsonl(GOLD_TRAIN)
privacy_results = {}

systems = {
    "bart-base-pii": "predictions_bart-base-pii.jsonl",
    "flan-t5-small-pii": "predictions_flan-t5-small-pii.jsonl",
    "spacy": "spacy_predictions.jsonl",
    "presidio": "presidio_predictions.jsonl",
}

for name, pred_file in systems.items():
    pred_path = os.path.join(PRED_DIR, pred_file)
    if not os.path.exists(pred_path):
        print(f"  [SKIP] {pred_file}")
        continue

    preds = load_jsonl(pred_path)
    pred_map = {p["id"]: p for p in preds}
    aligned_g = [g for g in gold_test if g["id"] in pred_map]
    aligned_p = [pred_map[g["id"]] for g in aligned_g]

    print(f"\n  Evaluating privacy: {name}...")

    crr = crr3(aligned_g, aligned_p)
    print(f"    CRR-3:  {crr:.2f}%")

    # ERA — only on first 500 records (GPU-heavy)
    era = entity_recovery_attack(aligned_g[:500], aligned_p[:500], train_records)
    print(f"    ERA@1:  {era['era_top1']:.2f}%")
    print(f"    ERA@5:  {era['era_top5']:.2f}%")

    uac = unique_attribute_combination_rate(aligned_g, aligned_p)
    print(f"    UAC:    {uac:.2f}%")

    privacy_results[name] = {"crr3": crr, "era": era, "uac": uac}

save_json(privacy_results, os.path.join(RESULTS_DIR, "task3_privacy.json"))


# ============================================================
# CELL 4: STEP 6 — Bootstrap Confidence Intervals
# ============================================================
print("\n" + "=" * 60)
print("  STEP 6: Bootstrap Confidence Intervals")
print("=" * 60)


def exact_entity_match(ent_text, pred_text):
    if not ent_text or not pred_text:
        return False
    return re.search(rf"(?<!\w){re.escape(ent_text)}(?!\w)", pred_text, re.IGNORECASE) is not None


def elr_metric(gold, preds):
    leaked, total = 0, 0
    for g, p in zip(gold, preds):
        pt = (p.get("anonymized_text") or "").strip()
        for ent in g.get("entities", []):
            total += 1
            if exact_entity_match(ent.get("text", ""), pt):
                leaked += 1
    return (leaked / total * 100) if total else 0


def bootstrap_ci(gold, preds, metric_fn, n_boot=1000, seed=42):
    rng = random.Random(seed)
    n = len(gold)
    scores = []
    for _ in range(n_boot):
        idx = [rng.randint(0, n-1) for _ in range(n)]
        scores.append(metric_fn([gold[i] for i in idx], [preds[i] for i in idx]))
    scores = np.array(scores)
    return {
        "mean": round(float(scores.mean()), 3),
        "ci_lower": round(float(np.percentile(scores, 2.5)), 3),
        "ci_upper": round(float(np.percentile(scores, 97.5)), 3),
    }


bart_preds = load_jsonl(os.path.join(PRED_DIR, "predictions_bart-base-pii.jsonl"))
pred_map = {p["id"]: p for p in bart_preds}
ci_g = [g for g in gold_test if g["id"] in pred_map]
ci_p = [pred_map[g["id"]] for g in ci_g]

print("  Bootstrapping BART-base ELR (1000 iterations)...")
ci_elr = bootstrap_ci(ci_g, ci_p, elr_metric)
print(f"    ELR:  {ci_elr['mean']:.3f}% [{ci_elr['ci_lower']:.3f}, {ci_elr['ci_upper']:.3f}]")

print("  Bootstrapping BART-base CRR-3...")
ci_crr = bootstrap_ci(ci_g, ci_p, crr3)
print(f"    CRR-3: {ci_crr['mean']:.3f}% [{ci_crr['ci_lower']:.3f}, {ci_crr['ci_upper']:.3f}]")

save_json({"bart-base-pii": {"elr": ci_elr, "crr3": ci_crr}},
          os.path.join(RESULTS_DIR, "bootstrap_cis.json"))


# ============================================================
# CELL 5: STEP 7 — Failure Taxonomy
# ============================================================
print("\n" + "=" * 60)
print("  STEP 7: Failure Taxonomy")
print("=" * 60)

from difflib import SequenceMatcher

FORMAT_PATTERNS = {
    "EMAIL": re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$"),
    "PHONE": re.compile(r"^[\d\s\+\-\(\)\.]{5,20}$"),
    "SSN": re.compile(r"^\d{3}[- ]?\d{2}[- ]?\d{4}$"),
    "CREDIT_CARD": re.compile(r"^[\d\s\-]{12,25}$"),
    "ZIPCODE": re.compile(r"^[A-Z0-9\- ]{3,12}$", re.IGNORECASE),
    "DATE": re.compile(r"\d"),
}


def extract_replacement(original, prediction, start, end):
    sm = SequenceMatcher(None, original, prediction)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "replace" and i1 <= start and i2 >= end:
            return prediction[j1:j2]
    return None


def classify_failures(gold_records, predictions):
    counts = {"clean": 0, "full_leak": 0, "boundary": 0,
              "type_confusion": 0, "ghost_leak": 0, "format_break": 0}
    for g, p in zip(gold_records, predictions):
        orig = g.get("original_text", "")
        pred = (p.get("anonymized_text") or "").strip()
        for ent in g.get("entities", []):
            if ent.get("start", -1) < 0:
                continue
            ent_text = ent["text"]
            ent_type = ent.get("type", "UNKNOWN")

            if re.search(rf"(?<!\w){re.escape(ent_text)}(?!\w)", pred, re.IGNORECASE):
                counts["full_leak"] += 1
                continue

            tokens = re.findall(r"\w+", ent_text)
            partial = [t for t in tokens if len(t) > 2 and
                       re.search(rf"(?<!\w){re.escape(t)}(?!\w)", pred, re.IGNORECASE)]
            if partial and len(partial) < len(tokens):
                counts["boundary"] += 1
                continue

            pat = FORMAT_PATTERNS.get(ent_type)
            if pat:
                repl = extract_replacement(orig, pred, ent["start"], ent.get("end", ent["start"]))
                if repl and not pat.search(repl):
                    counts["format_break"] += 1
                    continue

            ctx_s = max(0, ent["start"] - 50)
            ctx_e = min(len(orig), ent.get("end", ent["start"]) + 50)
            orig_ctx = orig[ctx_s:ctx_e].lower()
            pred_ctx_s = max(0, ent["start"] - 50)
            pred_ctx_e = min(len(pred), ent.get("end", ent["start"]) + 50)
            if pred_ctx_e <= len(pred):
                pred_ctx = pred[pred_ctx_s:pred_ctx_e].lower()
                ctx_tok = set(re.findall(r"\w+", orig_ctx)) - set(re.findall(r"\w+", ent_text.lower()))
                pred_tok = set(re.findall(r"\w+", pred_ctx))
                if ctx_tok and len(ctx_tok & pred_tok) / len(ctx_tok) > 0.8:
                    counts["ghost_leak"] += 1
                    continue

            counts["clean"] += 1
    return counts


# Run on all systems
taxonomy_results = {}
all_pred_files = {
    "bart-base-pii": "predictions_bart-base-pii.jsonl",
    "flan-t5-small-pii": "predictions_flan-t5-small-pii.jsonl",
    "t5-small-pii": "predictions_t5-small-pii.jsonl",
    "spacy": "spacy_predictions.jsonl",
    "presidio": "presidio_predictions.jsonl",
    "regex": "regex_predictions.jsonl",
}

for name, pf in all_pred_files.items():
    pp = os.path.join(PRED_DIR, pf)
    if not os.path.exists(pp):
        continue
    preds = load_jsonl(pp)
    pm = {p["id"]: p for p in preds}
    ag = [g for g in gold_test if g["id"] in pm]
    ap = [pm[g["id"]] for g in ag]

    counts = classify_failures(ag, ap)
    total = sum(counts.values())
    taxonomy_results[name] = counts

    print(f"\n  {name}:")
    for cat, c in counts.items():
        print(f"    {cat:15s} {c:5d}  ({c/total*100:5.1f}%)")

save_json(taxonomy_results, os.path.join(RESULTS_DIR, "failure_taxonomy_all.json"))


# ============================================================
# CELL 6: STEP 8 — Pareto Frontier
# ============================================================
print("\n" + "=" * 60)
print("  STEP 8: Pareto Frontier Analysis")
print("=" * 60)

all_results = {
    "regex":            {"elr": 83.39, "bertscore": 98.15},
    "spacy":            {"elr": 26.44, "bertscore": 91.86},
    "presidio":         {"elr": 33.77, "bertscore": 90.04},
    "bart-base":        {"elr": 0.93,  "bertscore": 92.74},
    "flan-t5-small":    {"elr": 0.99,  "bertscore": 92.47},
    "distilbart":       {"elr": 1.23,  "bertscore": 86.34},
    "t5-small":         {"elr": 1.54,  "bertscore": 92.59},
    "t5-eff-tiny":      {"elr": 4.14,  "bertscore": 92.57},
}

points = np.array([(1 - r["elr"]/100, r["bertscore"]/100) for r in all_results.values()])
models = list(all_results.keys())
pareto = []
for i, (px, py) in enumerate(points):
    dominated = any(points[j][0] >= px and points[j][1] >= py and
                     (points[j][0] > px or points[j][1] > py)
                     for j in range(len(points)) if j != i)
    if not dominated:
        pareto.append(models[i])

print(f"  Pareto-optimal: {pareto}")

# PUS sweep
for lam in [0.0, 0.3, 0.5, 0.7, 1.0]:
    row = []
    for name, r in all_results.items():
        priv = 1 - r["elr"] / 100
        util = r["bertscore"] / 100
        pus = lam * priv + (1 - lam) * util
        row.append(f"{name}={pus:.3f}")
    print(f"  λ={lam:.1f}: {', '.join(row)}")

# Plot
try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 7))
    for i, (name, r) in enumerate(all_results.items()):
        x, y = 1 - r["elr"]/100, r["bertscore"]/100
        color = "red" if name in pareto else "steelblue"
        marker = "*" if name in pareto else "o"
        size = 200 if name in pareto else 100
        ax.scatter(x, y, c=color, s=size, marker=marker, zorder=5)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=9)

    pareto_pts = sorted([(1-all_results[m]["elr"]/100, all_results[m]["bertscore"]/100) for m in pareto])
    if len(pareto_pts) > 1:
        px, py = zip(*pareto_pts)
        ax.plot(px, py, "r--", alpha=0.6, linewidth=2, label="Pareto frontier")

    ax.set_xlabel("Privacy (1 − ELR)", fontsize=13)
    ax.set_ylabel("Utility (BERTScore F1 / 100)", fontsize=13)
    ax.set_title("Privacy-Utility Pareto Frontier", fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0.5, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "pareto_frontier.png"), dpi=150)
    plt.show()
    print(f"  Plot saved.")
except ImportError:
    print("  [SKIP] matplotlib not available")


# ============================================================
# CELL 7: STEP 9 — BERT NER Training (GPU) [OPTIONAL]
# ============================================================
print("\n" + "=" * 60)
print("  STEP 9: BERT NER Baseline (needs GPU)")
print("=" * 60)

TRAIN_BERT = False  # Set to True to train

if TRAIN_BERT:
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForTokenClassification, AutoTokenizer,
        TrainingArguments, Trainer, DataCollatorForTokenClassification,
    )

    ENTITY_TYPES = [
        "FULLNAME", "FIRST_NAME", "LAST_NAME", "ID_NUMBER", "PASSPORT", "SSN",
        "PHONE", "EMAIL", "ADDRESS", "DATE", "TIME", "LOCATION", "ORGANIZATION",
        "ACCOUNT_NUMBER", "CREDIT_CARD", "ZIPCODE",
    ]
    BIO_LABELS = ["O"] + [f"B-{t}" for t in ENTITY_TYPES] + [f"I-{t}" for t in ENTITY_TYPES]
    LABEL2ID = {l: i for i, l in enumerate(BIO_LABELS)}
    ID2LABEL = {i: l for l, i in LABEL2ID.items()}

    class NERDataset(Dataset):
        def __init__(self, records, tokenizer, max_length=128):
            self.tokenizer = tokenizer
            self.samples = []
            for rec in records:
                text = rec.get("original_text", "")
                if not text:
                    continue
                char_labels = ["O"] * len(text)
                for ent in sorted(rec.get("entities", []), key=lambda e: e.get("start", -1)):
                    s, e = ent.get("start", -1), ent.get("end", -1)
                    etype = ent.get("type", "")
                    if s < 0 or etype not in ENTITY_TYPES:
                        continue
                    char_labels[s] = f"B-{etype}"
                    for i in range(s+1, min(e, len(text))):
                        char_labels[i] = f"I-{etype}"
                enc = tokenizer(text, truncation=True, max_length=max_length,
                                return_offsets_mapping=True, return_tensors=None)
                offsets = enc.pop("offset_mapping")
                token_labels = []
                for start, end in offsets:
                    if start == end:
                        token_labels.append(-100)
                    else:
                        token_labels.append(LABEL2ID.get(char_labels[start], 0))
                enc["labels"] = token_labels
                self.samples.append(enc)

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            item = self.samples[idx]
            return {k: item[k] for k in ["input_ids", "attention_mask", "labels"]}

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    model = AutoModelForTokenClassification.from_pretrained(
        "bert-base-cased", num_labels=len(BIO_LABELS), id2label=ID2LABEL, label2id=LABEL2ID)

    train_data = load_jsonl(GOLD_TRAIN)
    val_data = load_jsonl(GOLD_VAL)

    print(f"  Building datasets (train={len(train_data)}, val={len(val_data)})...")
    train_ds = NERDataset(train_data, tokenizer)
    val_ds = NERDataset(val_data, tokenizer)
    print(f"  Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    training_args = TrainingArguments(
        output_dir=os.path.join(WORK_DIR, "bert_ner"),
        num_train_epochs=5, per_device_train_batch_size=32,
        per_device_eval_batch_size=64, learning_rate=5e-5,
        weight_decay=0.01, warmup_ratio=0.1,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        logging_steps=200, fp16=torch.cuda.is_available(), report_to="none",
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True),
        tokenizer=tokenizer,
    )

    print("  Training...")
    trainer.train()

    save_path = os.path.join(WORK_DIR, "bert_ner", "best_model")
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"  Model saved to {save_path}")
else:
    print("  Set TRAIN_BERT = True to run BERT NER training")


# ============================================================
# CELL 8: Summary
# ============================================================
print("\n" + "=" * 60)
print("  ALL DONE — Results Summary")
print("=" * 60)
print(f"  Results saved to: {RESULTS_DIR}/")
for f in sorted(os.listdir(RESULTS_DIR)):
    if f.endswith(".json") or f.endswith(".png"):
        print(f"    {f}")

"""
SAHA-AL Benchmark — BERT NER Baseline
=======================================
Fine-tunes bert-base-cased for token classification (BIO scheme)
on the SAHA-AL training data, then uses detected spans + Faker for replacement.

Outputs both detection spans (Task 1) and anonymized text (Task 2).

Usage:
  # Train
  python -m baselines.bert_ner_baseline \
      --train data/train.jsonl --val data/validation.jsonl \
      --output-dir checkpoints/bert_ner

  # Predict
  python -m baselines.bert_ner_baseline \
      --predict data/test.jsonl \
      --model-dir checkpoints/bert_ner \
      --output predictions/bert_ner_predictions.jsonl \
      --save-spans
"""

import argparse
import json
import os
import re

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)

from eval.utils import load_jsonl, save_jsonl

# 16 clean NER types (excluding TITLE, GENDER, NUMBER, OTHER_PII, UNKNOWN)
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
        self.max_length = max_length
        self.samples = self._build_samples(records)

    def _build_samples(self, records):
        samples = []
        for rec in records:
            text = rec.get("original_text", "")
            entities = rec.get("entities", [])
            if not text:
                continue

            char_labels = ["O"] * len(text)
            for ent in sorted(entities, key=lambda e: e.get("start", -1)):
                s, e = ent.get("start", -1), ent.get("end", -1)
                etype = ent.get("type", "")
                if s < 0 or e < 0 or etype not in ENTITY_TYPES:
                    continue
                char_labels[s] = f"B-{etype}"
                for i in range(s + 1, min(e, len(text))):
                    char_labels[i] = f"I-{etype}"

            encoding = self.tokenizer(
                text, truncation=True, max_length=self.max_length,
                return_offsets_mapping=True, return_tensors=None,
            )
            offsets = encoding.pop("offset_mapping")

            token_labels = []
            for start, end in offsets:
                if start == end:
                    token_labels.append(-100)
                else:
                    label_str = char_labels[start]
                    token_labels.append(LABEL2ID.get(label_str, 0))

            encoding["labels"] = token_labels
            encoding["text"] = text
            samples.append(encoding)

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            "labels": item["labels"],
        }


def train_ner(train_path, val_path, output_dir, epochs=5, batch_size=16, lr=5e-5):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    model = AutoModelForTokenClassification.from_pretrained(
        "bert-base-cased", num_labels=len(BIO_LABELS),
        id2label=ID2LABEL, label2id=LABEL2ID,
    )

    train_records = load_jsonl(train_path)
    val_records = load_jsonl(val_path)

    train_ds = NERDataset(train_records, tokenizer)
    val_ds = NERDataset(val_records, tokenizer)

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=100,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(os.path.join(output_dir, "best_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "best_model"))
    print(f"Model saved to {output_dir}/best_model")


def predict_spans(text, model, tokenizer, device):
    """Run NER inference, return list of (start, end, type, text) spans."""
    encoding = tokenizer(
        text, truncation=True, max_length=128,
        return_offsets_mapping=True, return_tensors="pt",
    )
    offsets = encoding.pop("offset_mapping")[0].tolist()
    inputs = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
    predictions = torch.argmax(logits, dim=-1)[0].tolist()

    spans = []
    current_entity = None
    current_start = None

    for idx, (pred_id, (char_start, char_end)) in enumerate(zip(predictions, offsets)):
        if char_start == char_end:
            continue
        label = ID2LABEL.get(pred_id, "O")

        if label.startswith("B-"):
            if current_entity:
                spans.append((current_start, offsets[idx - 1][1], current_entity, text[current_start:offsets[idx - 1][1]]))
            current_entity = label[2:]
            current_start = char_start
        elif label.startswith("I-") and current_entity == label[2:]:
            continue
        else:
            if current_entity:
                prev_end = offsets[idx - 1][1] if idx > 0 else char_end
                spans.append((current_start, prev_end, current_entity, text[current_start:prev_end]))
                current_entity = None

    if current_entity:
        last_end = offsets[-1][1]
        spans.append((current_start, last_end, current_entity, text[current_start:last_end]))

    return spans


def run_prediction(test_path, model_dir, output_path, save_spans=False):
    try:
        from faker import Faker
        fake = Faker()
        Faker.seed(42)
    except ImportError:
        print("pip install faker")
        exit(1)

    from baselines.regex_faker_baseline import get_faker_replacement

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir).to(device)
    model.eval()

    records = load_jsonl(test_path)
    predictions = []
    span_records = []

    for i, rec in enumerate(records):
        text = rec.get("original_text", "")
        spans = predict_spans(text, model, tokenizer, device)

        anon = text
        for start, end, etype, _ in reversed(spans):
            replacement = get_faker_replacement(etype)
            anon = anon[:start] + replacement + anon[end:]

        predictions.append({"id": rec["id"], "anonymized_text": anon})
        if save_spans:
            span_records.append({
                "id": rec["id"],
                "detected_entities": [
                    {"start": s, "end": e, "type": t, "text": txt}
                    for s, e, t, txt in spans
                ],
            })
        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{len(records)}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_jsonl(predictions, output_path)
    print(f"Predictions saved to {output_path}")

    if save_spans:
        span_path = output_path.replace(".jsonl", "_spans.jsonl")
        save_jsonl(span_records, span_path)
        print(f"Spans saved to {span_path}")


def main():
    parser = argparse.ArgumentParser(description="SAHA-AL BERT NER baseline")
    parser.add_argument("--train", default=None, help="Train JSONL (for training mode)")
    parser.add_argument("--val", default=None, help="Validation JSONL")
    parser.add_argument("--predict", default=None, help="Test JSONL (for prediction mode)")
    parser.add_argument("--model-dir", default="checkpoints/bert_ner/best_model")
    parser.add_argument("--output-dir", default="checkpoints/bert_ner")
    parser.add_argument("--output", default="predictions/bert_ner_predictions.jsonl")
    parser.add_argument("--save-spans", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()

    if args.train:
        print("Training BERT NER...")
        train_ner(args.train, args.val, args.output_dir, args.epochs, args.batch_size, args.lr)
    elif args.predict:
        print("Running BERT NER prediction...")
        run_prediction(args.predict, args.model_dir, args.output, args.save_spans)
    else:
        parser.error("Provide --train (training) or --predict (inference)")


if __name__ == "__main__":
    main()

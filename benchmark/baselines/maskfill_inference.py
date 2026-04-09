"""
SAHA-AL Benchmark — pipeline_maskfill Inference (Cross-Dataset Baselines)
==========================================================================
Loads trained pipeline_maskfill Models 1 and 2 and runs them on the
SAHA-AL test set. These models were trained on ai4privacy/pii-masking-400k,
so results represent cross-dataset zero-shot transfer.

Requires: pipeline_maskfill checkpoints under CHECKPOINT_BASE_DIR.

Usage:
  python -m baselines.maskfill_inference \
      --gold data/test.jsonl \
      --model 1 \
      --checkpoint-dir ../pipeline_maskfill/outputs/model1_baseline \
      --output predictions/predictions_maskfill_baseline.jsonl

  python -m baselines.maskfill_inference \
      --gold data/test.jsonl \
      --model 2 \
      --checkpoint-dir ../pipeline_maskfill/outputs/model2_advanced \
      --output predictions/predictions_maskfill_dp.jsonl
"""

import argparse
import json
import os
import sys

from eval.utils import load_jsonl, save_jsonl


# Entity type mapping: pipeline_maskfill types -> SAHA-AL types
MASKFILL_TO_SAHA = {
    "PERSON": "FULLNAME",
    "LOC": "LOCATION",
    "ORG": "ORGANIZATION",
    "DATE": "DATE",
    "PHONE": "PHONE",
    "EMAIL": "EMAIL",
    "SSN": "SSN",
    "CREDIT_CARD": "CREDIT_CARD",
    "ADDRESS": "ADDRESS",
    "IP_ADDRESS": "IP_ADDRESS",
    "IBAN": "IBAN",
    "PASSPORT": "PASSPORT",
    "DRIVER_LICENSE": "ID_NUMBER",
    "USERNAME": "USERNAME",
    "URL": "URL",
    "MEDICAL": "OTHER_PII",
    "ACCOUNT": "ACCOUNT_NUMBER",
    "BUILDING": "LOCATION",
    "POSTCODE": "ZIPCODE",
}


def load_model1(checkpoint_dir, device="cpu"):
    """Load Model 1 (DeBERTa NER + Flan-T5 fill) from checkpoint directory."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "pipeline_maskfill", "src"))

    from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSeq2SeqLM
    import torch

    censor_dir = os.path.join(checkpoint_dir, "censor")
    halluc_dir = os.path.join(checkpoint_dir, "hallucinator")

    censor_tok = AutoTokenizer.from_pretrained(censor_dir)
    censor_model = AutoModelForTokenClassification.from_pretrained(censor_dir).to(device)
    censor_model.eval()

    halluc_tok = AutoTokenizer.from_pretrained(halluc_dir)
    halluc_model = AutoModelForSeq2SeqLM.from_pretrained(halluc_dir).to(device)
    halluc_model.eval()

    return censor_model, censor_tok, halluc_model, halluc_tok


def anonymize_model1(text, censor_model, censor_tok, halluc_model, halluc_tok, device="cpu"):
    """Run model1 pipeline: NER → mask → fill."""
    import torch

    encoding = censor_tok(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        logits = censor_model(**encoding).logits
    preds = torch.argmax(logits, dim=-1)[0].tolist()

    tokens = censor_tok.convert_ids_to_tokens(encoding["input_ids"][0])
    id2label = censor_model.config.id2label

    masked_tokens = []
    for tok, pred_id in zip(tokens, preds):
        label = id2label.get(pred_id, "O")
        if label.startswith("B-"):
            etype = label[2:]
            masked_tokens.append(f"[{etype}]")
        elif label.startswith("I-"):
            continue
        else:
            if tok.startswith("##"):
                if masked_tokens:
                    masked_tokens[-1] += tok[2:]
                else:
                    masked_tokens.append(tok[2:])
            else:
                masked_tokens.append(tok)

    masked_text = " ".join(masked_tokens)
    prompt = "Replace PII placeholders with realistic fake entities: " + masked_text

    input_ids = halluc_tok(prompt, return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)
    with torch.no_grad():
        output = halluc_model.generate(input_ids, max_new_tokens=256, num_beams=4)
    result = halluc_tok.decode(output[0], skip_special_tokens=True)
    return result


def main():
    parser = argparse.ArgumentParser(description="SAHA-AL pipeline_maskfill inference")
    parser.add_argument("--gold", required=True)
    parser.add_argument("--model", type=int, required=True, choices=[1, 2],
                        help="Model number (1=baseline, 2=advanced/DP)")
    parser.add_argument("--checkpoint-dir", required=True,
                        help="Path to model checkpoint directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    records = load_jsonl(args.gold)
    if args.sample:
        records = records[:args.sample]

    print(f"Loading MaskFill Model {args.model} from {args.checkpoint_dir}...")

    if args.model == 1:
        censor_model, censor_tok, halluc_model, halluc_tok = load_model1(args.checkpoint_dir, device)
        anonymize_fn = lambda text: anonymize_model1(text, censor_model, censor_tok, halluc_model, halluc_tok, device)
    elif args.model == 2:
        censor_model, censor_tok, halluc_model, halluc_tok = load_model1(args.checkpoint_dir, device)
        anonymize_fn = lambda text: anonymize_model1(text, censor_model, censor_tok, halluc_model, halluc_tok, device)

    print(f"Running on {len(records)} records...")
    predictions = []
    for i, rec in enumerate(records):
        text = rec.get("original_text", "")
        try:
            anon = anonymize_fn(text)
        except Exception as e:
            anon = text
            if i < 5:
                print(f"  [WARN] Failed on {rec['id']}: {e}")

        predictions.append({"id": rec["id"], "anonymized_text": anon})
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(records)}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_jsonl(predictions, args.output)
    print(f"Saved {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()

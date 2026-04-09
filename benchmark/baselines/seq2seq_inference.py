"""
SAHA-AL Benchmark — Seq2seq Model Inference
=============================================
Generates predictions from trained seq2seq checkpoints (BART, T5, Flan-T5, etc.).

Supports two checkpoint formats:
  A) Raw .pt files (model_state_dict inside a dict)
  B) HuggingFace model directories

Usage (local checkpoints):
  python -m baselines.seq2seq_inference \
      --gold data/test.jsonl \
      --model-name bart-base-pii \
      --base-model facebook/bart-base \
      --checkpoint path/to/best_model.pt \
      --output predictions/predictions_bart-base-pii.jsonl

Usage (HuggingFace repo):
  python -m baselines.seq2seq_inference \
      --gold data/test.jsonl \
      --model-name bart-base-pii \
      --base-model facebook/bart-base \
      --hf-repo JALAPENO11/pii_identification_and_anonymisations \
      --hf-filename checkpoints_pii_aware_loss/bart-base/best_model.pt \
      --prefix "" \
      --output predictions/predictions_bart-base-pii.jsonl
"""

import argparse
import json
import os

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


MODEL_PRESETS = {
    "t5-efficient-tiny-pii": {
        "base_model": "google/t5-efficient-tiny",
        "hf_filename": "checkpoints_pii_aware_loss/t5-efficient-tiny/best_model.pt",
        "prefix": "anonymize: ",
    },
    "t5-small-pii": {
        "base_model": "t5-small",
        "hf_filename": "checkpoints_pii_aware_loss/t5-small/best_model.pt",
        "prefix": "anonymize: ",
    },
    "flan-t5-small-pii": {
        "base_model": "google/flan-t5-small",
        "hf_filename": "checkpoints_pii_aware_loss/flan-t5-small/best_model.pt",
        "prefix": "Replace all personal identifiable information in the following text with realistic fake alternatives: ",
    },
    "bart-base-pii": {
        "base_model": "facebook/bart-base",
        "hf_filename": "checkpoints_pii_aware_loss/bart-base/best_model.pt",
        "prefix": "",
    },
    "distilbart-pii": {
        "base_model": "sshleifer/distilbart-cnn-6-6",
        "hf_filename": "checkpoints_pii_aware_loss/distilbart/best_model.pt",
        "prefix": "",
    },
    "flan-t5-base-qlora-pii": {
        "base_model": "google/flan-t5-base",
        "hf_filename": "checkpoints_pii_aware_loss/flan-t5-base-qlora/best_model.pt",
        "prefix": "Replace all personal identifiable information in the following text with realistic fake alternatives: ",
    },
}

DEFAULT_HF_REPO = "JALAPENO11/pii_identification_and_anonymisations"


def load_checkpoint(base_model_name, checkpoint_path, device):
    """Load base model + apply checkpoint weights."""
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        else:
            state_dict = checkpoint
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            model.load_state_dict(state_dict, strict=False)

    model = model.to(device)
    model.eval()
    return model, tokenizer


def run_inference(
    model, tokenizer, records, prefix="", batch_size=16,
    max_input_length=128, max_output_length=128, num_beams=4, device="cpu",
):
    """Run batched inference, return list of {id, anonymized_text}."""
    results = []
    for batch_start in tqdm(range(0, len(records), batch_size), desc="Inference"):
        batch = records[batch_start : batch_start + batch_size]
        input_texts = [prefix + rec.get("original_text", "") for rec in batch]

        inputs = tokenizer(
            input_texts, return_tensors="pt",
            max_length=max_input_length, truncation=True, padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_output_length,
                num_beams=num_beams,
                do_sample=False,
                early_stopping=True,
            )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        for rec, text in zip(batch, decoded):
            results.append({"id": rec["id"], "anonymized_text": text})

    return results


def main():
    parser = argparse.ArgumentParser(description="SAHA-AL seq2seq inference")
    parser.add_argument("--gold", required=True, help="Gold test.jsonl")
    parser.add_argument("--model-name", required=True, help="Model key (e.g. bart-base-pii) or custom name")
    parser.add_argument("--base-model", default=None, help="HF base model ID (auto from preset if omitted)")
    parser.add_argument("--checkpoint", default=None, help="Local checkpoint path")
    parser.add_argument("--hf-repo", default=None, help="HF repo for checkpoint download")
    parser.add_argument("--hf-filename", default=None, help="Filename inside HF repo")
    parser.add_argument("--prefix", default=None, help="Input prefix (auto from preset if omitted)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--output", required=True, help="Output predictions JSONL")
    parser.add_argument("--run-all-presets", action="store_true",
                        help="Ignore model-name/base-model/checkpoint; run all presets")
    args = parser.parse_args()

    with open(args.gold, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    print(f"Loaded {len(records)} records.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.run_all_presets:
        from huggingface_hub import hf_hub_download
        os.makedirs(os.path.dirname(args.output) or "predictions", exist_ok=True)
        for name, preset in MODEL_PRESETS.items():
            print(f"\n{'='*40} {name} {'='*40}")
            ckpt_path = hf_hub_download(repo_id=DEFAULT_HF_REPO, filename=preset["hf_filename"])
            model, tokenizer = load_checkpoint(preset["base_model"], ckpt_path, device)
            results = run_inference(
                model, tokenizer, records, prefix=preset["prefix"],
                batch_size=args.batch_size, num_beams=args.num_beams, device=device,
            )
            out_path = os.path.join(os.path.dirname(args.output) or "predictions", f"predictions_{name}.jsonl")
            with open(out_path, "w") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")
            print(f"Saved {len(results)} predictions to {out_path}")
            del model, tokenizer
            torch.cuda.empty_cache()
        return

    preset = MODEL_PRESETS.get(args.model_name, {})
    base_model = args.base_model or preset.get("base_model")
    prefix = args.prefix if args.prefix is not None else preset.get("prefix", "")

    if not base_model:
        parser.error(f"No base model for '{args.model_name}'. Use --base-model or a known preset.")

    checkpoint_path = args.checkpoint
    if not checkpoint_path and (args.hf_repo or preset.get("hf_filename")):
        from huggingface_hub import hf_hub_download
        repo = args.hf_repo or DEFAULT_HF_REPO
        filename = args.hf_filename or preset.get("hf_filename")
        if filename:
            checkpoint_path = hf_hub_download(repo_id=repo, filename=filename)

    model, tokenizer = load_checkpoint(base_model, checkpoint_path, device)
    print(f"Model: {base_model}, Params: {sum(p.numel() for p in model.parameters()):,}")

    results = run_inference(
        model, tokenizer, records, prefix=prefix,
        batch_size=args.batch_size, num_beams=args.num_beams, device=device,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(results)} predictions to {args.output}")


if __name__ == "__main__":
    main()

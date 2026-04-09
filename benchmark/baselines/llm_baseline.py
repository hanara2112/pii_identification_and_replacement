"""
SAHA-AL Benchmark — LLM Zero-Shot Anonymization Baseline
==========================================================
Uses an LLM to anonymize text in a single zero-shot prompt.

Supports both OpenAI-compatible APIs (GPT-4o-mini, Groq, etc.)
and local HuggingFace models (Qwen, Phi, Gemma) with 4-bit quantization.

Usage (API):
  OPENAI_API_KEY=... python -m baselines.llm_baseline \
      --gold data/test.jsonl --output predictions/predictions_gpt4omini.jsonl

Usage (local):
  python -m baselines.llm_baseline \
      --gold data/test.jsonl --output predictions/predictions_qwen7b.jsonl \
      --local --local-model Qwen/Qwen2.5-7B-Instruct
"""

import argparse
import json
import os
import time

from eval.utils import load_jsonl, save_jsonl


ANONYMIZE_PROMPT = """You are a text anonymization system. Replace ALL personally identifiable information (PII) in the following text with realistic fake alternatives of the same type. Preserve grammar, structure, and meaning. Do NOT add explanations — return only the anonymized text.

Text: {text}

Anonymized text:"""


def _load_local_model(model_name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading local model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    load_kwargs = {"device_map": "auto"}
    if torch.cuda.is_available():
        try:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
            )
            print("Using 4-bit quantization")
        except (ImportError, Exception):
            load_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Model loaded on {model.device}")
    return model, tokenizer


def anonymize_with_local_llm(text, model, tokenizer, max_new_tokens=256):
    import torch
    messages = [{"role": "user", "content": ANONYMIZE_PROMPT.format(text=text)}]
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        prompt = ANONYMIZE_PROMPT.format(text=text)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=0.3, do_sample=True, top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    result = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    if result.startswith("Anonymized text:"):
        result = result[len("Anonymized text:"):].strip()
    return result


def anonymize_with_api(text, client, model="gpt-4o-mini", max_retries=3, delay=0):
    if delay > 0:
        time.sleep(delay)
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": ANONYMIZE_PROMPT.format(text=text)}],
                temperature=0.3,
                max_tokens=512,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            wait = max(2 ** (attempt + 1), delay or 2)
            if attempt < max_retries - 1:
                time.sleep(wait)
            else:
                print(f"  [WARN] Failed after {max_retries} retries: {e}")
                return text


def main():
    parser = argparse.ArgumentParser(description="SAHA-AL LLM zero-shot baseline")
    parser.add_argument("--gold", required=True, help="Gold test.jsonl")
    parser.add_argument("--output", required=True, help="Output predictions JSONL")
    parser.add_argument("--model", default="gpt-4o-mini", help="API model name")
    parser.add_argument("--local", action="store_true", help="Use local HuggingFace model")
    parser.add_argument("--local-model", default="Qwen/Qwen2.5-7B-Instruct",
                        help="HuggingFace model ID for local inference")
    parser.add_argument("--sample", type=int, default=None,
                        help="Only process first N records (for cost/time control)")
    parser.add_argument("--delay", type=float, default=0,
                        help="Seconds to wait between API calls (for rate limiting)")
    args = parser.parse_args()

    records = load_jsonl(args.gold)
    if args.sample:
        records = records[:args.sample]

    local_model, local_tokenizer = None, None
    client = None

    if args.local:
        local_model, local_tokenizer = _load_local_model(args.local_model)
        print(f"Running LLM anonymization on {len(records)} records with {args.local_model}...")
    else:
        from openai import OpenAI
        client = OpenAI()
        print(f"Running LLM anonymization on {len(records)} records with {args.model}...")

    predictions = []
    t0 = time.time()
    for i, rec in enumerate(records):
        text = rec.get("original_text", "")
        if args.local:
            anon = anonymize_with_local_llm(text, local_model, local_tokenizer)
        else:
            anon = anonymize_with_api(text, client, model=args.model, delay=args.delay)
        predictions.append({"id": rec["id"], "anonymized_text": anon})

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(records) - i - 1) / rate
            print(f"  {i+1}/{len(records)} ({rate:.1f} rec/s, ETA {eta/60:.0f}m)")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_jsonl(predictions, args.output)
    print(f"Saved {len(predictions)} predictions to {args.output} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()

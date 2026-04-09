"""
SAHA-AL Benchmark — LLM Zero-Shot Anonymization Baseline
==========================================================
Uses GPT-4o-mini to anonymize text in a single zero-shot prompt.

Requires: OPENAI_API_KEY environment variable.

Usage:
  python -m baselines.llm_baseline \
      --gold data/test.jsonl \
      --output predictions/predictions_gpt4omini.jsonl \
      --sample 500
"""

import argparse
import json
import os
import time

from eval.utils import load_jsonl, save_jsonl


ANONYMIZE_PROMPT = """You are a text anonymization system. Replace ALL personally identifiable information (PII) in the following text with realistic fake alternatives of the same type. Preserve grammar, structure, and meaning. Do NOT add explanations — return only the anonymized text.

Text: {text}

Anonymized text:"""


def anonymize_with_llm(text, client, model="gpt-4o-mini", max_retries=3):
    """Call the OpenAI API to anonymize a single text."""
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
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  [WARN] Failed after {max_retries} retries: {e}")
                return text


def main():
    parser = argparse.ArgumentParser(description="SAHA-AL LLM zero-shot baseline")
    parser.add_argument("--gold", required=True, help="Gold test.jsonl")
    parser.add_argument("--output", required=True, help="Output predictions JSONL")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name")
    parser.add_argument("--sample", type=int, default=None,
                        help="Only process first N records (for cost control)")
    args = parser.parse_args()

    try:
        from openai import OpenAI
    except ImportError:
        print("pip install openai")
        exit(1)

    client = OpenAI()
    records = load_jsonl(args.gold)
    if args.sample:
        records = records[:args.sample]

    print(f"Running LLM anonymization on {len(records)} records with {args.model}...")

    predictions = []
    for i, rec in enumerate(records):
        text = rec.get("original_text", "")
        anon = anonymize_with_llm(text, client, model=args.model)
        predictions.append({"id": rec["id"], "anonymized_text": anon})

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(records)}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_jsonl(predictions, args.output)
    print(f"Saved {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()

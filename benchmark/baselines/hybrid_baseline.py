"""
SAHA-AL Benchmark — Hybrid Baseline (spaCy+regex detect → LLM replace)
========================================================================
Isolates detection quality vs generation quality by using spaCy + regex
for entity detection and an LLM for realistic replacement.

Supports both OpenAI-compatible APIs and local HuggingFace models.

Usage (local):
  python -m baselines.hybrid_baseline \
      --gold data/test.jsonl --output predictions/predictions_hybrid.jsonl \
      --local --local-model Qwen/Qwen2.5-7B-Instruct --save-spans

Usage (API):
  OPENAI_API_KEY=... python -m baselines.hybrid_baseline \
      --gold data/test.jsonl --output predictions/predictions_hybrid.jsonl
"""

import argparse
import os
import re
import time
from collections import OrderedDict

from eval.utils import load_jsonl, save_jsonl


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
])

REPLACE_PROMPT = """The following entities were detected as PII in this text:
{entity_list}

Original text: {text}

Replace ONLY the listed entities with realistic fake alternatives of the same type. Keep everything else exactly the same. Return only the anonymized text, nothing else."""


def _resolve_overlapping_spans(spans):
    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    resolved = []
    last_end = -1
    for span in spans:
        if span[0] >= last_end:
            resolved.append(span)
            last_end = span[1]
    return resolved


def detect_entities(text, nlp):
    """Detect entities using spaCy NER + regex patterns."""
    doc = nlp(text)
    spans = []
    for ent in doc.ents:
        if ent.label_ in ("PERSON", "ORG", "GPE", "LOC", "DATE", "FAC", "NORP", "EVENT"):
            spans.append((ent.start_char, ent.end_char, ent.label_, ent.text))
    for ent_type, pattern in REGEX_PATTERNS.items():
        for match in pattern.finditer(text):
            spans.append((match.start(), match.end(), ent_type, match.group()))
    return _resolve_overlapping_spans(spans)


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


def replace_with_local_llm(text, entities, model, tokenizer, max_new_tokens=256):
    import torch
    if not entities:
        return text

    entity_list = "\n".join(f'- "{e[3]}" (type: {e[2]})' for e in entities)
    prompt = REPLACE_PROMPT.format(entity_list=entity_list, text=text)

    messages = [{"role": "user", "content": prompt}]
    try:
        full_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        full_prompt = prompt

    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=0.3, do_sample=True, top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def replace_with_api(text, entities, client, model="gpt-4o-mini", max_retries=3):
    if not entities:
        return text

    entity_list = "\n".join(f'- "{e[3]}" (type: {e[2]})' for e in entities)
    prompt = REPLACE_PROMPT.format(entity_list=entity_list, text=text)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3, max_tokens=512,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  [WARN] LLM failed: {e}")
                return text


def main():
    parser = argparse.ArgumentParser(description="SAHA-AL hybrid baseline (spaCy+regex → LLM)")
    parser.add_argument("--gold", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--llm-model", default="gpt-4o-mini", help="API model name")
    parser.add_argument("--local", action="store_true", help="Use local HuggingFace model")
    parser.add_argument("--local-model", default="Qwen/Qwen2.5-7B-Instruct",
                        help="HuggingFace model ID for local inference")
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--save-spans", action="store_true")
    args = parser.parse_args()

    import spacy
    nlp = spacy.load("en_core_web_lg")

    local_model, local_tokenizer = None, None
    client = None

    if args.local:
        local_model, local_tokenizer = _load_local_model(args.local_model)
        model_label = args.local_model
    else:
        from openai import OpenAI
        client = OpenAI()
        model_label = args.llm_model

    records = load_jsonl(args.gold)
    if args.sample:
        records = records[:args.sample]

    print(f"Hybrid baseline: spaCy+regex detect → {model_label} replace on {len(records)} records...")

    predictions = []
    span_records = []
    t0 = time.time()

    for i, rec in enumerate(records):
        text = rec.get("original_text", "")
        entities = detect_entities(text, nlp)

        if args.local:
            anon = replace_with_local_llm(text, entities, local_model, local_tokenizer)
        else:
            anon = replace_with_api(text, entities, client, model=args.llm_model)

        predictions.append({"id": rec["id"], "anonymized_text": anon})
        if args.save_spans:
            span_records.append({
                "id": rec["id"],
                "detected_entities": [
                    {"start": s, "end": e, "type": t, "text": txt}
                    for s, e, t, txt in entities
                ],
            })

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(records) - i - 1) / rate
            print(f"  {i+1}/{len(records)} ({rate:.1f} rec/s, ETA {eta/60:.0f}m)")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_jsonl(predictions, args.output)
    print(f"Predictions saved to {args.output} ({time.time()-t0:.0f}s)")

    if args.save_spans:
        span_path = args.output.replace(".jsonl", "_spans.jsonl")
        save_jsonl(span_records, span_path)
        print(f"Spans saved to {span_path}")


if __name__ == "__main__":
    main()
